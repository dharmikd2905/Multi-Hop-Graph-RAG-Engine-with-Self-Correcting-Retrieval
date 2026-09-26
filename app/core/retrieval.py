"""
Phase 3 -- Algorithmic Hybrid Retrieval & Graph Centrality Scoring (PRD 4.3).

Pipeline:
 1. Vector Match       -> top-k dense chunks from ChromaDB (cosine similarity).
 2. Seed Entity ID     -> naive entity extraction from the query, matched
                           against graph node names.
 3. Graph Traversal    -> 2-hop BFS from seed nodes to surface relationally
                           connected entities missed by vector search.
 4. Algorithmic Ranking -> PageRank over the full graph; each candidate
                           chunk's final score is:

        CRI(chunk) = vector_score(chunk) * (1 + mean_pagerank(entities_in_chunk))

    which multiplies vector similarity by graph centrality, per PRD 4.3.
    Chunks reached ONLY via graph traversal (no vector hit) are still
    included with a base vector_score of 0, so purely structural multi-hop
    context can surface even when semantically dissimilar to the raw query.
"""
from __future__ import annotations

import re

from app.core.config import Settings, get_settings
from app.core.graph_store import GraphStore
from app.core.vector_store import VectorStore
from app.schemas.models import RetrievedContext

_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "how", "does", "do", "what",
    "why", "when", "where", "which", "who", "to", "of", "in", "on", "for",
    "and", "or", "with", "affect", "indirectly", "through", "via",
}


def naive_entity_extraction(query: str, graph: GraphStore) -> list[str]:
    """
    Cheap, dependency-free entity spotting: tokenize the query and keep
    tokens/phrases (up to 3-grams) that exactly match a known graph node.
    This avoids an extra LLM round-trip for the common case; a production
    system could swap this for spaCy NER or an LLM extraction call.
    """
    tokens = [t for t in re.findall(r"[A-Za-z0-9_]+", query.lower()) if t not in _STOPWORDS]
    candidates: set[str] = set()
    for n in (1, 2, 3):
        for i in range(len(tokens) - n + 1):
            candidates.add(" ".join(tokens[i : i + n]))
    return [c for c in candidates if c in graph.graph.nodes]


class HybridRetriever:
    def __init__(self, vector_store: VectorStore, graph_store: GraphStore, settings: Settings | None = None) -> None:
        self.vector_store = vector_store
        self.graph_store = graph_store
        self.settings = settings or get_settings()

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedContext]:
        settings = self.settings
        top_k = top_k or settings.vector_top_k

        # 1. Vector match
        vector_hits = self.vector_store.similarity_search(query, top_k=top_k)
        chunk_pool: dict[str, dict] = {h["chunk_id"]: h for h in vector_hits}

        # 2. Seed entity identification
        seed_nodes = naive_entity_extraction(query, self.graph_store)

        # 3. Graph traversal (2-hop BFS)
        hop_distances = self.graph_store.bfs_k_hop(seed_nodes, hops=settings.graph_hops)
        graph_nodes = list(hop_distances.keys())
        node_to_chunks = self.graph_store.chunk_ids_for_nodes(graph_nodes)

        # Pull in chunks reached purely through the graph (not already in the vector pool).
        # We don't have chunk text for graph-only hits unless it was previously
        # indexed in the vector store, so we look them up there.
        graph_only_ids = set()
        for node, cids in node_to_chunks.items():
            graph_only_ids |= cids
        graph_only_ids -= chunk_pool.keys()

        if graph_only_ids:
            fetched = self.vector_store.collection.get(ids=list(graph_only_ids))
            for cid, text, meta in zip(fetched.get("ids", []), fetched.get("documents", []), fetched.get("metadatas", [])):
                chunk_pool[cid] = {
                    "chunk_id": cid,
                    "text": text,
                    "source": (meta or {}).get("source", "unknown"),
                    "doc_id": (meta or {}).get("doc_id", "unknown"),
                    "vector_score": 0.0,
                }

        # 4. PageRank + Composite Relevance Index
        pr_scores = self.graph_store.pagerank(damping=settings.pagerank_damping)

        # Map chunk_id -> set of graph nodes that cite it, and min hop distance
        chunk_to_nodes: dict[str, set[str]] = {}
        chunk_to_hop: dict[str, int] = {}
        for node, cids in node_to_chunks.items():
            hop = hop_distances.get(node, 99)
            for cid in cids:
                chunk_to_nodes.setdefault(cid, set()).add(node)
                chunk_to_hop[cid] = min(chunk_to_hop.get(cid, 99), hop)

        results: list[RetrievedContext] = []
        for cid, chunk in chunk_pool.items():
            nodes_for_chunk = chunk_to_nodes.get(cid, set())
            if nodes_for_chunk:
                mean_pr = sum(pr_scores.get(n, 0.0) for n in nodes_for_chunk) / len(nodes_for_chunk)
            else:
                mean_pr = 0.0

            vector_score = chunk["vector_score"]
            cri = vector_score * (1 + mean_pr) if vector_score > 0 else mean_pr

            results.append(
                RetrievedContext(
                    chunk_id=cid,
                    text=chunk["text"],
                    source=chunk["source"],
                    vector_score=round(vector_score, 4),
                    pagerank_score=round(mean_pr, 6),
                    composite_relevance_index=round(cri, 6),
                    graph_hops_used=chunk_to_hop.get(cid, 0),
                )
            )

        results.sort(key=lambda r: r.composite_relevance_index, reverse=True)
        return results[: settings.max_context_chunks]
