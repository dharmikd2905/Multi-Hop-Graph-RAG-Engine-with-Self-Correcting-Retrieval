"""
Phase 2 (structural half) + Phase 3 graph algorithms -- PRD 3.1, 4.2, 4.3.

Wraps a NetworkX DiGraph as the structural knowledge store:
- Nodes = normalized entities.
- Edges = predicate relationships, carrying `chunk_ids` provenance so any
  graph edge can be traced back to the source text chunk(s) it came from.
- Provides PageRank centrality and 2-hop BFS traversal used by the hybrid
  retrieval engine to compute the Composite Relevance Index (CRI).
"""
from __future__ import annotations

#Interacts with the host operating system's filesystem.like use for creating directories, checking if a file exists, etc.
#here we use it because we want to persist(save, load) the graph to disk and load it back later.
import os

#Serializes and deserializes Python object structures to and from binary bytecode streams.
import pickle

# DQ is Used in the bfs_k_hop method for efficient FIFO queue operations during breadth-first search traversal.
from collections import deque

# NetworkX is a Python library for creating, manipulating, and studying complex networks of nodes and edges. Here we use it to represent the knowledge graph as a directed graph (DiGraph).
"""
--> nx.DiGraph() creates the underlying directed graph structure storing nodes and weighted/attributed edges.

--> nx.pagerank(self.graph, alpha=damping) runs the PageRank algorithm over all entities in the graph to score structural centrality.

--> Provides exception handling (nx.PowerIterationFailedConvergence) when PageRank fails to converge on certain disconnected graph topologies.

--> Supplies methods like graph.successors(), graph.predecessors(), out_edges(), and in_edges() for exploring connections and reading provenance metadata (chunk_ids).
"""
import networkx as nx

from app.schemas.models import Chunk, Triple


class GraphStore:
    def __init__(self, persist_path: str | None = None) -> None:
        self.persist_path = persist_path
        self.graph: nx.DiGraph = nx.DiGraph()
        if persist_path and os.path.exists(persist_path):
            self.load(persist_path)

    # ------------------------------------------------------------------ build
    def add_triples(self, triples: list[Triple], chunk_id: str) -> None:
        for t in triples:
            s, p, o = t.as_tuple()
            if not s or not o:
                continue
            self.graph.add_node(s)
            self.graph.add_node(o)
            if self.graph.has_edge(s, o):
                self.graph[s][o]["predicates"].add(p)
                self.graph[s][o]["chunk_ids"].add(chunk_id)
            else:
                self.graph.add_edge(s, o, predicates={p}, chunk_ids={chunk_id})

    def ingest_chunks(self, chunks: list[Chunk]) -> None:
        for chunk in chunks:
            self.add_triples(chunk.triples, chunk.chunk_id)

    # -------------------------------------------------------------- algorithms
    def pagerank(self, damping: float = 0.85) -> dict[str, float]:
        """PR(u) = (1-d)/N + d * sum_{v in M(u)} PR(v)/L(v)  -- PRD 4.3 formula."""
        if self.graph.number_of_nodes() == 0:
            return {}
        try:
            return nx.pagerank(self.graph, alpha=damping)
        except nx.PowerIterationFailedConvergence:
            # Fall back to uniform scores rather than crashing retrieval.
            n = self.graph.number_of_nodes()
            return {node: 1.0 / n for node in self.graph.nodes}

    def find_seed_nodes(self, entities: list[str]) -> list[str]:
        normalized = {e.strip().lower() for e in entities}
        return [n for n in self.graph.nodes if n in normalized]

    def bfs_k_hop(self, seed_nodes: list[str], hops: int = 2) -> dict[str, int]:
        """
        Undirected-style k-hop BFS (traverses both successors and
        predecessors, since a relevant relation may point either way)
        from each seed node. Returns {node: min_hop_distance}.
        """
        visited: dict[str, int] = {}
        frontier: deque[tuple[str, int]] = deque()
        for seed in seed_nodes:
            if seed in self.graph:
                visited[seed] = 0
                frontier.append((seed, 0))

        while frontier:
            node, dist = frontier.popleft()
            if dist >= hops:
                continue
            neighbors = set(self.graph.successors(node)) | set(self.graph.predecessors(node))
            for nb in neighbors:
                if nb not in visited or visited[nb] > dist + 1:
                    visited[nb] = dist + 1
                    frontier.append((nb, dist + 1))
        return visited

    def chunk_ids_for_nodes(self, nodes: list[str]) -> dict[str, set[str]]:
        """Map each node -> set of chunk_ids provenance from its incident edges."""
        result: dict[str, set[str]] = {}
        for node in nodes:
            ids: set[str] = set()
            for _, _, data in self.graph.out_edges(node, data=True):
                ids |= data.get("chunk_ids", set())
            for _, _, data in self.graph.in_edges(node, data=True):
                ids |= data.get("chunk_ids", set())
            result[node] = ids
        return result

    # ---------------------------------------------------------------- persist
    def save(self, path: str | None = None) -> None:
        path = path or self.persist_path
        if not path:
            return
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.graph, f)

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            self.graph = pickle.load(f)

    def stats(self) -> dict[str, int]:
        return {"nodes": self.graph.number_of_nodes(), "edges": self.graph.number_of_edges()}
