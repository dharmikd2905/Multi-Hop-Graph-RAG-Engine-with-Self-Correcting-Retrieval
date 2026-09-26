"""
Unit tests for the deterministic, LLM-free parts of the system:
chunking, graph construction, PageRank, BFS traversal, and CRI scoring.

Run with: pytest -v
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import Settings
from app.core.graph_store import GraphStore
from app.core.ingestion import chunk_document
from app.schemas.models import Chunk, Triple


def make_settings(**overrides) -> Settings:
    return Settings(**overrides)


def test_chunking_respects_size_and_overlap():
    settings = make_settings(chunk_size=100, chunk_overlap=20)
    text = "A" * 500
    chunks = chunk_document("doc1", "source.txt", text, settings)
    assert len(chunks) > 1
    for c in chunks:
        assert len(c.text) <= 100
    assert all(c.doc_id == "doc1" for c in chunks)
    assert len({c.chunk_id for c in chunks}) == len(chunks)  # unique ids


def test_chunking_empty_text_returns_no_chunks():
    settings = make_settings()
    chunks = chunk_document("doc1", "source.txt", "", settings)
    assert chunks == []


def test_graph_store_add_triples_creates_nodes_and_edges():
    gs = GraphStore()
    triples = [Triple(subject="FastAPI", predicate="depends_on", object="Starlette")]
    gs.add_triples(triples, chunk_id="c1")
    assert gs.graph.number_of_nodes() == 2
    assert gs.graph.number_of_edges() == 1
    assert gs.graph.has_edge("fastapi", "starlette")
    assert "c1" in gs.graph["fastapi"]["starlette"]["chunk_ids"]


def test_graph_store_merges_parallel_edges():
    gs = GraphStore()
    gs.add_triples([Triple(subject="A", predicate="causes", object="B")], chunk_id="c1")
    gs.add_triples([Triple(subject="A", predicate="influences", object="B")], chunk_id="c2")
    assert gs.graph.number_of_edges() == 1  # merged, not duplicated
    edge = gs.graph["a"]["b"]
    assert edge["predicates"] == {"causes", "influences"}
    assert edge["chunk_ids"] == {"c1", "c2"}


def test_pagerank_returns_normalized_scores_summing_near_one():
    gs = GraphStore()
    gs.add_triples(
        [
            Triple(subject="A", predicate="causes", object="B"),
            Triple(subject="B", predicate="causes", object="C"),
            Triple(subject="C", predicate="causes", object="A"),
        ],
        chunk_id="c1",
    )
    scores = gs.pagerank()
    assert set(scores.keys()) == {"a", "b", "c"}
    assert abs(sum(scores.values()) - 1.0) < 1e-6


def test_pagerank_empty_graph_returns_empty_dict():
    gs = GraphStore()
    assert gs.pagerank() == {}


def test_bfs_two_hop_discovers_indirect_relation():
    """
    Models the PRD's canonical example: 'How does Component A indirectly
    affect Component C through Component B?' -- A -> B -> C should be
    reachable within 2 hops from seed node A, even though A and C have
    no direct edge (and thus no shared vector-similarity signal).
    """
    gs = GraphStore()
    gs.add_triples(
        [
            Triple(subject="Component A", predicate="affects", object="Component B"),
            Triple(subject="Component B", predicate="affects", object="Component C"),
        ],
        chunk_id="c1",
    )
    hop_map = gs.bfs_k_hop(["component a"], hops=2)
    assert hop_map["component a"] == 0
    assert hop_map["component b"] == 1
    assert hop_map["component c"] == 2
    assert not gs.graph.has_edge("component a", "component c")  # confirms indirect-only


def test_bfs_respects_hop_limit():
    gs = GraphStore()
    gs.add_triples(
        [
            Triple(subject="A", predicate="r", object="B"),
            Triple(subject="B", predicate="r", object="C"),
            Triple(subject="C", predicate="r", object="D"),
        ],
        chunk_id="c1",
    )
    hop_map = gs.bfs_k_hop(["a"], hops=1)
    assert "b" in hop_map
    assert "c" not in hop_map  # beyond 1 hop
    assert "d" not in hop_map


def test_chunk_ids_for_nodes_returns_provenance():
    gs = GraphStore()
    gs.add_triples([Triple(subject="X", predicate="r", object="Y")], chunk_id="chunk_42")
    result = gs.chunk_ids_for_nodes(["x", "y"])
    assert result["x"] == {"chunk_42"}
    assert result["y"] == {"chunk_42"}


def test_ingest_chunks_populates_graph_from_multiple_chunks():
    gs = GraphStore()
    chunks = [
        Chunk(chunk_id="c1", text="t1", source="s", doc_id="d1", triples=[Triple(subject="A", predicate="r", object="B")]),
        Chunk(chunk_id="c2", text="t2", source="s", doc_id="d1", triples=[Triple(subject="B", predicate="r", object="C")]),
    ]
    gs.ingest_chunks(chunks)
    assert gs.stats() == {"nodes": 3, "edges": 2}
