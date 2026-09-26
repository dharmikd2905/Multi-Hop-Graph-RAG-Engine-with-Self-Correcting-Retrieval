"""
Evaluation Harness for Graph-Augmented Self-RAG Engine.

Compares Vector-Only Retrieval vs. Hybrid CRI-Ranked Retrieval across hand-written
multi-hop benchmark questions based on ingested architecture documentation.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
from typing import TypedDict

from app.core.config import get_settings
from app.core.graph_store import GraphStore
from app.core.ingestion import chunk_document, extract_triples_for_document
from app.core.llm_provider import get_chat_model
from app.core.retrieval import HybridRetriever, naive_entity_extraction
from app.core.vector_store import VectorStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("eval_harness")

# Benchmark Questions with Expected Source Chunks & Multi-hop Reasoning Details
BENCHMARK_QUESTIONS = [
    {
        "id": "Q1",
        "question": "How does Component A indirectly affect the metrics freshness of Component C in the microservices architecture?",
        "expected_chunks": ["doc1_microservices_1_53f89f42", "doc1_microservices_2_3512b0ff"],
        "primary_target": "doc1_microservices_1_53f89f42",
        "reasoning": "Component A (API Gateway) -> Auth Service -> Message Queue (Component B) -> Analytics Pipeline (Component C). Chunk 1 contains Component C details without mentioning Component A."
    },
    {
        "id": "Q2",
        "question": "What cascade of events occurs when Redis-X experiences a memory overflow in the Omega Ecosystem?",
        "expected_chunks": ["doc2_omega_3_1dcece1e", "doc2_omega_4_8e11ec97"],
        "primary_target": "doc2_omega_4_8e11ec97",
        "reasoning": "Redis-X overflow -> Engine Alpha cache degradation -> MongoDB Atlas latency spike -> Guardrail-Zero -> Circuit Breaker Delta -> Pulsar Event Bus Hard Drop Mode -> Project Aurora session rollback."
    },
    {
        "id": "Q3",
        "question": "How does a failure in Engine Alpha's caching layer lead to user session rollbacks in Project Aurora?",
        "expected_chunks": ["doc2_omega_1_4b83c790", "doc2_omega_3_1dcece1e", "doc2_omega_4_8e11ec97"],
        "primary_target": "doc2_omega_4_8e11ec97",
        "reasoning": "Engine Alpha (Redis-X) -> Guardrail-Zero -> Circuit Breaker Delta -> Pulsar Hard Drop -> Project Aurora session rollback. Chunk 4 has Project Aurora rollback details."
    },
    {
        "id": "Q4",
        "question": "Which reporting system relies on metrics derived from user login events published to Kafka?",
        "expected_chunks": ["doc1_microservices_0_3386f4da", "doc1_microservices_1_53f89f42"],
        "primary_target": "doc1_microservices_1_53f89f42",
        "reasoning": "Login events -> Auth Service -> Message Queue (Kafka) [Chunk 0] -> Analytics Pipeline -> Data Warehouse -> Reporting Service [Chunk 1]."
    },
    {
        "id": "Q5",
        "question": "How are entities scored by PageRank in NetworkX eventually stored in ChromaDB?",
        "expected_chunks": ["doc2_omega_2_f678b2f8"],
        "primary_target": "doc2_omega_2_f678b2f8",
        "reasoning": "Engine Beta -> transaction logs -> NetworkX -> PageRank > 0.15 -> promoted to ChromaDB as dense vectors."
    },
    {
        "id": "Q6",
        "question": "What monitoring platform measures the latency impact on Project Aurora when Circuit Breaker Delta is active?",
        "expected_chunks": ["doc2_omega_4_8e11ec97"],
        "primary_target": "doc2_omega_4_8e11ec97",
        "reasoning": "Circuit Breaker Delta -> Project Aurora -> Observability Dashboard -> LangSmith trace paths."
    },
    {
        "id": "Q7",
        "question": "If the Auth Service becomes unavailable, how does the API Gateway maintain operations, and what happens to downstream analytics?",
        "expected_chunks": ["doc1_microservices_0_3386f4da", "doc1_microservices_1_53f89f42"],
        "primary_target": "doc1_microservices_0_3386f4da",
        "reasoning": "Auth Service unavailable -> API Gateway fallback to cached tokens (5 min) -> delayed login events -> Analytics Pipeline impact."
    },
    {
        "id": "Q8",
        "question": "What mechanism connects Project Aurora to the execution engines, and how is flow controlled under heavy load?",
        "expected_chunks": ["doc2_omega_0_e08175e7"],
        "primary_target": "doc2_omega_0_e08175e7",
        "reasoning": "Project Aurora -> Pulsar Event Bus -> Core Processing Cluster (Engine Alpha/Beta), with Backpressure Flow Control (BFC)."
    },
    {
        "id": "Q9",
        "question": "How does data flow from unstructured transaction logs to the primary database and graph processing engines in Omega?",
        "expected_chunks": ["doc2_omega_1_4b83c790", "doc2_omega_2_f678b2f8"],
        "primary_target": "doc2_omega_2_f678b2f8",
        "reasoning": "MongoDB Atlas -> Engine Beta reads logs -> extracts triples -> NetworkX directed graph."
    },
]


def run_evaluation():
    settings = get_settings()

    # Use persistent eval storage directory
    eval_dir = "./data_eval_harness"
    if os.path.exists(eval_dir):
        shutil.rmtree(eval_dir)

    eval_settings = settings.model_copy(
        update={
            "chroma_persist_dir": f"{eval_dir}/chroma",
            "graph_persist_path": f"{eval_dir}/graph.gpickle",
        }
    )

    vector_store = VectorStore(eval_settings)
    graph_store = GraphStore(persist_path=eval_settings.graph_persist_path)
    llm = get_chat_model(eval_settings)

    print("==> Ingesting evaluation documents...")
    with open("sample_data/microservices_architecture.md", "r", encoding="utf-8") as f:
        doc1_text = f.read()
    chunks1 = chunk_document("doc1_microservices", "microservices_architecture.md", doc1_text, eval_settings)
    chunks1 = extract_triples_for_document(chunks1, llm)

    with open("test-data/test1.txt", "r", encoding="utf-8") as f:
        doc2_text = f.read()
    chunks2 = chunk_document("doc2_omega", "test1.txt", doc2_text, eval_settings)
    chunks2 = extract_triples_for_document(chunks2, llm)

    all_chunks = chunks1 + chunks2
    vector_store.add_chunks(all_chunks)
    graph_store.ingest_chunks(all_chunks)
    graph_store.save()

    print(f"Ingested {len(all_chunks)} chunks into vector store & knowledge graph.")
    print(f"Graph stats: {graph_store.stats()}")

    retriever = HybridRetriever(vector_store, graph_store, eval_settings)

    results = []

    print("\n" + "=" * 80)
    print("RUNNING EVALUATION: VECTOR-ONLY vs HYBRID CRI RETRIEVAL")
    print("=" * 80)

    for item in BENCHMARK_QUESTIONS:
        q_id = item["id"]
        q_text = item["question"]
        expected = item["expected_chunks"]
        primary_target = item["primary_target"]

        # 1. Vector-only retrieval (bypassing graph)
        vector_hits_raw = vector_store.similarity_search(q_text, top_k=5)
        vec_chunk_ids = [h["chunk_id"] for h in vector_hits_raw]

        # Check if primary target is in top-5 vector hits
        vec_hit = primary_target in vec_chunk_ids or any(c in vec_chunk_ids for c in expected)

        # 2. Hybrid CRI retrieval
        hybrid_hits_raw = retriever.retrieve(q_text, top_k=5)
        hybrid_chunk_ids = [h.chunk_id for h in hybrid_hits_raw[:5]]

        # Check if primary target is in top-5 hybrid hits
        hybrid_hit = primary_target in hybrid_chunk_ids or any(c in hybrid_chunk_ids for c in expected)

        # Graph entity extraction & seeds
        seed_nodes = naive_entity_extraction(q_text, graph_store)
        hop_map = graph_store.bfs_k_hop(seed_nodes, hops=2)

        results.append(
            {
                "id": q_id,
                "question": q_text,
                "expected_chunks": expected,
                "primary_target": primary_target,
                "vector_top5": vec_chunk_ids,
                "vector_hit": vec_hit,
                "hybrid_top5": hybrid_chunk_ids,
                "hybrid_hit": hybrid_hit,
                "seed_nodes": seed_nodes,
                "visited_entities_count": len(hop_map),
                "reasoning": item["reasoning"],
            }
        )

        print(f"\n[{q_id}] {q_text}")
        print(f"  Target Chunks: {expected}")
        print(f"  Seed Nodes  : {seed_nodes}")
        print(f"  Vector-Only Top-5 : {vec_chunk_ids} (Hit Target? {vec_hit})")
        print(f"  Hybrid CRI  Top-5 : {hybrid_chunk_ids} (Hit Target? {hybrid_hit})")

    # Summary Statistics
    vec_success = sum(1 for r in results if r["vector_hit"])
    hybrid_success = sum(1 for r in results if r["hybrid_hit"])
    win_cases = [r for r in results if r["hybrid_hit"] and not r["vector_hit"]]

    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    print(f"Total Questions: {len(BENCHMARK_QUESTIONS)}")
    print(f"Vector-Only Accuracy (Top-5): {vec_success}/{len(BENCHMARK_QUESTIONS)} ({vec_success/len(BENCHMARK_QUESTIONS)*100:.1f}%)")
    print(f"Hybrid CRI  Accuracy (Top-5): {hybrid_success}/{len(BENCHMARK_QUESTIONS)} ({hybrid_success/len(BENCHMARK_QUESTIONS)*100:.1f}%)")
    print(f"Hybrid Win Cases (Hybrid OK, Vector Fail): {len(win_cases)}")

    # Write detailed output JSON for report generator
    with open("eval_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    run_evaluation()
