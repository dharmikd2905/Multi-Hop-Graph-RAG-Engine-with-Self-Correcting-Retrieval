"""
Interactive visualization layer (PRD 2.1: "interactive visualization scripts
using Streamlit and PyVis to render real-time node connections during graph
retrieval").

Run with:  streamlit run streamlit_app.py

This talks directly to the same core modules used by the FastAPI service
(no HTTP hop needed) so it also works fully offline against Ollama.
"""

from __future__ import annotations

import streamlit as st
from pyvis.network import Network
import tempfile
import os

from app.core.config import get_settings
from app.core.graph_store import GraphStore
from app.core.ingestion import chunk_document, extract_triples_for_document
from app.core.llm_provider import get_chat_model
from app.core.retrieval import HybridRetriever, naive_entity_extraction
from app.core.self_correction_graph import run_self_correcting_query
from app.core.vector_store import VectorStore

st.set_page_config(page_title="Graph-Augmented Self-RAG Engine", layout="wide")

settings = get_settings()


@st.cache_resource
def get_stores():
    vs = VectorStore(settings)
    gs = GraphStore(persist_path=settings.graph_persist_path)
    return vs, gs


vector_store, graph_store = get_stores()
retriever = HybridRetriever(vector_store, graph_store, settings)

st.title("🕸️ Graph-Augmented Self-RAG Engine")
st.caption(
    f"LLM provider: `{settings.llm_provider}` · Graph nodes: {graph_store.stats()['nodes']} · "
    f"Graph edges: {graph_store.stats()['edges']} · Vector chunks: {vector_store.count()}"
)

tab_ingest, tab_query, tab_graph = st.tabs(["📥 Ingest", "🔎 Query", "🌐 Full Graph"])

with tab_ingest:
    st.subheader("Ingest a document")
    doc_id = st.text_input("Document ID", value="demo_doc_1")
    source_name = st.text_input("Source name", value="demo_source.txt")
    text = st.text_area("Paste document text", height=250)
    if st.button("Ingest", type="primary"):
        with st.spinner("Chunking, extracting triples, and indexing..."):
            llm = get_chat_model(settings)
            chunks = chunk_document(doc_id, source_name, text, settings)
            chunks = extract_triples_for_document(chunks, llm)
            vector_store.add_chunks(chunks)
            graph_store.ingest_chunks(chunks)
            graph_store.save()
        st.success(f"Ingested {len(chunks)} chunks, {sum(len(c.triples) for c in chunks)} triples.")
        st.rerun()

with tab_query:
    st.subheader("Ask a multi-hop question")
    q = st.text_input("Query", value="How does Component A indirectly affect Component C?")
    if st.button("Run Self-RAG pipeline", type="primary"):
        with st.spinner("Retrieving, generating, and self-evaluating..."):
            llm = get_chat_model(settings)
            final_state = run_self_correcting_query(q, retriever, llm, settings)

        st.markdown("### Answer")
        st.write(final_state["answer"])

        ev = final_state["evaluation"]
        col1, col2, col3 = st.columns(3)
        col1.metric("Hallucination score", f"{ev.hallucination_score:.2f}" if ev else "n/a")
        col2.metric("Relevance score", f"{ev.relevance_score:.2f}" if ev else "n/a")
        col3.metric("Retries used", final_state["retries_used"])

        st.markdown("### Retrieved context (ranked by Composite Relevance Index)")
        for c in final_state["contexts"]:
            with st.expander(f"[{c.chunk_id}] CRI={c.composite_relevance_index:.4f} · hops={c.graph_hops_used}"):
                st.write(c.text)
                st.caption(f"vector_score={c.vector_score} · pagerank={c.pagerank_score} · source={c.source}")

        st.markdown("### Execution trace")
        for step in final_state["trace"]:
            st.text(f"[{step.node}] {step.detail}")

        # --- Graph visualization of seed nodes + 2-hop neighborhood -------
        seed_nodes = naive_entity_extraction(final_state["current_query"], graph_store)
        hop_map = graph_store.bfs_k_hop(seed_nodes, hops=settings.graph_hops)
        if hop_map:
            st.markdown("### Retrieval subgraph (seed nodes + k-hop neighborhood)")
            net = Network(height="500px", width="100%", directed=True, bgcolor="#0e1117", font_color="white")
            pr_scores = graph_store.pagerank(damping=settings.pagerank_damping)
            for node, hop in hop_map.items():
                color = "#e74c3c" if hop == 0 else ("#3498db" if hop == 1 else "#2ecc71")
                size = 15 + 100 * pr_scores.get(node, 0)
                net.add_node(node, label=node, color=color, size=size, title=f"PageRank={pr_scores.get(node, 0):.4f}")
            for u, v, data in graph_store.graph.edges(data=True):
                if u in hop_map and v in hop_map:
                    net.add_edge(u, v, title=", ".join(data.get("predicates", [])))
            output_file = os.path.join(tempfile.gettempdir(), "graph_viz.html")
            net.save_graph(output_file)
            with open(output_file, "r", encoding="utf-8") as f:
                st.components.v1.html(f.read(), height=520)
        else:
            st.info("No graph entities matched this query -- answer relied purely on vector search.")

with tab_graph:
    st.subheader("Full knowledge graph")
    if graph_store.graph.number_of_nodes() == 0:
        st.info("Graph is empty. Ingest a document first.")
    else:
        pr_scores = graph_store.pagerank(damping=settings.pagerank_damping)
        net = Network(height="600px", width="100%", directed=True, bgcolor="#0e1117", font_color="white")
        for node in graph_store.graph.nodes:
            size = 15 + 150 * pr_scores.get(node, 0)
            net.add_node(node, label=node, size=size, title=f"PageRank={pr_scores.get(node, 0):.4f}")
        for u, v, data in graph_store.graph.edges(data=True):
            net.add_edge(u, v, title=", ".join(data.get("predicates", [])))
        # net.save_graph("/tmp/full_graph_viz.html")
        output_file = os.path.join(tempfile.gettempdir(), "full_graph_viz.html")
        net.save_graph(output_file)
        with open(output_file, "r", encoding="utf-8") as f:
            st.components.v1.html(f.read(), height=620)
