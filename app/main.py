"""
Phase 5 -- API Gateway & Streaming Output (PRD 4.5).

    POST /api/v1/ingest  -> document upload, chunking, triple extraction, dual indexing.
    POST /api/v1/query   -> multi-hop retrieval + self-correction pipeline, streamed answer.
    GET  /api/v1/health  -> liveness + index stats.
"""
from __future__ import annotations

import logging
import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

#get_settings is a object that contains all the .env variables.
from app.core.config import get_settings

from app.core.graph_store import GraphStore
from app.core.ingestion import chunk_document, extract_triples_for_document
from app.core.llm_provider import get_chat_model
from app.core.retrieval import HybridRetriever
from app.core.self_correction_graph import run_self_correcting_query
from app.core.vector_store import VectorStore
from app.schemas.models import (
    EvaluationResult,
    IngestRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("graph_rag_engine")

settings = get_settings()

if settings.langchain_tracing_v2:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = settings.langchain_project
    if settings.langchain_api_key:
        os.environ["LANGCHAIN_API_KEY"] = settings.langchain_api_key

app = FastAPI(title=settings.api_title, version=settings.api_version)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Singletons -- shared graph + vector store across requests, persisted to disk.
vector_store = VectorStore(settings)
graph_store = GraphStore(persist_path=settings.graph_persist_path)
retriever = HybridRetriever(vector_store, graph_store, settings)


@app.get("/api/v1/health")
def health() -> dict:
    return {
        "status": "ok",
        "llm_provider": settings.llm_provider,
        "graph": graph_store.stats(),
        "vector_chunks": vector_store.count(),
    }


@app.post("/api/v1/ingest", response_model=IngestResponse)
def ingest(request: IngestRequest) -> IngestResponse:
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="text must not be empty")

    llm = get_chat_model(settings)

    chunks = chunk_document(request.doc_id, request.source_name, request.text, settings)
    chunks = extract_triples_for_document(chunks, llm)

    vector_store.add_chunks(chunks)
    graph_store.ingest_chunks(chunks)
    graph_store.save()

    total_triples = sum(len(c.triples) for c in chunks)
    stats = graph_store.stats()

    logger.info("Ingested doc_id=%s chunks=%d triples=%d", request.doc_id, len(chunks), total_triples)

    return IngestResponse(
        doc_id=request.doc_id,
        chunks_created=len(chunks),
        triples_extracted=total_triples,
        graph_nodes=stats["nodes"],
        graph_edges=stats["edges"],
    )


def _run_pipeline(query: str, top_k: int | None):
    llm = get_chat_model(settings)
    local_settings = settings
    if top_k:
        local_settings = settings.model_copy(update={"vector_top_k": top_k})
        local_retriever = HybridRetriever(vector_store, graph_store, local_settings)
    else:
        local_retriever = retriever
    return run_self_correcting_query(query, local_retriever, llm, local_settings)


@app.post("/api/v1/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="query must not be empty")

    final_state = _run_pipeline(request.query, request.top_k)
    evaluation: EvaluationResult = final_state["evaluation"] or EvaluationResult(
        hallucination_score=1.0, relevance_score=0.0, passed=False, reasoning="No evaluation produced."
    )

    return QueryResponse(
        query=request.query,
        final_query=final_state["current_query"],
        answer=final_state["answer"],
        contexts=final_state["contexts"],
        evaluation=evaluation,
        retries_used=final_state["retries_used"],
        trace=final_state["trace"],
        langsmith_trace_url=None,
    )


@app.post("/api/v1/query/stream")
def query_stream(request: QueryRequest):
    """
    Token-by-token streaming variant. The self-correction loop must fully
    resolve (retrieve -> generate -> evaluate -> [rewrite loop]) before the
    *final* accepted answer is streamed back, since we can't guarantee the
    first draft passes the hallucination/relevance guardrail.
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="query must not be empty")

    final_state = _run_pipeline(request.query, request.top_k)
    answer = final_state["answer"]

    def token_stream():
        for word in answer.split(" "):
            yield word + " "

    return StreamingResponse(token_stream(), media_type="text/plain")
