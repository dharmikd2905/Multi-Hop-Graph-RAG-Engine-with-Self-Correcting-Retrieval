"""
Pydantic schemas shared across the ingestion, retrieval, and API layers.

The Triple / TripleExtractionResult pair is deliberately used as the
*structured output schema* passed to the LLM (function-calling /
`with_structured_output`) so extraction is deterministic and validated
per PRD 4.1 ("Prompts an LLM with Pydantic schema validation").
"""
from __future__ import annotations

from typing import Literal

#this is used to define the data models for the application. Pydantic is a library that provides data validation.ansures that the data being passed around in the application conforms to the expected structure and types.
from pydantic import BaseModel, Field


class Triple(BaseModel):
    """A single normalized Knowledge Triple: (Subject, Predicate, Object)."""

    subject: str = Field(..., description="Normalized entity, e.g. 'FastAPI'")
    predicate: str = Field(..., description="Normalized relation, e.g. 'depends_on'")
    object: str = Field(..., description="Normalized entity, e.g. 'Starlette'")

    def as_tuple(self) -> tuple[str, str, str]:
        return (self.subject.strip().lower(), self.predicate.strip().lower(), self.object.strip().lower())

#this class is pydantic model that defines the structure of the output.
class TripleExtractionResult(BaseModel):
    """Structured output contract for the extraction LLM call."""

    triples: list[Triple] = Field(default_factory=list)


class Chunk(BaseModel):
    chunk_id: str
    text: str
    source: str
    doc_id: str
    triples: list[Triple] = Field(default_factory=list)


class IngestRequest(BaseModel):
    doc_id: str
    source_name: str
    text: str


class IngestResponse(BaseModel):
    doc_id: str
    chunks_created: int
    triples_extracted: int
    graph_nodes: int
    graph_edges: int


class RetrievedContext(BaseModel):
    chunk_id: str
    text: str
    source: str
    vector_score: float
    pagerank_score: float
    composite_relevance_index: float
    graph_hops_used: int = 0


class EvaluationResult(BaseModel):
    hallucination_score: float = Field(..., ge=0, le=1, description="0 = fully grounded, 1 = fully hallucinated")
    relevance_score: float = Field(..., ge=0, le=1, description="0 = irrelevant, 1 = fully resolves the query")
    passed: bool
    reasoning: str


class QueryRequest(BaseModel):
    query: str
    stream: bool = True
    top_k: int | None = None


class QueryTraceStep(BaseModel):
    node: Literal["retrieve", "generate", "evaluate", "rewrite"]
    detail: str


class QueryResponse(BaseModel):
    query: str
    final_query: str
    answer: str
    contexts: list[RetrievedContext]
    evaluation: EvaluationResult
    retries_used: int
    trace: list[QueryTraceStep]
    langsmith_trace_url: str | None = None
