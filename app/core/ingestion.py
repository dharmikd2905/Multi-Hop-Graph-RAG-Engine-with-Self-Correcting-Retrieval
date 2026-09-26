"""
Phase 1 -- Ingestion & Knowledge Extraction Pipeline (PRD 4.1).

1. Splits raw text with RecursiveCharacterTextSplitter (800 / 150, per PRD).
2. Extracts normalized (Subject, Predicate, Object) triples per chunk using
   an LLM constrained to the `TripleExtractionResult` Pydantic schema.
"""
from __future__ import annotations

import hashlib
import logging

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import Settings, get_settings
from app.schemas.models import Chunk, Triple, TripleExtractionResult

logger = logging.getLogger(__name__)

TRIPLE_EXTRACTION_SYSTEM_PROMPT = """You are a precise knowledge-graph extraction engine.
Read the text chunk and extract factual Knowledge Triples of the form
(Subject, Predicate, Object) that capture relationships between entities,
components, systems, or concepts mentioned in the text.

Rules:
- Normalize entity names (consistent casing/naming across triples).
- Predicates should be short, snake_case verb phrases (e.g. "depends_on", "causes", "part_of").
- Only extract triples explicitly supported by the text. Do not infer facts not present.
- If no clear relational facts exist in the chunk, return an empty list.
- Extract at most 8 triples per chunk.
"""

#function to split the document into chunks of text and return a list of Chunk objects. Each Chunk object contains a unique chunk_id, the text of the chunk, the source name, and the document ID.
def chunk_document(doc_id: str, source_name: str, text: str, settings: Settings | None = None) -> list[Chunk]:
    settings = settings or get_settings()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    raw_chunks = splitter.split_text(text)

    #this line defines an empty list called chunks that will be used to store the Chunk objects created from the raw_chunks. The type hint list[Chunk] indicates that this list is expected to contain objects of the Chunk class. 
    chunks: list[Chunk] = []
    #this loop is for generating Chunk objects from the raw chunks.
    for i, raw in enumerate(raw_chunks):
        #this line generates a unique chunk_id for each chunk.here we use puthon's hashlib library to generate a SHA-1 hash of the raw chunk. this will become chunk_id.
        chunk_id = f"{doc_id}_{i}_{hashlib.sha1(raw.encode()).hexdigest()[:8]}"
        #now we append the chunk which is exactly look like Chunk(chunk_id=chunk_id, text=raw, source=source_name, doc_id=doc_id) defined as in models.py file.
        chunks.append(Chunk(chunk_id=chunk_id, text=raw, source=source_name, doc_id=doc_id))
    return chunks

# this function is used to extract triples(subject, predicate, object) from a single chunk of text using a structured LLM output. It takes a Chunk object and a BaseChatModel object as input and returns a list of Triple objects.
def extract_triples_for_chunk(chunk: Chunk, llm: BaseChatModel) -> list[Triple]:
    """Extract triples from a single chunk using structured LLM output."""
    #here we use the with_structured_output() method to wrap the llm model with a structured output parser that expects the output to conform to the TripleExtractionResult Pydantic schema. This allows us to easily parse the LLM's output into a structured format that we can work with in our code.
    structured_llm = llm.with_structured_output(TripleExtractionResult)

    #here we use try catch block to handle any exceptions that may occur during the triple extraction process. If an exception occurs, we log a warning message and return an empty list of triples for that chunk. This ensures that the ingestion process continues even if triple extraction fails for a specific chunk.
    #because here we use llm to extract triples so it may happen sometimes that triples won't be generate at that condition we don't want to stop the ingestion process so we return empty list of triples for that chunk in catch block.
    try:

        #this will invoke the structured LLM with a system prompt that instructs it to extract triples from the text chunk. The system prompt is defined in the TRIPLE_EXTRACTION_SYSTEM_PROMPT constant top of the file. The human prompt provides the actual text chunk to be processed. The result will be a TripleExtractionResult object containing the extracted triples.
        result: TripleExtractionResult = structured_llm.invoke(
            [
                ("system", TRIPLE_EXTRACTION_SYSTEM_PROMPT),
                ("human", f"Text chunk:\n\n{chunk.text}"),
            ]
        )
        return result.triples
    except Exception as exc:  # noqa: BLE001 -- extraction must never crash ingestion
        logger.warning("Triple extraction failed for chunk %s: %s", chunk.chunk_id, exc)
        return []

#this function is for adding one list key in the chunk object which is remaining after the chunking process. 
def extract_triples_for_document(chunks: list[Chunk], llm: BaseChatModel) -> list[Chunk]:
    """Mutates and returns chunks with `.triples` populated."""

    #for every chunk in the list of chunks we call the extract_triples_for_chunk() function to extract triples from that chunk and assign the result to the triples attribute of the chunk. This effectively append_back each chunk with its corresponding extracted triples.
    for chunk in chunks:
        chunk.triples = extract_triples_for_chunk(chunk, llm)
    return chunks
#now our chunk is ready with chunk_text of 800characters + triples extracted from the chunk_text.
#chunk_text is used in vector_store.py for embedding and triples are used in graph_store.py to create a knowledge graph.