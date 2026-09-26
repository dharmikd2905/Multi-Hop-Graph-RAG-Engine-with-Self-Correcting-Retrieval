"""
Central configuration for the Graph-Augmented Self-RAG Engine.

All values are overridable via environment variables / .env so the same
codebase runs against OpenAI, Groq, or a local Ollama model with zero
code changes (only .env changes) -- per PRD section 2 (zero-cost local
reproduction requirement).
"""
from functools import lru_cache
from typing import Literal

# base_settings allows us to load settings from .env and override them with environment variables
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- LLM Provider selection -------------------------------------------------
    # "openai"  -> uses OPENAI_API_KEY, requires billing
    # "groq"    -> uses GROQ_API_KEY, free tier, Llama 3.1/3.3 hosted, very fast
    # "ollama"  -> fully local, zero cost, zero API key (requires `ollama serve`)
    # literal type ensures only one of these three is allowed.
    llm_provider: Literal["openai", "groq", "ollama"] = "ollama"

    openai_api_key: str | None = None
    openai_chat_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"

    groq_api_key: str | None = None
    groq_chat_model: str = "llama-3.3-70b-versatile"

    ollama_base_url: str = "http://localhost:11434"
    ollama_chat_model: str = "llama3.1"
    ollama_embedding_model: str = "nomic-embed-text"

    # --- Embeddings ---------------------------------------------------------
    #here in .env we set EMBEDDING_BACKEND=openai to use OpenAI embeddings, otherwise it defaults to local embeddings.
    #doesn't matter which LLM provider is used, the embedding backend can be set independently.
    embedding_backend: Literal["openai", "local"] = "local"
    local_embedding_model: str = "all-MiniLM-L6-v2"

    # --- Chunking (PRD 4.1) --------------------------------------------------
    chunk_size: int = 800
    chunk_overlap: int = 150

    # --- Retrieval (PRD 4.3) --------------------------------------------------
    vector_top_k: int = 6
    graph_hops: int = 2
    pagerank_damping: float = 0.85
    max_context_chunks: int = 8

    # --- Self-correction loop (PRD 4.4) --------------------------------------
    max_correction_retries: int = 2
    hallucination_threshold: float = 0.6
    relevance_threshold: float = 0.6

    # --- Storage --------------------------------------------------------------
    # On Railway, mount a Volume at /data (Service -> Settings -> Volumes) and
    # set these two env vars to paths under it, e.g. /data/chroma and
    # /data/graph.gpickle, so the index survives restarts/redeploys. Locally
    # the ./data default just works.
    chroma_persist_dir: str = "./data/chroma"
    graph_persist_path: str = "./data/graph.gpickle"

    # --- Observability ---------------------------------------------------------
    langchain_tracing_v2: bool = False
    langchain_project: str = "graph-augmented-self-rag"
    langchain_api_key: str | None = None

    # --- API ---------------------------------------------------------------
    api_title: str = "Graph-Augmented Self-RAG Engine"
    api_version: str = "1.0.0"

#lru_cache means that the settings object is only created once and reused, which is important for performance and consistency across the application.
@lru_cache
def get_settings() -> Settings:
    # Returns a cached Settings object, loading from .env and environment variables.
    return Settings()
