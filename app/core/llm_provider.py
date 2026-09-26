"""
Provider-agnostic LLM + embedding factory.

Swapping `LLM_PROVIDER` in .env between openai / groq / ollama changes the
entire backend with zero code changes, satisfying the PRD's "zero-cost
local execution using Ollama or Groq alongside OpenAI" requirement.
"""

#basically this file is used to get the LLM model and embedding function based on the settings provided in the .env file.
#based on our .env file it will return the llm model and embedding model 

from functools import lru_cache

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel

from app.core.config import Settings, get_settings


def get_chat_model(settings: Settings | None = None) -> BaseChatModel:
    settings = settings or get_settings()

    if settings.llm_provider == "openai":
        from langchain_openai import ChatOpenAI

        if not settings.openai_api_key:
            raise RuntimeError("LLM_PROVIDER=openai requires OPENAI_API_KEY in .env")
        return ChatOpenAI(model=settings.openai_chat_model, api_key=settings.openai_api_key, temperature=0.1)

    if settings.llm_provider == "groq":
        from langchain_groq import ChatGroq

        if not settings.groq_api_key:
            raise RuntimeError("LLM_PROVIDER=groq requires GROQ_API_KEY in .env")
        return ChatGroq(model=settings.groq_chat_model, api_key=settings.groq_api_key, temperature=0.1)

    if settings.llm_provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(model=settings.ollama_chat_model, base_url=settings.ollama_base_url, temperature=0.1)

    raise ValueError(f"Unknown LLM_PROVIDER: {settings.llm_provider}")


@lru_cache
def get_embedding_function(settings: Settings | None = None) -> Embeddings:
    """
    Returns a LangChain Embeddings object. Defaults to a local
    sentence-transformers model (`all-MiniLM-L6-v2`) so ChromaDB indexing
    works fully offline without any API key. Set EMBEDDING_BACKEND=openai
    to use `text-embedding-3-small` instead (per PRD 3.1).
    """
    settings = settings or get_settings()

    if settings.embedding_backend == "openai":
        from langchain_openai import OpenAIEmbeddings

        if not settings.openai_api_key:
            raise RuntimeError("EMBEDDING_BACKEND=openai requires OPENAI_API_KEY in .env")
        return OpenAIEmbeddings(model=settings.openai_embedding_model, api_key=settings.openai_api_key)

    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(model_name=settings.local_embedding_model)
