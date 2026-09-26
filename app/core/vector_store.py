"""
Phase 2 (semantic half) -- Dense vector store on ChromaDB (PRD 3.1, 4.2).
"""
from __future__ import annotations

import chromadb
from chromadb.api.models.Collection import Collection

from app.core.config import Settings, get_settings
from app.core.llm_provider import get_embedding_function
from app.schemas.models import Chunk


class VectorStore:
    def __init__(self, settings: Settings | None = None) -> None:

        #settings is a object that contains all the .env variables.
        self.settings = settings or get_settings()

        # PersistentClient is a ChromaDB client that persists(create) the database to disk at the specified path(path is defined in settings(as env variable))).
        # Everything(sqlite, binary files, index) is automatically stored on defined path. Without PersistentClient everything would disappear after restarting the app.
        self.client = chromadb.PersistentClient(path=self.settings.chroma_persist_dir)

        # we get the embedding model using the get_embedding_function() function defined in llm_provider.py file. This function returns a LangChain Embeddings object.
        self.embedder = get_embedding_function()

        # get_or_create_collection() method is used to get the collection named "graph_rag_chunks" from the ChromaDB database. If the collection does not exist, it will be created with the specified metadata.
        # metadata tells How should similarity be measured? here we use cosine similarity for vector comparisons. This is important for the retrieval process, as it determines how "close" or "similar" two vectors are in the embedding space.
        self.collection: Collection = self.client.get_or_create_collection(
            name="graph_rag_chunks",#collection(table) name.
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        # we extract the text from each chunk to create a list of texts that will be embedded.
        texts = [c.text for c in chunks]
        # we use the embedder to convert the list of texts into a list of embeddings(vectors).
        embeddings = self.embedder.embed_documents(texts)
        # we use the upsert() method(update or insert) to add the chunks to the ChromaDB collection. The upsert() method takes the following parameters:
        self.collection.upsert(
            #chunk id, extracted from the each chunk object.
            ids=[c.chunk_id for c in chunks],
            # the actual text content of the chunks.
            documents=texts,
            # the embeddings(vectors) of the chunks.
            embeddings=embeddings,
            #some metadata about chunks.
            metadatas=[{"source": c.source, "doc_id": c.doc_id} for c in chunks],
        )

    def similarity_search(self, query: str, top_k: int = 6) -> list[dict]:
        #here we convert text query into embedded query.
        query_embedding = self.embedder.embed_query(query)
        #here we run query over the collection(the table where chunks are stored) for the most similar chunks.
        #here we use the query_embeddings parameter to pass the embedded queries(like we can pass multiple queries query_embeddings=[query1_embedding, query2_embedding, ...]) and n_results parameter to specify how many results we want to retrieve.
        results = self.collection.query(query_embeddings=[query_embedding], n_results=top_k)
        #if we pass multiple queries to the query_embeddings parameter, the results will be returned as a list of lists(array of arrays(2D array)).
        #here we pass only one query, so we get [[chunk1, chunk2, ...]] as the result. So we use results.get("ids", [[]])[0] to get the first list of results.

        out = []
        #in ids we have[chunk1_id, chunk2_id, ...], in docs we have [chunk1_text, chunk2_text, ...], in metas we have [chunk1_meta, chunk2_meta, ...], in distances we have [chunk1_distance, chunk2_distance, ...].
        ids = results.get("ids", [[]])[0]
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]  # cosine distance, lower = closer

        # this for loop iterates over the retrieved chunks and constructs a list of dictionaries containing the chunk_id, text, source, doc_id, and vector_score (cosine similarity) for each chunk. The vector_score is calculated as 1.0 - distance to convert cosine distance to cosine similarity.
        for chunk_id, text, meta, dist in zip(ids, docs, metas, distances):
            similarity = 1.0 - dist  # cosine distance -> cosine similarity
            out.append(
                {
                    "chunk_id": chunk_id,
                    "text": text,
                    "source": meta.get("source", "unknown"),
                    "doc_id": meta.get("doc_id", "unknown"),
                    "vector_score": max(0.0, similarity),
                }
            )
        return out

    #gives the total number of chunks stored in the ChromaDB collection. This can be useful for monitoring the size of the vector store and for debugging purposes.
    def count(self) -> int:
        return self.collection.count()
