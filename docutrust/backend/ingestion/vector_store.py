"""
Vector store wrapper around ChromaDB.

Deliberately thin: every method signature here is something you could
re-implement against Qdrant or Pinecone without touching any caller code.
That's the abstraction boundary referenced in docs/ARCHITECTURE.md when
the README says "swap to Qdrant for production scale."
"""
from __future__ import annotations

from dataclasses import dataclass

import chromadb
from chromadb.api.models.Collection import Collection

from backend.config import get_settings
from backend.ingestion.chunker import Chunk

settings = get_settings()


@dataclass
class RetrievedCandidate:
    chunk_id: str
    document_id: str
    document_name: str
    page_number: int
    text: str
    distance: float  # raw vector-store distance (lower = more similar, for cosine: 1 - cos_sim)


class VectorStore:
    """Persistent local ChromaDB collection for chunk embeddings."""

    def __init__(self, persist_dir: str | None = None, collection_name: str | None = None):
        self._client = chromadb.PersistentClient(
            path=persist_dir or settings.chroma_persist_dir
        )
        self._collection: Collection = self._client.get_or_create_collection(
            name=collection_name or settings.chroma_collection,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        """Add (or overwrite, by chunk_id) chunk embeddings + metadata."""
        if not chunks:
            return
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"chunks ({len(chunks)}) and embeddings ({len(embeddings)}) length mismatch"
            )

        self._collection.upsert(
            ids=[c.chunk_id for c in chunks],
            embeddings=embeddings,
            documents=[c.text for c in chunks],
            metadatas=[
                {
                    "document_id": c.document_id,
                    "document_name": c.document_name,
                    "page_number": c.page_number,
                    "chunk_index": c.chunk_index,
                }
                for c in chunks
            ],
        )

    def query(self, query_embedding: list[float], top_k: int) -> list[RetrievedCandidate]:
        """Bi-encoder recall stage: fetch the top_k nearest chunks by vector
        distance. This is the WIDE, cheap pass -- precision filtering happens
        later in the cross-encoder reranker, not here."""
        if self._collection.count() == 0:
            return []

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self._collection.count()),
        )

        candidates: list[RetrievedCandidate] = []
        ids = results["ids"][0]
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        for chunk_id, text, meta, distance in zip(ids, documents, metadatas, distances):
            candidates.append(
                RetrievedCandidate(
                    chunk_id=chunk_id,
                    document_id=meta["document_id"],
                    document_name=meta["document_name"],
                    page_number=meta["page_number"],
                    text=text,
                    distance=distance,
                )
            )

        return candidates

    def document_count(self) -> int:
        return self._collection.count()

    def delete_document(self, document_id: str) -> None:
        """Remove all chunks belonging to a document (e.g. on re-upload)."""
        self._collection.delete(where={"document_id": document_id})


_store_singleton: VectorStore | None = None


def get_vector_store() -> VectorStore:
    """Process-wide singleton so repeated calls reuse the same persistent
    Chroma client instead of re-opening the on-disk store each time."""
    global _store_singleton
    if _store_singleton is None:
        _store_singleton = VectorStore()
    return _store_singleton
