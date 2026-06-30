"""
High-level ingestion orchestration: the single function the FastAPI upload
route calls. Wires together parser -> chunker -> embedder -> vector store ->
document metadata, so the route handler itself stays thin.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from backend.config import get_settings
from backend.db.document_store import save_document_metadata
from backend.ingestion.chunker import chunk_pages
from backend.ingestion.embedder import get_embedder
from backend.ingestion.parser import parse_pdf
from backend.ingestion.vector_store import get_vector_store

settings = get_settings()


def _content_hash_id(file_path: str | Path) -> str:
    """Derives a stable document_id from file content (not a random UUID).

    Why this matters: re-uploading the same PDF (the same bytes) should be
    idempotent -- it should replace the existing chunks, not silently
    duplicate them in the vector store. A random UUID per upload makes that
    impossible to detect; a content hash makes "same file" detection free.
    """
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        for block in iter(lambda: f.read(8192), b""):
            hasher.update(block)
    return hasher.hexdigest()[:32]


def ingest_pdf(file_path: str | Path, original_filename: str) -> dict:
    """Runs the full ingestion pipeline on a single PDF already saved to
    disk. Returns a summary dict suitable for an API response.

    Re-ingesting the same file content is idempotent: any existing chunks
    for this document_id are deleted before the new ones are inserted, so
    re-uploads (or re-running ingestion after a chunking config change)
    never leave stale duplicate chunks in the vector store.
    """
    document_id = _content_hash_id(file_path)

    pages = parse_pdf(file_path)
    chunks = chunk_pages(
        pages,
        document_id=document_id,
        document_name=original_filename,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    embedder = get_embedder()
    vectors = embedder.embed_documents([c.text for c in chunks])

    store = get_vector_store()
    store.delete_document(document_id)  # clear any prior chunks for this exact content first
    store.upsert_chunks(chunks, vectors)

    save_document_metadata(
        document_id=document_id,
        filename=original_filename,
        page_count=len(pages),
        chunk_count=len(chunks),
    )

    return {
        "document_id": document_id,
        "filename": original_filename,
        "page_count": len(pages),
        "chunk_count": len(chunks),
    }
