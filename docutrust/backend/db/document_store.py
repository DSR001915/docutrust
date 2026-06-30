"""
Document metadata storage: one record per uploaded file. Same Mongo/
in-memory fallback pattern as backend/db/trace_logger.py.
"""
from __future__ import annotations

import threading
import time

from backend.db.mongo_client import documents_collection


class InMemoryDocumentStore:
    def __init__(self):
        self._docs: dict[str, dict] = {}
        self._lock = threading.Lock()

    def save(self, document_id: str, document: dict) -> None:
        with self._lock:
            self._docs[document_id] = document

    def list_all(self) -> list[dict]:
        with self._lock:
            return sorted(self._docs.values(), key=lambda d: d.get("uploaded_at", 0), reverse=True)

    def get(self, document_id: str) -> dict | None:
        with self._lock:
            return self._docs.get(document_id)

    def delete(self, document_id: str) -> None:
        with self._lock:
            self._docs.pop(document_id, None)


_memory_store = InMemoryDocumentStore()


def save_document_metadata(document_id: str, filename: str, page_count: int, chunk_count: int) -> None:
    document = {
        "document_id": document_id,
        "filename": filename,
        "page_count": page_count,
        "chunk_count": chunk_count,
        "uploaded_at": time.time(),
    }

    collection = documents_collection()
    if collection is not None:
        collection.replace_one({"document_id": document_id}, document, upsert=True)
    else:
        _memory_store.save(document_id, document)


def list_documents() -> list[dict]:
    collection = documents_collection()
    if collection is not None:
        cursor = collection.find({}, {"_id": 0}).sort("uploaded_at", -1)
        return list(cursor)
    return _memory_store.list_all()


def delete_document_metadata(document_id: str) -> None:
    collection = documents_collection()
    if collection is not None:
        collection.delete_one({"document_id": document_id})
    else:
        _memory_store.delete(document_id)
