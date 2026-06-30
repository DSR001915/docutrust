"""
Persists CRAG run traces (and reads them back for the /trace/{run_id}
endpoint and the eval scripts). Falls back to an in-memory store when
MongoDB isn't reachable -- see backend/db/mongo_client.py for why that
matters in this codebase specifically.
"""
from __future__ import annotations

import threading
import time
from typing import Any

from backend.db.mongo_client import runs_collection


class InMemoryTraceStore:
    """Process-local fallback store. Thread-safe for the simple get/set
    access pattern used here (FastAPI's default threadpool can serve
    multiple requests concurrently even in an otherwise-async app)."""

    def __init__(self):
        self._runs: dict[str, dict] = {}
        self._lock = threading.Lock()

    def save_run(self, run_id: str, document: dict) -> None:
        with self._lock:
            self._runs[run_id] = document

    def get_run(self, run_id: str) -> dict | None:
        with self._lock:
            return self._runs.get(run_id)

    def list_runs(self, limit: int = 50) -> list[dict]:
        with self._lock:
            runs = sorted(self._runs.values(), key=lambda r: r.get("created_at", 0), reverse=True)
            return runs[:limit]


_memory_store = InMemoryTraceStore()


def save_run_trace(
    run_id: str,
    original_query: str,
    final_answer: str,
    citations: list[str],
    trace: list[dict],
    used_web_fallback: bool,
    iteration_count: int,
    grade: str,
) -> None:
    """Persist a completed run. Document shape is intentionally flat and
    JSON-serializable as-is, since it's returned directly by the
    /trace/{run_id} API route to the frontend with no transformation."""
    document: dict[str, Any] = {
        "run_id": run_id,
        "original_query": original_query,
        "final_answer": final_answer,
        "citations": citations,
        "trace": trace,
        "used_web_fallback": used_web_fallback,
        "iteration_count": iteration_count,
        "final_grade": grade,
        "created_at": time.time(),
    }

    collection = runs_collection()
    if collection is not None:
        collection.replace_one({"run_id": run_id}, document, upsert=True)
    else:
        _memory_store.save_run(run_id, document)


def get_run_trace(run_id: str) -> dict | None:
    collection = runs_collection()
    if collection is not None:
        doc = collection.find_one({"run_id": run_id}, {"_id": 0})
        return doc
    return _memory_store.get_run(run_id)


def list_recent_runs(limit: int = 50) -> list[dict]:
    collection = runs_collection()
    if collection is not None:
        cursor = collection.find({}, {"_id": 0}).sort("created_at", -1).limit(limit)
        return list(cursor)
    return _memory_store.list_runs(limit)
