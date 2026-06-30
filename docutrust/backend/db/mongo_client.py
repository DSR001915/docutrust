"""
MongoDB client wrapper.

Collections used:
  - documents: one record per uploaded file (metadata, chunk count, status)
  - runs: one record per query, with the full node-by-node trace embedded
          (this is the "agent observability" data the README points to)

Connection is lazy and tolerant: if MongoDB isn't reachable (e.g. running
this codebase outside docker-compose, or in a sandbox with no Mongo
service), get_db() returns None and callers fall back to in-memory storage
via backend/db/trace_logger.py's InMemoryTraceStore. This keeps the rest of
the app (especially the LangGraph pipeline, which doesn't depend on Mongo
at all) testable without standing up infrastructure.
"""
from __future__ import annotations

from pymongo import MongoClient
from pymongo.database import Database
from pymongo.errors import ServerSelectionTimeoutError

from backend.config import get_settings

settings = get_settings()

_client: MongoClient | None = None
_db: Database | None = None
_connection_attempted = False


def get_db() -> Database | None:
    """Returns the MongoDB database handle, or None if MongoDB is
    unreachable. Connection is attempted once per process (not retried on
    every call) so a down Mongo doesn't add latency to every request --
    callers should treat a None return as "logging degraded, app still
    works" rather than a fatal error."""
    global _client, _db, _connection_attempted

    if _connection_attempted:
        return _db

    _connection_attempted = True
    try:
        _client = MongoClient(settings.mongo_uri, serverSelectionTimeoutMS=2000)
        _client.admin.command("ping")  # forces actual connection check, not just object creation
        _db = _client[settings.mongo_db_name]
        print(f"[mongo] Connected to {settings.mongo_uri}, db={settings.mongo_db_name}")
    except ServerSelectionTimeoutError as exc:
        print(
            f"[mongo] Could not connect to MongoDB at {settings.mongo_uri} ({exc}). "
            f"Trace logging and document metadata will use in-memory fallback storage "
            f"for this process -- data will not persist across restarts. Run "
            f"`docker-compose up` to start MongoDB, or set MONGO_URI to a reachable instance."
        )
        _db = None

    return _db


def documents_collection():
    db = get_db()
    return db["documents"] if db is not None else None


def runs_collection():
    db = get_db()
    return db["runs"] if db is not None else None
