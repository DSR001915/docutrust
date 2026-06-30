"""
DocuTrust FastAPI application.

Routes:
  POST /api/upload          -- upload a PDF, runs the ingestion pipeline synchronously
  GET  /api/documents       -- list ingested documents
  POST /api/query           -- ask a question, returns the full CRAG result (blocking)
  GET  /api/trace/{run_id}  -- fetch a previously-run query's full trace
  WS   /ws/query             -- ask a question, stream node-by-node trace events live
  GET  /api/health          -- liveness check + which providers are active

Static files (frontend/) are mounted at / so this single process serves
both the API and the UI -- no separate frontend server needed for the demo.
"""
from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backend.config import get_settings
from backend.db.document_store import list_documents
from backend.db.trace_logger import get_run_trace, list_recent_runs
from backend.graph.query_service import run_query, run_query_streaming
from backend.ingestion.service import ingest_pdf

settings = get_settings()

app = FastAPI(title="DocuTrust", description="Self-correcting RAG platform")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # demo project; tighten this for any real deployment
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"


class QueryRequest(BaseModel):
    query: str


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "provider_mode": settings.docutrust_provider_mode,
        "llm_provider": settings.llm_provider,
        "web_search_provider": settings.web_search_provider,
    }


@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        result = ingest_pdf(tmp_path, original_filename=file.filename)
    except ValueError as exc:
        # parse_pdf raises ValueError for image-only/no-text PDFs -- surface
        # that as a 400, not a 500, since it's a client-correctable input problem.
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        os.unlink(tmp_path)

    return result


@app.get("/api/documents")
def get_documents():
    return {"documents": list_documents()}


@app.post("/api/query")
def post_query(request: QueryRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query must not be empty.")
    return run_query(request.query)


@app.get("/api/trace/{run_id}")
def get_trace(run_id: str):
    trace = get_run_trace(run_id)
    if trace is None:
        raise HTTPException(status_code=404, detail=f"No run found with id {run_id}")
    return trace


@app.get("/api/runs")
def get_recent_runs():
    return {"runs": list_recent_runs(limit=20)}


@app.websocket("/ws/query")
async def websocket_query(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            payload = await websocket.receive_json()
            query = payload.get("query", "").strip()
            if not query:
                await websocket.send_json({"type": "error", "data": "Query must not be empty."})
                continue

            async def on_step(event: dict):
                await websocket.send_json(event)

            try:
                await run_query_streaming(query, on_step)
            except Exception as exc:  # noqa: BLE001 - report to client instead of dropping connection
                await websocket.send_json({"type": "error", "data": str(exc)})
    except WebSocketDisconnect:
        pass


# --- Static frontend -------------------------------------------------------
# Mounted last so it doesn't shadow the /api and /ws routes above.
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

    @app.get("/")
    def serve_index():
        return FileResponse(str(FRONTEND_DIR / "index.html"))
