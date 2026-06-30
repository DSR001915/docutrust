"""
Shared state for the CRAG LangGraph state machine.

Every node in backend/graph/nodes.py reads from and writes to this single
TypedDict. Keeping it flat and explicit (rather than nesting objects) makes
it trivial to serialize the full state into a trace log entry after each
node transition -- see backend/db/trace_logger.py.
"""
from __future__ import annotations

from typing import TypedDict

from backend.llm.provider import GenerationResult
from backend.reranker.cross_encoder import RankedCandidate


class GraphState(TypedDict, total=False):
    # --- input ---
    run_id: str
    original_query: str

    # --- mutated across iterations ---
    current_query: str           # may differ from original_query after rewrite
    iteration_count: int         # how many retrieve->grade->rewrite loops so far

    # --- retrieval / grading results (overwritten each iteration) ---
    candidates: list[RankedCandidate]   # post-rerank, pre-grade-threshold
    top_score: float
    grade: str                          # "correct" | "ambiguous" | "incorrect"

    # --- corrective path ---
    used_web_fallback: bool
    web_results: list[RankedCandidate]  # web search results, normalized to RankedCandidate shape
    rewrite_reason: str                 # why a rewrite was triggered, fed to rewrite_query()

    # --- final output ---
    final_answer: str
    citations: list[str]                # chunk_ids cited in final_answer

    # --- trace (append-only log of node transitions, for the UI + Mongo) ---
    trace: list[dict]


def initial_state(run_id: str, query: str) -> GraphState:
    return GraphState(
        run_id=run_id,
        original_query=query,
        current_query=query,
        iteration_count=0,
        candidates=[],
        top_score=0.0,
        grade="",
        used_web_fallback=False,
        web_results=[],
        rewrite_reason="",
        final_answer="",
        citations=[],
        trace=[],
    )
