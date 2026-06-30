"""
LangGraph node functions for the CRAG state machine.

Each function takes the current GraphState and returns a dict of fields to
merge into it (the standard LangGraph node contract). Nodes are kept free of
direct provider instantiation where possible -- they call the factories in
embedder.py / cross_encoder.py / provider.py / web_search.py, so swapping
mock <-> local <-> real providers never requires touching this file.

Read this file top-to-bottom and it IS the CRAG algorithm description:
retrieve -> grade -> (generate | rewrite -> retrieve again, capped | web_fallback) -> corrective_generate
"""
from __future__ import annotations

import time

from backend.config import get_settings
from backend.graph.state import GraphState
from backend.graph.web_search import get_web_search
from backend.ingestion.embedder import get_embedder
from backend.ingestion.vector_store import get_vector_store
from backend.llm.provider import get_llm
from backend.reranker.cross_encoder import grade_relevance, get_reranker

settings = get_settings()


def _trace_event(state: GraphState, node: str, **details) -> dict:
    """Build a single trace entry. Appended to state["trace"] by every node
    so the frontend can stream a live step-by-step log and Mongo can persist
    the full run for later inspection (backend/db/trace_logger.py)."""
    return {
        "node": node,
        "timestamp": time.time(),
        **details,
    }


def retrieve_node(state: GraphState) -> dict:
    """Bi-encoder recall stage: embed the current query, fetch a wide
    candidate set, then immediately rerank down to a precise top-k.
    Retrieval and reranking are combined in one node because they're always
    called together -- there's no useful intermediate state between them
    that another node would act on."""
    embedder = get_embedder()
    reranker = get_reranker()
    store = get_vector_store()

    query = state["current_query"]
    query_vector = embedder.embed_query(query)
    raw_candidates = store.query(query_vector, top_k=settings.retrieval_top_k)
    ranked = reranker.rerank(query, raw_candidates, top_k=settings.rerank_top_k)

    top_score = ranked[0].score if ranked else 0.0

    trace = state.get("trace", []) + [
        _trace_event(
            state,
            "retrieve",
            query=query,
            candidates_found=len(raw_candidates),
            reranked_to=len(ranked),
            top_score=round(top_score, 4),
        )
    ]

    return {
        "candidates": ranked,
        "top_score": top_score,
        "trace": trace,
    }


def grade_documents_node(state: GraphState) -> dict:
    """Maps the reranker's top score to a CRAG grade. This node does no
    LLM call -- grading is purely the cross-encoder score against the
    thresholds in config.py. That's a deliberate design choice: a numeric
    threshold is auditable and reproducible in a way an LLM's self-judgment
    isn't."""
    grade = grade_relevance(state["top_score"])

    trace = state.get("trace", []) + [
        _trace_event(
            state,
            "grade_documents",
            grade=grade,
            top_score=round(state["top_score"], 4),
            threshold_correct=settings.relevance_threshold_correct,
            threshold_ambiguous=settings.relevance_threshold_ambiguous,
        )
    ]

    return {"grade": grade, "trace": trace}


def rewrite_query_node(state: GraphState) -> dict:
    """Triggered when grade is ambiguous/incorrect and we haven't hit the
    iteration cap yet. Reformulates the query and increments the loop
    counter -- the counter is what lets the conditional edge function
    (backend/graph/edges.py) enforce MAX_CORRECTION_ITERATIONS."""
    llm = get_llm()
    reason = f"top reranker score {state['top_score']:.3f} graded as '{state['grade']}'"
    rewritten = llm.rewrite_query(state["current_query"], reason)

    trace = state.get("trace", []) + [
        _trace_event(
            state,
            "rewrite_query",
            original_query=state["current_query"],
            rewritten_query=rewritten,
            reason=reason,
            iteration=state["iteration_count"] + 1,
        )
    ]

    return {
        "current_query": rewritten,
        "iteration_count": state["iteration_count"] + 1,
        "rewrite_reason": reason,
        "trace": trace,
    }


def web_fallback_node(state: GraphState) -> dict:
    """Last resort: internal KB retrieval failed even after the rewrite
    loop. Pull external context via web search, clearly tagged so the
    generation node (and the final answer) can disclose the source."""
    web_search = get_web_search()
    results = web_search.search(state["current_query"], max_results=3)

    trace = state.get("trace", []) + [
        _trace_event(
            state,
            "web_fallback",
            query=state["current_query"],
            results_found=len(results),
        )
    ]

    return {
        "web_results": results,
        "used_web_fallback": True,
        "trace": trace,
    }


def generate_node(state: GraphState) -> dict:
    """Happy path: grade was 'correct' on the first or a later retrieval
    pass. Generate directly from the graded candidates, no web fallback
    needed."""
    llm = get_llm()
    result = llm.generate_answer(state["original_query"], state["candidates"], used_web_fallback=False)

    trace = state.get("trace", []) + [
        _trace_event(
            state,
            "generate",
            citations_count=len(result.citations),
        )
    ]

    return {
        "final_answer": result.answer,
        "citations": result.citations,
        "trace": trace,
    }


def corrective_generate_node(state: GraphState) -> dict:
    """Generates the final answer after correction: combines web fallback
    results with internal candidates -- but only internal candidates that
    cleared at least the 'ambiguous' threshold. Candidates that graded
    'incorrect' are dropped entirely rather than passed to the LLM, because
    presenting low-confidence internal chunks alongside (clearly-labeled)
    web results would let the model quietly treat noise as evidence. The
    LLM is explicitly told which context came from web fallback so it can
    disclose that in the answer."""
    llm = get_llm()

    internal_candidates = state.get("candidates", [])
    if state["grade"] == "incorrect":
        # Below even the ambiguous bar -- not worth presenting to the LLM.
        kept_internal = []
    else:
        kept_internal = internal_candidates

    combined_context = list(kept_internal) + list(state.get("web_results", []))
    result = llm.generate_answer(state["original_query"], combined_context, used_web_fallback=True)

    trace = state.get("trace", []) + [
        _trace_event(
            state,
            "corrective_generate",
            internal_chunks_used=len(kept_internal),
            internal_chunks_dropped=len(internal_candidates) - len(kept_internal),
            web_chunks_used=len(state.get("web_results", [])),
            citations_count=len(result.citations),
        )
    ]

    return {
        "final_answer": result.answer,
        "citations": result.citations,
        "trace": trace,
    }
