"""
Query orchestration: invokes the compiled CRAG graph for a single question
and persists the resulting trace. Used by both the synchronous /query REST
endpoint and the /ws/query WebSocket endpoint (the latter additionally
streams each node transition as it happens -- see backend/main.py).
"""
from __future__ import annotations

import uuid

from backend.db.trace_logger import save_run_trace
from backend.graph.build_graph import get_crag_graph
from backend.graph.state import initial_state


def run_query(query: str) -> dict:
    """Runs the full CRAG graph synchronously and returns + persists the result."""
    run_id = str(uuid.uuid4())
    graph = get_crag_graph()
    state = initial_state(run_id=run_id, query=query)

    result = graph.invoke(state)

    save_run_trace(
        run_id=run_id,
        original_query=query,
        final_answer=result["final_answer"],
        citations=result["citations"],
        trace=result["trace"],
        used_web_fallback=result["used_web_fallback"],
        iteration_count=result["iteration_count"],
        grade=result.get("grade", ""),
    )

    return {
        "run_id": run_id,
        "query": query,
        "answer": result["final_answer"],
        "citations": result["citations"],
        "used_web_fallback": result["used_web_fallback"],
        "iteration_count": result["iteration_count"],
        "trace": result["trace"],
    }


async def run_query_streaming(query: str, on_step):
    """Runs the CRAG graph, calling `on_step(event: dict)` after each node
    transition completes -- this is what powers the live trace stream over
    WebSocket. Uses LangGraph's `.astream()` so we get incremental state
    updates rather than only the final result."""
    run_id = str(uuid.uuid4())
    graph = get_crag_graph()
    state = initial_state(run_id=run_id, query=query)

    final_state = dict(state)
    seen_trace_len = 0

    async for step_output in graph.astream(state):
        # step_output is {node_name: partial_state_dict} for whichever
        # node(s) just ran.
        for node_name, partial_state in step_output.items():
            final_state.update(partial_state)

            # Stream only the NEW trace entries since the last step, so the
            # frontend gets one event per node transition rather than the
            # whole accumulated trace list each time.
            trace = final_state.get("trace", [])
            new_entries = trace[seen_trace_len:]
            seen_trace_len = len(trace)

            for entry in new_entries:
                await on_step({"type": "trace", "node": node_name, "data": entry})

    save_run_trace(
        run_id=run_id,
        original_query=query,
        final_answer=final_state.get("final_answer", ""),
        citations=final_state.get("citations", []),
        trace=final_state.get("trace", []),
        used_web_fallback=final_state.get("used_web_fallback", False),
        iteration_count=final_state.get("iteration_count", 0),
        grade=final_state.get("grade", ""),
    )

    await on_step(
        {
            "type": "final",
            "data": {
                "run_id": run_id,
                "answer": final_state.get("final_answer", ""),
                "citations": final_state.get("citations", []),
                "used_web_fallback": final_state.get("used_web_fallback", False),
            },
        }
    )
