"""
Conditional edge functions for the CRAG graph.

These are the functions LangGraph calls after grade_documents_node to decide
which node runs next. This is where MAX_CORRECTION_ITERATIONS is enforced --
the single most commonly-asked interview question about this project
("what stops the correction loop from running forever?") has its answer
right here, not buried in node logic.
"""
from __future__ import annotations

from backend.config import get_settings
from backend.graph.state import GraphState

settings = get_settings()


def route_after_grading(state: GraphState) -> str:
    """Returns the name of the next node based on the grade + iteration count.

    Decision table:
      grade == "correct"                              -> "generate"
      grade in {"ambiguous","incorrect"} AND iteration
        count < MAX_CORRECTION_ITERATIONS               -> "rewrite_query"
      grade in {"ambiguous","incorrect"} AND iteration
        count >= MAX_CORRECTION_ITERATIONS               -> "web_fallback"
    """
    grade = state["grade"]
    iteration_count = state["iteration_count"]

    if grade == "correct":
        return "generate"

    if iteration_count < settings.max_correction_iterations:
        return "rewrite_query"

    # Exhausted our rewrite budget and retrieval is still poor -- escalate
    # to web fallback rather than looping again or silently answering from
    # bad context.
    return "web_fallback"
