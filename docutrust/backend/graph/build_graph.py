"""
Assembles the CRAG state machine from the nodes (backend/graph/nodes.py)
and conditional routing (backend/graph/edges.py) into a compiled LangGraph
StateGraph.

Graph shape:

    retrieve -> grade_documents -> [conditional] -+-> generate -> END
                       ^                           |
                       |                           +-> rewrite_query -> retrieve (loop, capped)
                       |                           |
                       +---------------------------+-> web_fallback -> corrective_generate -> END

Note rewrite_query loops back to retrieve (not grade_documents directly) --
a rewritten query needs a fresh retrieval pass before it can be graded again.
"""
from __future__ import annotations

from langgraph.graph import StateGraph, END

from backend.graph.edges import route_after_grading
from backend.graph.nodes import (
    corrective_generate_node,
    generate_node,
    grade_documents_node,
    retrieve_node,
    rewrite_query_node,
    web_fallback_node,
)
from backend.graph.state import GraphState


def build_crag_graph():
    graph = StateGraph(GraphState)

    graph.add_node("retrieve", retrieve_node)
    graph.add_node("grade_documents", grade_documents_node)
    graph.add_node("generate", generate_node)
    graph.add_node("rewrite_query", rewrite_query_node)
    graph.add_node("web_fallback", web_fallback_node)
    graph.add_node("corrective_generate", corrective_generate_node)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "grade_documents")

    graph.add_conditional_edges(
        "grade_documents",
        route_after_grading,
        {
            "generate": "generate",
            "rewrite_query": "rewrite_query",
            "web_fallback": "web_fallback",
        },
    )

    # Rewritten query needs a fresh retrieval pass before it can be re-graded.
    graph.add_edge("rewrite_query", "retrieve")

    graph.add_edge("web_fallback", "corrective_generate")

    graph.add_edge("generate", END)
    graph.add_edge("corrective_generate", END)

    return graph.compile()


_compiled_graph = None


def get_crag_graph():
    """Process-wide singleton -- compiling the graph has a small fixed cost
    that's wasteful to repeat on every request."""
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_crag_graph()
    return _compiled_graph
