"""
Tests for the CRAG state machine: routing decisions, the iteration cap,
and end-to-end path verification using a real (temp, isolated) vector
store populated with known content.

These tests are the project's most important regression suite: they pin
down the exact behavior described in docs/ARCHITECTURE.md and answer the
interview question "what stops the correction loop from running forever?"
with an executable assertion, not just a comment.
"""
import shutil
import tempfile

import pytest

from backend.config import get_settings
from backend.graph.edges import route_after_grading
from backend.graph.state import initial_state

settings = get_settings()


# ---- Unit tests: routing logic in isolation (no graph execution needed) ----

def test_route_after_grading_correct_goes_to_generate():
    state = initial_state(run_id="r1", query="q")
    state["grade"] = "correct"
    state["iteration_count"] = 0
    assert route_after_grading(state) == "generate"


def test_route_after_grading_ambiguous_under_cap_goes_to_rewrite():
    state = initial_state(run_id="r1", query="q")
    state["grade"] = "ambiguous"
    state["iteration_count"] = 0
    assert route_after_grading(state) == "rewrite_query"


def test_route_after_grading_incorrect_under_cap_goes_to_rewrite():
    state = initial_state(run_id="r1", query="q")
    state["grade"] = "incorrect"
    state["iteration_count"] = settings.max_correction_iterations - 1
    assert route_after_grading(state) == "rewrite_query"


def test_route_after_grading_at_cap_escalates_to_web_fallback():
    """The single most important routing test in this suite: once
    iteration_count reaches MAX_CORRECTION_ITERATIONS, routing must
    escalate to web_fallback regardless of grade, never loop again. This
    is what guarantees the graph terminates."""
    state = initial_state(run_id="r1", query="q")
    state["grade"] = "ambiguous"
    state["iteration_count"] = settings.max_correction_iterations
    assert route_after_grading(state) == "web_fallback"


def test_route_after_grading_correct_skips_cap_entirely():
    """Even if iteration_count is already at/past the cap, a 'correct'
    grade should still go straight to generate -- the cap only governs the
    ambiguous/incorrect retry path, it should never block a successful
    grade from completing normally."""
    state = initial_state(run_id="r1", query="q")
    state["grade"] = "correct"
    state["iteration_count"] = settings.max_correction_iterations + 5
    assert route_after_grading(state) == "generate"


# ---- Integration tests: full graph execution against a real (temp) vector store ----

@pytest.fixture
def populated_graph(monkeypatch):
    """Builds a fresh CRAG graph wired to an isolated, temp-dir vector
    store populated with known content, so these tests don't depend on
    (or pollute) the project's default ./chroma_data store."""
    from backend.ingestion import vector_store as vs_module
    from backend.ingestion.chunker import chunk_pages
    from backend.ingestion.embedder import MockEmbedder
    from backend.ingestion.parser import PageText
    from backend.ingestion.vector_store import VectorStore
    from backend.graph.build_graph import build_crag_graph

    tmp_dir = tempfile.mkdtemp(prefix="docutrust_graph_test_")
    test_store = VectorStore(persist_dir=tmp_dir, collection_name="test_graph")
    monkeypatch.setattr(vs_module, "_store_singleton", test_store)

    pages = [
        PageText(
            page_number=1,
            text=(
                "Data Retention Policy. All customer transaction records must be "
                "retained for a minimum of seven years from the date of the "
                "transaction, in accordance with financial regulatory requirements."
            ),
        )
    ]
    chunks = chunk_pages(pages, document_id="doc-1", document_name="policy.pdf", chunk_size=400, chunk_overlap=40)

    embedder = MockEmbedder()
    vectors = embedder.embed_documents([c.text for c in chunks])
    test_store.upsert_chunks(chunks, vectors)

    graph = build_crag_graph()

    yield graph

    shutil.rmtree(tmp_dir, ignore_errors=True)


def test_in_corpus_query_takes_direct_generate_path(populated_graph):
    state = initial_state(run_id="t1", query="How long must customer transaction records be retained?")
    result = populated_graph.invoke(state)

    assert result["grade"] == "correct"
    assert result["iteration_count"] == 0
    assert result["used_web_fallback"] is False
    assert len(result["final_answer"]) > 0
    assert "retrieve" in [t["node"] for t in result["trace"]]
    assert "generate" in [t["node"] for t in result["trace"]]
    assert "web_fallback" not in [t["node"] for t in result["trace"]]


def test_out_of_corpus_query_escalates_to_web_fallback(populated_graph):
    state = initial_state(run_id="t2", query="What is the boiling point of liquid nitrogen?")
    result = populated_graph.invoke(state)

    assert result["used_web_fallback"] is True
    assert result["iteration_count"] >= 1
    trace_nodes = [t["node"] for t in result["trace"]]
    assert "web_fallback" in trace_nodes
    assert "corrective_generate" in trace_nodes


def test_correction_loop_never_exceeds_max_iterations(populated_graph):
    """End-to-end termination guarantee: regardless of how poor retrieval
    is, the graph must terminate within MAX_CORRECTION_ITERATIONS rewrite
    attempts before escalating -- this test would hang or fail if the
    iteration cap logic in backend/graph/edges.py were ever removed or
    broken."""
    state = initial_state(run_id="t3", query="completely unrelated nonsense query about astrophysics")
    result = populated_graph.invoke(state)

    assert result["iteration_count"] <= settings.max_correction_iterations
    rewrite_count = sum(1 for t in result["trace"] if t["node"] == "rewrite_query")
    assert rewrite_count <= settings.max_correction_iterations


def test_final_answer_always_populated(populated_graph):
    """Every path through the graph -- direct generate or corrective
    generate -- must leave final_answer non-empty. An empty final_answer
    reaching the API layer would be a silent failure."""
    for query in [
        "How long must customer transaction records be retained?",
        "What is the boiling point of liquid nitrogen?",
    ]:
        state = initial_state(run_id=f"t-{hash(query)}", query=query)
        result = populated_graph.invoke(state)
        assert result["final_answer"], f"Empty final_answer for query: {query}"
