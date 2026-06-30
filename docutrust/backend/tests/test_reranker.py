"""Tests for the reranker / CRAG grading logic."""
from backend.config import get_settings
from backend.ingestion.vector_store import RetrievedCandidate
from backend.reranker.cross_encoder import MockReranker, grade_relevance

settings = get_settings()


def _candidate(text: str, distance: float = 0.5) -> RetrievedCandidate:
    return RetrievedCandidate(
        chunk_id="c1",
        document_id="d1",
        document_name="test.pdf",
        page_number=1,
        text=text,
        distance=distance,
    )


def test_reranker_ranks_relevant_above_irrelevant():
    reranker = MockReranker()
    query = "what is the data retention period for customer records"
    candidates = [
        _candidate("the cafeteria menu changes every Tuesday for lunch service"),
        _candidate("customer records must be retained for seven years per the data retention policy"),
        _candidate("remote employees must use company-issued laptops with encryption"),
    ]

    ranked = reranker.rerank(query, candidates, top_k=3)

    assert "retention" in ranked[0].text
    assert ranked[0].score > ranked[1].score
    assert ranked[0].score > ranked[2].score


def test_reranker_empty_candidates_returns_empty():
    reranker = MockReranker()
    result = reranker.rerank("any query", [], top_k=5)
    assert result == []


def test_reranker_respects_top_k():
    reranker = MockReranker()
    candidates = [_candidate(f"document number {i} about various topics") for i in range(10)]
    ranked = reranker.rerank("various topics", candidates, top_k=3)
    assert len(ranked) == 3


def test_grade_relevance_correct_band():
    score = settings.relevance_threshold_correct + 0.05
    assert grade_relevance(score) == "correct"


def test_grade_relevance_ambiguous_band():
    midpoint = (settings.relevance_threshold_correct + settings.relevance_threshold_ambiguous) / 2
    assert grade_relevance(midpoint) == "ambiguous"


def test_grade_relevance_incorrect_band():
    score = settings.relevance_threshold_ambiguous - 0.05
    assert grade_relevance(score) == "incorrect"


def test_grade_relevance_boundary_is_inclusive_correct():
    """Score exactly AT the correct threshold should grade as correct
    (>=), not fall into ambiguous -- boundary behavior worth pinning down
    explicitly since off-by-one errors here change CRAG's routing."""
    assert grade_relevance(settings.relevance_threshold_correct) == "correct"


def test_grade_relevance_boundary_is_inclusive_ambiguous():
    assert grade_relevance(settings.relevance_threshold_ambiguous) == "ambiguous"
