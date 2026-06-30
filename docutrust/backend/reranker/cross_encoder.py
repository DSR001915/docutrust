"""
Cross-encoder reranker -- this IS the CRAG relevance grader.

Two implementations behind one interface, same pattern as embedder.py:
  - LocalReranker: real cross-encoder (BAAI/bge-reranker-v2-m3 via
    sentence-transformers' CrossEncoder class). Downloads weights from
    HuggingFace Hub on first use.
  - MockReranker: deterministic lexical-overlap scorer with the same output
    shape (a float score per query-document pair, roughly comparable in
    range to a real cross-encoder's sigmoid-ish output).

Why a cross-encoder is the grader (not an LLM "is this relevant? yes/no"
call): a cross-encoder jointly attends over the query and document in a
single forward pass, is purpose-trained for relevance scoring, and gives
you a continuous, thresholdable score instead of a categorical judgment
that varies run-to-run. That continuous score is what RERANK_THRESHOLD_*
in config.py operates on -- this file is where "grading" actually happens;
backend/graph/nodes.py just calls into it and routes on the result.
"""
from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

from backend.config import get_settings
from backend.ingestion.vector_store import RetrievedCandidate

settings = get_settings()

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    "a an the is are was were be been being of for to in on at by with "
    "and or but if then than this that these those it its as from into "
    "about must may can will shall not no do does did has have had "
    "our your their his her there here what which who whom".split()
)


@dataclass
class RankedCandidate:
    chunk_id: str
    document_id: str
    document_name: str
    page_number: int
    text: str
    score: float  # higher = more relevant, roughly in [0, 1]


class BaseReranker(ABC):
    @abstractmethod
    def rerank(
        self, query: str, candidates: list[RetrievedCandidate], top_k: int
    ) -> list[RankedCandidate]:
        """Score and re-sort candidates by true relevance to the query,
        returning the top_k. This is the precision stage that follows the
        vector store's recall stage."""


class LocalReranker(BaseReranker):
    """Real cross-encoder reranker. Requires `sentence-transformers` and
    network access to download model weights on first use."""

    def __init__(self, model_name: str | None = None):
        from sentence_transformers import CrossEncoder  # local import: optional heavy dep

        self.model_name = model_name or settings.reranker_model
        self._model = CrossEncoder(self.model_name)

    def rerank(
        self, query: str, candidates: list[RetrievedCandidate], top_k: int
    ) -> list[RankedCandidate]:
        if not candidates:
            return []

        pairs = [(query, c.text) for c in candidates]
        raw_scores = self._model.predict(pairs)  # bge-reranker outputs are not bounded [0,1]; squash below

        scored = []
        for candidate, raw_score in zip(candidates, raw_scores):
            score = _sigmoid(float(raw_score))
            scored.append(
                RankedCandidate(
                    chunk_id=candidate.chunk_id,
                    document_id=candidate.document_id,
                    document_name=candidate.document_name,
                    page_number=candidate.page_number,
                    text=candidate.text,
                    score=score,
                )
            )

        scored.sort(key=lambda c: c.score, reverse=True)
        return scored[:top_k]


class MockReranker(BaseReranker):
    """Deterministic lexical-overlap reranker: scores each candidate by a
    blend of (a) token overlap with the query and (b) the vector store's
    own distance ranking (as a tiebreaker / prior), producing a score in
    [0, 1]. Good enough to drive the CORRECT / AMBIGUOUS / INCORRECT grade
    thresholds in tests and restricted environments; not a substitute for
    a real cross-encoder's semantic judgment.
    """

    def rerank(
        self, query: str, candidates: list[RetrievedCandidate], top_k: int
    ) -> list[RankedCandidate]:
        if not candidates:
            return []

        query_tokens = _tokens(query)
        scored: list[RankedCandidate] = []

        for candidate in candidates:
            doc_tokens = _tokens(candidate.text)
            overlap_score = _jaccard_like(query_tokens, doc_tokens)
            # Vector distance is in roughly [0, 2] for cosine; convert to a
            # mild [0, 1] prior so it nudges, but doesn't dominate, the score.
            distance_prior = max(0.0, 1.0 - (candidate.distance / 2.0))
            score = 0.75 * overlap_score + 0.25 * distance_prior

            scored.append(
                RankedCandidate(
                    chunk_id=candidate.chunk_id,
                    document_id=candidate.document_id,
                    document_name=candidate.document_name,
                    page_number=candidate.page_number,
                    text=candidate.text,
                    score=score,
                )
            )

        scored.sort(key=lambda c: c.score, reverse=True)
        return scored[:top_k]


def _tokens(text: str) -> set[str]:
    raw = _TOKEN_RE.findall(text.lower())
    return {t for t in raw if t not in _STOPWORDS and len(t) > 2}


def _jaccard_like(query_tokens: set[str], doc_tokens: set[str]) -> float:
    """Overlap-coefficient style score: fraction of query tokens found in
    the document, which behaves better than strict Jaccard when documents
    are much longer than the query (the usual case here)."""
    if not query_tokens:
        return 0.0
    overlap = query_tokens & doc_tokens
    return len(overlap) / len(query_tokens)


def _sigmoid(x: float) -> float:
    import math

    return 1.0 / (1.0 + math.exp(-x))


def get_reranker() -> BaseReranker:
    """Factory: returns the configured reranker provider. Mirrors
    backend/ingestion/embedder.py's get_embedder() -- same setting,
    same graceful-fallback behavior."""
    mode = settings.docutrust_provider_mode.lower()

    if mode == "local":
        try:
            return LocalReranker()
        except Exception as exc:  # noqa: BLE001 - intentional broad catch for graceful fallback
            print(
                f"[reranker] Falling back to MockReranker: could not initialize "
                f"LocalReranker ({exc}). Set DOCUTRUST_PROVIDER_MODE=local on a "
                f"machine with HuggingFace access to use the real cross-encoder."
            )
            return MockReranker()

    return MockReranker()


def grade_relevance(top_score: float) -> str:
    """Maps the best candidate's reranker score to a CRAG grade.

    Thresholds live in config.py (relevance_threshold_correct /
    relevance_threshold_ambiguous) -- this function is intentionally just
    the comparison logic so the actual numbers are easy to find and tune
    in one place.
    """
    if top_score >= settings.relevance_threshold_correct:
        return "correct"
    if top_score >= settings.relevance_threshold_ambiguous:
        return "ambiguous"
    return "incorrect"
