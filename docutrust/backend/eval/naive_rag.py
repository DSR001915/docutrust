"""
Naive RAG baseline -- retrieve top-k, stuff into the LLM, generate. No
grading, no correction, no web fallback, no iteration cap.

This file exists for exactly one reason: backend/eval/run_eval.py uses it
as the "before" comparison against the full CRAG pipeline (backend/graph/
build_graph.py), so the eval can report a real, reproducible answer to "is
the corrective loop actually worth the extra latency and complexity?"
instead of just asserting that it is. Do not add grading or correction
logic here -- if you find yourself wanting to, that belongs in the CRAG
graph, not this baseline.

Deliberately no confidence score, no abstention: naive RAG's vector-store
distance is not a calibrated relevance signal (raw cosine distance between
arbitrary text pairs does not reliably separate "relevant" from
"irrelevant" without a reranking step -- see docs/EVAL_RESULTS.md for the
empirical measurement that motivated this). A naive pipeline that "stuffs
the top-k into the LLM" has nothing to threshold against, so it has no
principled way to decide when to abstain -- which IS the architectural gap
CRAG's cross-encoder reranking + grading step exists to close. Synthesizing
a fake confidence score here just to give naive RAG an abstention mechanism
would erase exactly the comparison this eval is supposed to make.
"""
from __future__ import annotations

from backend.config import get_settings
from backend.ingestion.embedder import get_embedder
from backend.ingestion.vector_store import get_vector_store
from backend.llm.provider import GenerationResult, get_llm
from backend.reranker.cross_encoder import RankedCandidate

settings = get_settings()


def naive_rag_answer(query: str) -> GenerationResult:
    """Embed -> retrieve top-k by vector distance only (no reranking, no
    grading) -> generate. This is deliberately the simplest possible RAG
    pipeline, representing what most "RAG tutorial" implementations do.

    Note on the score field: RankedCandidate.score is set to a fixed 1.0
    for every candidate here, NOT derived from vector distance. This is
    intentional -- see the module docstring. Passing a fixed high score
    means MockLLM's abstention-floor check (backend/llm/provider.py) never
    triggers for naive RAG, which is the correct behavior to model: naive
    RAG always presents whatever it retrieved as if it were a confident
    answer, never recognizing when retrieval was poor.
    """
    embedder = get_embedder()
    store = get_vector_store()
    llm = get_llm()

    query_vector = embedder.embed_query(query)
    candidates = store.query(query_vector, top_k=settings.rerank_top_k)

    ranked_like = [
        RankedCandidate(
            chunk_id=c.chunk_id,
            document_id=c.document_id,
            document_name=c.document_name,
            page_number=c.page_number,
            text=c.text,
            score=1.0,  # naive RAG has no relevance signal -- always "confident", by construction
        )
        for c in candidates
    ]

    return llm.generate_answer(query, ranked_like, used_web_fallback=False)
