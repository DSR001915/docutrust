"""
LLM generation provider abstraction.

Three implementations behind one interface:
  - AnthropicLLM: real Claude API calls. Requires ANTHROPIC_API_KEY.
  - OllamaLLM: real local model calls via a running Ollama server. Free, but
    requires Ollama installed and a model pulled (e.g. `ollama pull llama3.1:8b`).
  - MockLLM: template-based deterministic generation. No network, no API key,
    no local model server -- builds an answer directly from the highest-
    scoring retrieved chunks using simple extraction + a fixed citation
    format. This is what lets the LangGraph state machine, the generation
    node's citation formatting, and the corrective-generation node all be
    exercised end-to-end in network-restricted environments.

This file is also where the citation contract lives: every generation
provider returns (answer_text, citations), where citations is a list of
chunk_ids the answer actually drew on. Nodes downstream (backend/graph/
nodes.py) use this to render "[1]"-style references back to source
chunks/pages without re-parsing the answer text.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from backend.config import get_settings
from backend.reranker.cross_encoder import RankedCandidate

settings = get_settings()


@dataclass
class GenerationResult:
    answer: str
    citations: list[str]  # chunk_ids referenced in the answer
    used_web_fallback: bool = False


class BaseLLM(ABC):
    @abstractmethod
    def generate_answer(
        self, query: str, context_chunks: list[RankedCandidate], used_web_fallback: bool = False
    ) -> GenerationResult:
        """Generate a grounded answer from the provided context chunks,
        with inline citations back to chunk_ids."""

    @abstractmethod
    def rewrite_query(self, original_query: str, reason: str) -> str:
        """Reformulate a query that produced poor retrieval, to try again."""


_SYSTEM_PROMPT = """You are DocuTrust, an enterprise document Q&A assistant.

Rules you must follow:
1. Answer ONLY using the provided context chunks. Do not use outside knowledge
   unless the context is explicitly marked as web-fallback search results.
2. Every factual claim must be followed by a citation marker like [1], [2]
   referencing the chunk number it came from (chunks are numbered in the
   context below).
3. If the context does not contain enough information to answer the
   question, say so explicitly. Do not guess or fabricate an answer.
4. Be concise. Do not repeat the question back. Do not add unsupported
   caveats beyond what the context warrants."""


def _build_context_block(context_chunks: list[RankedCandidate]) -> str:
    lines = []
    for i, chunk in enumerate(context_chunks, start=1):
        lines.append(f"[{i}] (source: {chunk.document_name}, page {chunk.page_number})\n{chunk.text}")
    return "\n\n".join(lines)


def _extract_cited_chunk_ids(answer_text: str, context_chunks: list[RankedCandidate]) -> list[str]:
    """Parse [1], [2]-style markers out of the answer and map them back to
    real chunk_ids, so the frontend can render clickable source references."""
    import re

    cited_numbers = {int(n) for n in re.findall(r"\[(\d+)\]", answer_text)}
    chunk_ids = []
    for i, chunk in enumerate(context_chunks, start=1):
        if i in cited_numbers:
            chunk_ids.append(chunk.chunk_id)
    return chunk_ids


class AnthropicLLM(BaseLLM):
    def __init__(self, api_key: str | None = None, model: str | None = None):
        from anthropic import Anthropic  # local import: optional dep

        self._client = Anthropic(api_key=api_key or settings.anthropic_api_key)
        self.model = model or settings.anthropic_model

    def generate_answer(
        self, query: str, context_chunks: list[RankedCandidate], used_web_fallback: bool = False
    ) -> GenerationResult:
        context_block = _build_context_block(context_chunks)
        fallback_note = (
            "\n\nNOTE: some of the chunks above came from a web search fallback "
            "because internal documents did not sufficiently cover this question. "
            "Mention this in your answer if you use that information."
            if used_web_fallback
            else ""
        )

        user_message = (
            f"Context chunks:\n\n{context_block}{fallback_note}\n\n"
            f"Question: {query}\n\n"
            f"Answer the question using only the context above, with [n] citations."
        )

        response = self._client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )
        answer_text = "".join(
            block.text for block in response.content if getattr(block, "type", None) == "text"
        )
        citations = _extract_cited_chunk_ids(answer_text, context_chunks)
        return GenerationResult(answer=answer_text, citations=citations, used_web_fallback=used_web_fallback)

    def rewrite_query(self, original_query: str, reason: str) -> str:
        prompt = (
            f"The following search query returned poor/irrelevant results from a "
            f"document knowledge base. Reason: {reason}\n\n"
            f"Original query: {original_query}\n\n"
            f"Rewrite it as a single, clearer, more specific search query that is "
            f"more likely to retrieve relevant document chunks. Reply with ONLY "
            f"the rewritten query, no explanation."
        )
        response = self._client.messages.create(
            model=self.model,
            max_tokens=128,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(block.text for block in response.content if getattr(block, "type", None) == "text")
        return text.strip().strip('"')


class OllamaLLM(BaseLLM):
    def __init__(self, base_url: str | None = None, model: str | None = None):
        import ollama  # local import: optional dep

        self._client = ollama.Client(host=base_url or settings.ollama_host)
        self.model = model or settings.ollama_model

    def generate_answer(
        self, query: str, context_chunks: list[RankedCandidate], used_web_fallback: bool = False
    ) -> GenerationResult:
        context_block = _build_context_block(context_chunks)
        user_message = (
            f"Context chunks:\n\n{context_block}\n\nQuestion: {query}\n\n"
            f"Answer using only the context above, with [n] citations."
        )
        response = self._client.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
        )
        answer_text = response["message"]["content"]
        citations = _extract_cited_chunk_ids(answer_text, context_chunks)
        return GenerationResult(answer=answer_text, citations=citations, used_web_fallback=used_web_fallback)

    def rewrite_query(self, original_query: str, reason: str) -> str:
        prompt = (
            f"Rewrite this search query to be clearer and more specific, because "
            f"it returned poor results ({reason}). Original: {original_query}\n"
            f"Reply with ONLY the rewritten query."
        )
        response = self._client.chat(model=self.model, messages=[{"role": "user", "content": prompt}])
        return response["message"]["content"].strip().strip('"')


class MockLLM(BaseLLM):
    """Deterministic, template-based "generation" -- no model call at all.

    Extracts the most relevant sentence(s) from each top-scoring chunk and
    assembles them into an answer with citation markers. This is intentionally
    extractive rather than abstractive: it proves the citation-mapping and
    prompt-assembly code paths work correctly without needing a real LLM call,
    which is exactly the boundary worth testing automatically (see
    backend/tests/test_graph.py).

    Abstention floor: the system prompt's rule #3 ("if the context does not
    contain enough information... say so explicitly") is something a real
    LLM call would judge contextually. MockLLM has no judgment to apply, so
    it approximates the rule with a numeric floor on the chunk score.

    That floor is read from settings.relevance_threshold_ambiguous (the
    same threshold CRAG's grader uses for "ambiguous"), NOT a separate
    hardcoded constant. This matters because of a real, documented quirk
    of the mock embedder (backend/ingestion/embedder.py): its hash-bucket
    features give even semantically-unrelated text a noise-floor cosine
    similarity of roughly 0.05-0.2 (verified empirically -- see
    docs/EVAL_RESULTS.md), which is well above zero. A fixed abstention
    floor picked in isolation (e.g. "0.3") can land on the wrong side of
    that noise floor depending on which score scale it's compared against
    (raw cosine vs. the rerank-stage's blended overlap+distance score).
    Tying it to the already-calibrated CRAG threshold keeps naive RAG's
    abstention behavior comparable to CRAG's "ambiguous" cutoff rather than
    an independently-guessed number.
    """

    def generate_answer(
        self, query: str, context_chunks: list[RankedCandidate], used_web_fallback: bool = False
    ) -> GenerationResult:
        if not context_chunks:
            return GenerationResult(
                answer=(
                    "I don't have enough information in the available documents "
                    "to answer this question."
                ),
                citations=[],
                used_web_fallback=used_web_fallback,
            )

        # used_web_fallback=True means at least some chunks are web results,
        # which don't carry a meaningful "internal relevance" score in the
        # same sense -- only apply the abstention floor to pure-internal
        # generation (the naive RAG / CRAG "correct" path), not the
        # corrective-generate path where web fallback already happened.
        abstention_floor = settings.relevance_threshold_ambiguous
        if not used_web_fallback and all(c.score < abstention_floor for c in context_chunks):
            return GenerationResult(
                answer=(
                    "I don't have enough information in the available documents "
                    "to answer this question confidently. The most relevant "
                    "content found does not appear to address what was asked."
                ),
                citations=[],
                used_web_fallback=used_web_fallback,
            )

        sentences_with_source = []
        for i, chunk in enumerate(context_chunks, start=1):
            best_sentence = _best_matching_sentence(query, chunk.text)
            sentences_with_source.append((best_sentence, i))

        answer_parts = [f"{sentence} [{idx}]" for sentence, idx in sentences_with_source]
        prefix = (
            "Based on a web search fallback combined with available documents: "
            if used_web_fallback
            else "Based on the available documents: "
        )
        answer_text = prefix + " ".join(answer_parts)
        citations = [chunk.chunk_id for chunk in context_chunks]

        return GenerationResult(answer=answer_text, citations=citations, used_web_fallback=used_web_fallback)

    def rewrite_query(self, original_query: str, reason: str) -> str:
        """Deterministic query rewrite: strips stopwords AND expands a
        small set of domain synonyms, so the rewritten query is genuinely
        different from the original rather than just shorter. A real LLM
        rewrite (AnthropicLLM/OllamaLLM) would do this far more generally;
        this mock version exists so the correction loop has *something*
        real to demonstrate without a model call, and so it's honest about
        being a fixed lookup rather than true reformulation.

        Note: expansion only ever looks at whole-word matches against the
        synonym table and de-duplicates the output, so re-running this on
        an already-expanded query (as happens across correction loop
        iterations) doesn't compound -- "time-off" expanded once doesn't
        get re-expanded into "time off absence" on the next pass.
        """
        import re

        # Small, explicit synonym table -- intentionally not exhaustive.
        # Real generalization requires an LLM call; faking broad coverage
        # here would be more misleading than a short, honest list.
        synonyms = {
            "kept": "retained stored",
            "keep": "retain store",
            "store": "retain",
            "delete": "destroy remove",
            "removed": "destroyed deleted",
            "leave": "absence time-off",
            "pay": "reimbursement compensation",
            "paid": "reimbursed compensated",
            "rules": "policy procedure",
            "law": "regulation requirement",
        }

        words = re.findall(r"[a-zA-Z0-9]+", original_query.lower())
        question_words = {"what", "is", "the", "are", "how", "does", "do", "a", "an", "for", "of", "to"}
        keywords = [w for w in words if w not in question_words]

        seen: set[str] = set()
        expanded: list[str] = []
        for w in keywords:
            if w not in seen:
                expanded.append(w)
                seen.add(w)
            if w in synonyms:
                for syn_word in synonyms[w].replace("-", " ").split():
                    if syn_word not in seen:
                        expanded.append(syn_word)
                        seen.add(syn_word)

        if not expanded:
            return original_query
        return " ".join(expanded)


def _best_matching_sentence(query: str, text: str) -> str:
    import re

    sentences = re.split(r"(?<=[.!?])\s+", text)
    query_words = set(re.findall(r"[a-zA-Z0-9]+", query.lower()))

    best_sentence = sentences[0] if sentences else text
    best_overlap = -1
    for sentence in sentences:
        sentence_words = set(re.findall(r"[a-zA-Z0-9]+", sentence.lower()))
        overlap = len(query_words & sentence_words)
        if overlap > best_overlap:
            best_overlap = overlap
            best_sentence = sentence

    return best_sentence.strip()


def get_llm() -> BaseLLM:
    """Factory: returns the configured LLM provider based on
    settings.llm_provider ("anthropic" | "ollama" | "mock"), with graceful
    fallback to MockLLM if the configured provider can't be initialized
    (missing key, server not running, etc.)."""
    provider = settings.llm_provider.lower()

    if provider == "anthropic":
        try:
            return AnthropicLLM()
        except Exception as exc:  # noqa: BLE001
            print(f"[llm] Falling back to MockLLM: could not initialize AnthropicLLM ({exc}).")
            return MockLLM()

    if provider == "ollama":
        try:
            return OllamaLLM()
        except Exception as exc:  # noqa: BLE001
            print(f"[llm] Falling back to MockLLM: could not initialize OllamaLLM ({exc}).")
            return MockLLM()

    return MockLLM()
