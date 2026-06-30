"""
Embedding provider abstraction.

Two implementations behind one interface:
  - LocalEmbedder: real sentence-transformers bi-encoder (BAAI/bge-small-en-v1.5).
    Downloads weights from HuggingFace Hub on first use -- needs network access
    to huggingface.co (or a pre-populated HF cache / local model path).
  - MockEmbedder: deterministic, dependency-free stand-in with the same output
    shape (same dimensionality, cosine-similarity-meaningful vectors derived
    from token overlap). Used in network-restricted environments (CI runners,
    sandboxes) and in tests, so the rest of the pipeline -- chunking, vector
    store upsert, retrieval, reranking, the LangGraph state machine -- can be
    exercised end-to-end without requiring a model download.

Why this split instead of just skipping embeddings in restricted environments:
the whole point of DocuTrust is the CRAG control flow (retrieve -> grade ->
correct). That logic needs *some* vector to retrieve against. A mock embedder
that's at least topically consistent (same words -> similar vectors) lets the
graph logic, conditional edges, and trace logging all be tested for real,
while the swap to production-quality embeddings is a one-line config change.

Switch via config: settings.docutrust_provider_mode (see backend/config.py) --
see backend/config.py.
"""
from __future__ import annotations

import hashlib
import math
import re
from abc import ABC, abstractmethod

from backend.config import get_settings

settings = get_settings()

EMBEDDING_DIM = 384  # matches BAAI/bge-small-en-v1.5 output dimensionality

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Common function words contribute almost no topical signal but appear in
# nearly every sentence, which dilutes the mock embedder's already-limited
# ability to separate relevant from irrelevant text. Filtering them out is
# a standard, defensible preprocessing step (the same thing TF-IDF's
# `stop_words="english"` does) -- not a thumb on the scale for any one demo.
_STOPWORDS = frozenset(
    "a an the is are was were be been being of for to in on at by with "
    "and or but if then than this that these those it its as from into "
    "about must may can will shall not no do does did has have had "
    "our your their his her there here what which who whom".split()
)


class BaseEmbedder(ABC):
    """Interface every embedding provider must satisfy."""

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of chunk texts (used at ingestion time)."""

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string (used at retrieval time)."""


class LocalEmbedder(BaseEmbedder):
    """Real sentence-transformers bi-encoder. Requires the `sentence-transformers`
    package and network access to download model weights on first use."""

    def __init__(self, model_name: str | None = None):
        from sentence_transformers import SentenceTransformer  # local import: optional heavy dep

        self.model_name = model_name or settings.embedding_model
        self._model = SentenceTransformer(self.model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # normalize_embeddings=True -> cosine similarity via dot product downstream
        vectors = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        vector = self._model.encode([text], normalize_embeddings=True, show_progress_bar=False)
        return vector[0].tolist()


class MockEmbedder(BaseEmbedder):
    """Deterministic, hash-based bag-of-words + character-trigram embedder.

    Not semantically meaningful in the way a trained transformer is, but two
    texts that share vocabulary -- including partial/morphological overlap
    via character trigrams (e.g. "retention" / "retain" / "retained" all
    share trigrams) -- land closer together in cosine space than two texts
    that don't. That's enough signal to exercise retrieval, reranking
    thresholds, and the CRAG correction loop end-to-end in tests and
    network-restricted environments. It will NOT capture true synonymy
    (e.g. "incident response" vs "security breach procedure" share no
    surface forms) -- that gap is expected and is exactly why LocalEmbedder
    exists for anything beyond pipeline-logic testing.
    """

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed_one(text)

    @staticmethod
    def _embed_one(text: str) -> list[float]:
        vec = [0.0] * EMBEDDING_DIM
        raw_tokens = _TOKEN_RE.findall(text.lower())
        tokens = [t for t in raw_tokens if t not in _STOPWORDS and len(t) > 2]
        if not tokens:
            return vec

        features: list[str] = []
        features.extend(tokens)  # whole-word features
        for token in tokens:
            padded = f"#{token}#"
            features.extend(padded[i : i + 3] for i in range(len(padded) - 2))  # char trigrams

        for feature in features:
            # Stable hash -> bucket index, so the same feature always lands
            # in the same dimension across calls (deterministic, no model state).
            digest = hashlib.sha256(feature.encode("utf-8")).hexdigest()
            bucket = int(digest, 16) % EMBEDDING_DIM
            sign = 1.0 if int(digest[-1], 16) % 2 == 0 else -1.0
            # Whole-word matches count for more than sub-word trigram overlap.
            weight = 1.0 if feature in tokens else 0.3
            vec[bucket] += sign * weight

        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]


def get_embedder() -> BaseEmbedder:
    """Factory: returns the configured embedder provider.

    Reads settings.docutrust_provider_mode (see backend/config.py).
    Defaults to mock if local deps aren't installed or fail to initialize,
    so the app degrades gracefully instead of crashing on import.
    """
    mode = settings.docutrust_provider_mode.lower()

    if mode == "local":
        try:
            return LocalEmbedder()
        except Exception as exc:  # noqa: BLE001 - intentional broad catch for graceful fallback
            print(
                f"[embedder] Falling back to MockEmbedder: could not initialize "
                f"LocalEmbedder ({exc}). Set DOCUTRUST_PROVIDER_MODE=local on a "
                f"machine with HuggingFace access to use real embeddings."
            )
            return MockEmbedder()

    return MockEmbedder()
