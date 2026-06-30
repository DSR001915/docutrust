"""
Web fallback search provider -- triggered only when internal retrieval
grades as "incorrect" even after the rewrite-and-retry loop. This is the
last resort in the CRAG correction path, not a general-purpose search tool.

Two implementations:
  - TavilySearch: real web search via the Tavily API (free tier available),
    purpose-built for LLM agent use (returns clean, pre-summarized results
    rather than raw HTML).
  - MockWebSearch: deterministic stand-in returning a clearly-labeled
    placeholder result, so the corrective_generate node and citation
    handling for web-sourced content can be tested without an API key.
"""
from __future__ import annotations

import uuid
from abc import ABC, abstractmethod

from backend.config import get_settings
from backend.reranker.cross_encoder import RankedCandidate

settings = get_settings()


class BaseWebSearch(ABC):
    @abstractmethod
    def search(self, query: str, max_results: int = 3) -> list[RankedCandidate]:
        """Run a web search and return results normalized into the same
        RankedCandidate shape used by internal retrieval, so downstream
        generation code doesn't need to special-case the source."""


class TavilySearch(BaseWebSearch):
    def __init__(self, api_key: str | None = None):
        from tavily import TavilyClient  # local import: optional dep

        self._client = TavilyClient(api_key=api_key or settings.tavily_api_key)

    def search(self, query: str, max_results: int = 3) -> list[RankedCandidate]:
        response = self._client.search(query=query, max_results=max_results)
        results = []
        for item in response.get("results", []):
            results.append(
                RankedCandidate(
                    chunk_id=f"web-{uuid.uuid4()}",
                    document_id="web-search",
                    document_name=item.get("url", "web search result"),
                    page_number=0,
                    text=item.get("content", ""),
                    score=float(item.get("score", 0.5)),
                )
            )
        return results


class MockWebSearch(BaseWebSearch):
    """Deterministic stand-in. Clearly labels its output as a placeholder
    so nobody mistakes it for a real answer if PROVIDER_MODE is left on
    mock by accident in a non-demo context."""

    def search(self, query: str, max_results: int = 3) -> list[RankedCandidate]:
        return [
            RankedCandidate(
                chunk_id=f"web-mock-{uuid.uuid4()}",
                document_id="web-search-mock",
                document_name="[mock web search -- no live network call made]",
                page_number=0,
                text=(
                    f"[MockWebSearch placeholder] No real web search was performed for "
                    f"the query '{query}'. Set WEB_SEARCH_PROVIDER=tavily and provide "
                    f"TAVILY_API_KEY to enable real web fallback search."
                ),
                score=0.5,
            )
        ]


def get_web_search() -> BaseWebSearch:
    provider = settings.web_search_provider.lower()

    if provider == "tavily":
        try:
            return TavilySearch()
        except Exception as exc:  # noqa: BLE001
            print(f"[web_search] Falling back to MockWebSearch: could not initialize TavilySearch ({exc}).")
            return MockWebSearch()

    return MockWebSearch()
