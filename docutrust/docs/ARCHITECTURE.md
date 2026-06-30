# DocuTrust Architecture

## Why CRAG, not naive RAG

A naive RAG pipeline (embed query → fetch top-k by vector similarity → stuff into the LLM prompt → generate) has no mechanism to notice when retrieval failed. If the top-k chunks happen to be irrelevant — wrong document, ambiguous phrasing, a question the corpus simply doesn't cover — the LLM still generates a fluent, confident-sounding answer from whatever it was given. That's the central failure mode this project is built around: **retrieval failures present identically to retrieval successes unless something explicitly checks.**

Corrective RAG (CRAG) adds that check as an explicit, scored step in the pipeline rather than leaving it to the LLM's discretion.

## The state machine

```
                    ┌──────────┐
        ┌──────────▶│ retrieve │◀────────────────┐
        │           └────┬─────┘                 │
        │                │                       │
        │                ▼                       │
        │         ┌──────────────┐               │
        │         │grade_documents│              │
        │         └──────┬───────┘               │
        │                │                       │
        │       ┌────────┼────────────┐          │
        │       │        │            │          │
        │   correct  ambiguous/   ambiguous/      │
        │       │   incorrect    incorrect        │
        │       │   (iter<cap)   (iter>=cap)       │
        │       ▼        │            │          │
        │  ┌─────────┐   ▼            ▼      ┌────┴───────┐
        │  │generate │ ┌──────────┐ ┌────────┐│rewrite_query│
        │  └────┬────┘ │   END    │ │web_    │└────────────┘
        │       │       └──────────┘ │fallback│
        │       ▼                    └───┬────┘
        │  ┌─────────┐                   │
        │  │   END   │                   ▼
        │  └─────────┘          ┌────────────────────┐
        │                       │corrective_generate │
        │                       └──────────┬──────────┘
        │                                  │
        │                                  ▼
        │                            ┌──────────┐
        └────────────────────────────│   END    │
                                      └──────────┘
```

(See `backend/graph/build_graph.py` for the exact compiled graph definition, and `backend/graph/nodes.py` for every node's implementation — this diagram and that code are kept in sync by hand, so if they ever diverge, the code is the source of truth.)

### Nodes

| Node | File | What it does |
|---|---|---|
| `retrieve` | `nodes.py::retrieve_node` | Embeds the current query, pulls `RETRIEVAL_TOP_K_CANDIDATES` (default 20) from the vector store by cosine distance, then reranks down to `RERANK_TOP_K_FINAL` (default 5) with the cross-encoder. |
| `grade_documents` | `nodes.py::grade_documents_node` | Maps the top reranked score to `correct` / `ambiguous` / `incorrect` via `RELEVANCE_THRESHOLD_CORRECT` / `RELEVANCE_THRESHOLD_AMBIGUOUS`. Pure threshold comparison — no LLM call. |
| `rewrite_query` | `nodes.py::rewrite_query_node` | Reformulates the query (via the LLM provider's `rewrite_query()`) and increments `iteration_count`. |
| `web_fallback` | `nodes.py::web_fallback_node` | Pulls external search results when internal retrieval still isn't good enough after the rewrite budget is exhausted. |
| `generate` | `nodes.py::generate_node` | Direct-path generation when grade is `correct` on the first or a later pass. |
| `corrective_generate` | `nodes.py::corrective_generate_node` | Generation after correction: combines web fallback results with any internal chunks that cleared at least the `ambiguous` bar (chunks graded `incorrect` are dropped, not passed to the LLM — see the code comment in `nodes.py` for why). |

### The iteration cap (the most-asked interview question about this project)

`route_after_grading()` in `backend/graph/edges.py` is the single place this is enforced:

```python
if grade == "correct":
    return "generate"
if iteration_count < settings.max_correction_iterations:
    return "rewrite_query"
return "web_fallback"
```

Once `iteration_count` reaches `MAX_CORRECTION_ITERATIONS` (default 2), the graph stops rewriting and escalates to web fallback unconditionally — regardless of grade. This guarantees termination: there is no path through the graph that can loop indefinitely. `backend/tests/test_graph.py::test_correction_loop_never_exceeds_max_iterations` pins this down as an executable test, not just a comment.

## Why a cross-encoder, not an LLM self-judgment, for grading

Asking an LLM "is this context relevant? yes/no" is the obvious alternative, and many CRAG tutorials do exactly that. This project uses a cross-encoder instead (`backend/reranker/cross_encoder.py`) for three concrete reasons:

1. **A continuous, thresholdable score.** A cross-encoder outputs a number you can tune against (`RELEVANCE_THRESHOLD_CORRECT` / `_AMBIGUOUS`). An LLM's "yes/no" is harder to calibrate and drifts between prompts/model versions.
2. **Reproducibility.** The same query/chunk pair gets the same score every time from a cross-encoder. An LLM judgment call can vary run to run.
3. **Cost and latency.** A cross-encoder forward pass is far cheaper than an additional LLM round-trip, and this grading step runs on every single query (not just the corrected ones) — it's on the critical path for the entire system, not just the slow path.

## Provider abstraction (mock / local / API)

Every model-dependent component — embedder, reranker, LLM, web search — is written against a small interface with at least two implementations: a real one and a deterministic `Mock*` one. See:

- `backend/ingestion/embedder.py` (`LocalEmbedder` / `MockEmbedder`)
- `backend/reranker/cross_encoder.py` (`LocalReranker` / `MockReranker`)
- `backend/llm/provider.py` (`AnthropicLLM` / `OllamaLLM` / `MockLLM`)
- `backend/graph/web_search.py` (`TavilySearch` / `MockWebSearch`)

Switches are centralized in `backend/config.py`'s `Settings` class (`docutrust_provider_mode`, `llm_provider`, `web_search_provider`) — not scattered `os.getenv()` calls — so there's one place to look to know what's actually running. Mock mode requires no API keys, no HuggingFace access, and no live network calls, which is what this repo's own test suite and eval script run against by default. Flipping to real models is a config change, not a code change.

This isn't just a convenience for restricted environments — it's the same interface boundary you'd want in production if you ever needed to swap embedding models or move from Anthropic to a self-hosted model. The abstraction would exist either way; restricted-environment development just forced it to exist from day one.

## Known limitations of the mock providers (read before trusting eval numbers at face value)

- **`MockEmbedder`** is a hashed bag-of-words + character-trigram scheme, not a trained semantic model. It correctly ranks lexically-overlapping text higher than non-overlapping text, but it cannot capture true synonymy (e.g. "incident response" vs. "security breach procedure" share no surface forms and won't score as related). Empirically, even semantically unrelated text pairs show a baseline cosine similarity around 0.05–0.2 with this scheme (verified directly — see `docs/EVAL_RESULTS.md`), which is why naive RAG's lack of a calibrated relevance signal is treated as a real architectural property in the eval, not patched around with a synthetic confidence score.
- **`MockReranker`** blends token-overlap with the vector store's own distance ranking — a reasonable stand-in for grading-logic testing, but it will not catch cases where a real cross-encoder's deeper semantic matching would differ from surface overlap.
- **`MockLLM`** is purely extractive (best-matching sentence per chunk, not synthesized), and applies a hard abstention floor on `relevance_threshold_ambiguous` to avoid presenting clearly-irrelevant chunks as answers (see the docstring in `backend/llm/provider.py` for exactly why that floor is tied to the same threshold the CRAG grader uses, not an independently-chosen number).

None of this affects the CRAG control-flow logic itself (the state machine, routing, and iteration cap are exercised identically regardless of provider) — it affects how literally to read the eval's absolute numbers versus its directional conclusion. Re-run `backend/eval/run_eval.py` with `DOCUTRUST_PROVIDER_MODE=local LLM_PROVIDER=anthropic` on a machine with HuggingFace + Anthropic API access to get numbers from the real model stack.

## Data flow: ingestion

```
PDF upload → parser.py (pypdf, page-by-page text extraction)
           → chunker.py (recursive character split + overlap, page-number-tagged)
           → embedder.py (bi-encoder, batched)
           → vector_store.py (ChromaDB upsert, content-hash document_id for idempotent re-upload)
           → document_store.py (MongoDB metadata, or in-memory fallback)
```

Re-uploading the exact same file content is idempotent: `document_id` is derived from a SHA-256 content hash (`backend/ingestion/service.py::_content_hash_id`), and any existing chunks for that ID are deleted before the new ones are inserted — so re-running ingestion never silently duplicates chunks in the vector store.

## Data flow: query

```
WebSocket/REST query → query_service.py → compiled CRAG graph (build_graph.py)
                                          → node-by-node execution, trace appended at each step
                                          → trace_logger.py (MongoDB, or in-memory fallback)
                                          → final answer + citations returned/streamed to frontend
```

The WebSocket path (`/ws/query` in `backend/main.py`) uses LangGraph's `.astream()` to emit one event per node transition as it happens, which is what powers the frontend's live "Verification Ledger" — the same underlying graph execution as the synchronous `/api/query` REST endpoint, just observed incrementally instead of awaited to completion.
