# DocuTrust 🔍

> A self-correcting RAG platform that grades its own retrieval quality before answering — and corrects itself when it's wrong, instead of confidently hallucinating from bad context.

DocuTrust implements the **Corrective RAG (CRAG)** pattern as an explicit [LangGraph](https://github.com/langchain-ai/langgraph) state machine: every retrieval is scored by a cross-encoder, graded against tuned thresholds, and routed down one of three paths — answer directly, rewrite the query and retry, or escalate to a web-search fallback — with a hard cap that guarantees the correction loop always terminates.

## The problem

Naive RAG (embed → fetch top-k → stuff into the LLM → generate) has no way to notice when retrieval failed. If the retrieved chunks are irrelevant, the LLM still produces a fluent, confident answer from whatever it was given — retrieval failure and retrieval success look identical from the outside. That's the gap CRAG closes: an explicit, scored checkpoint between retrieval and generation.

## Results

Evaluated on 25 hand-written questions (20 in-corpus, 5 deliberately out-of-corpus) against a sample corporate policy handbook — see [`docs/EVAL_RESULTS.md`](docs/EVAL_RESULTS.md) for the full per-question breakdown and methodology notes.

| Metric | Naive RAG | CRAG |
|---|---|---|
| Faithfulness (in-corpus) | 95.0% | 95.0% |
| Correctly handled (out-of-corpus) | **0.0%** | **100.0%** |

Naive RAG has no mechanism to recognize when retrieval missed — it confidently presents whatever it retrieved, every time, regardless of relevance. CRAG's grading step catches this and either abstains or discloses a web-search fallback, on every out-of-corpus question in this eval.

*(These numbers come from the project's deterministic mock providers — see [Provider modes](#provider-modes-mock--local--api) below. The control-flow result — naive RAG never catches bad retrieval, CRAG always does — is architectural and provider-independent; the absolute faithfulness numbers will shift with real embedding/reranker/LLM models.)*

## Architecture

```
retrieve → grade_documents ─┬─→ generate ──────────────────────→ END
                ▲            │
                │            ├─→ rewrite_query → (loop back to retrieve, capped)
                │            │
                │            └─→ web_fallback → corrective_generate → END
                └─────────────────────────┘
```

Full diagram, node-by-node breakdown, and the iteration-cap termination guarantee: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

**Stack:**

| Layer | Choice |
|---|---|
| Agent orchestration | LangGraph (explicit state machine, not implicit chains) |
| Embeddings | `BAAI/bge-small-en-v1.5` (local, free) |
| Reranker / grader | `BAAI/bge-reranker-v2-m3` cross-encoder (local, free) |
| Vector store | ChromaDB |
| Backend | FastAPI + WebSocket streaming |
| Trace/metadata store | MongoDB (graceful in-memory fallback if unreachable) |
| LLM | Anthropic Claude or Ollama (local), pluggable |
| Web fallback search | Tavily |
| Frontend | Vanilla HTML/CSS/JS, no build step |

## Quickstart

```bash
git clone <this-repo>
cd docutrust
docker-compose up
```

Open `http://localhost:8000`. The default config runs entirely in **mock mode** — no API keys, no model downloads, no external network calls required. Upload `sample_data/corporate_policy_handbook.pdf`, ask it a question, and watch the live verification ledger.

To use real models: copy `.env.example` to `.env`, set `DOCUTRUST_PROVIDER_MODE=local` (downloads the embedder + reranker from HuggingFace on first use) and `LLM_PROVIDER=anthropic` (with an API key) or `LLM_PROVIDER=ollama` (with a local Ollama server running).

### Running without Docker

```bash
pip install -r requirements-mock.txt   # or requirements.txt for real local models
python scripts/generate_sample_pdf.py  # regenerates sample_data/corporate_policy_handbook.pdf
uvicorn backend.main:app --reload
```

MongoDB is optional in mock-mode local dev — if it's unreachable, trace logging and document metadata fall back to in-memory storage automatically (you'll see a one-line warning in the logs, not a crash).

### Running tests

```bash
pytest backend/tests/ -v
```

25 tests covering chunking (including a regression test for overlap-duplication bugs), the reranker's grading thresholds, and — most importantly — the CRAG graph's routing logic and iteration-cap termination guarantee.

### Running the eval

```bash
python -m backend.eval.run_eval
```

Compares naive RAG against CRAG on the golden dataset and regenerates `docs/EVAL_RESULTS.md`.

## Provider modes: mock / local / API

Every model-dependent component (embedder, reranker, LLM, web search) is written against a small interface with both a real implementation and a deterministic mock — see `backend/config.py`'s `Settings` class for the three independent switches (`docutrust_provider_mode`, `llm_provider`, `web_search_provider`). This isn't just a dev convenience: it's the same abstraction boundary you'd want in production to swap embedding models or LLM vendors without touching the graph logic. Full rationale and the mock providers' known limitations: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md#known-limitations-of-the-mock-providers-read-before-trusting-eval-numbers-at-face-value).

## Project structure

```
backend/
├── config.py              # centralized settings (provider modes, CRAG thresholds, etc.)
├── main.py                 # FastAPI app: REST + WebSocket routes
├── ingestion/               # PDF parsing, chunking, embedding, vector store, idempotent upload
├── graph/                   # the CRAG state machine: nodes, edges, build_graph, state
├── reranker/                 # cross-encoder grading
├── llm/                       # generation providers (Anthropic / Ollama / mock)
├── db/                         # MongoDB clients with in-memory fallback
├── eval/                        # naive-vs-CRAG comparison + golden dataset
└── tests/                        # pytest suite (25 tests)
frontend/                          # vanilla JS, WebSocket-driven live trace UI
docs/                                # architecture write-up + eval results
sample_data/                          # generated sample PDF used throughout
```

## What I'd add with more time

- A second grading pass on web-fallback results (right now they're trusted once retrieved, not re-graded)
- Hybrid retrieval (BM25 + vector) for queries with exact keywords/jargon the embedding model under-weights
- RAGAS-based eval scoring as an alternative to the current substring-match proxy, for subtler hallucination detection
- Multi-document corpus testing — the current eval is single-document by design (clarity over scale), but production relevance grading behaves differently across many overlapping documents
