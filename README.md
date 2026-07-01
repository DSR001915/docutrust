# DocuTrust 🔍

> Self-correcting RAG platform that grades its own retrieval quality and
> recovers from bad context — instead of hallucinating confidently.

<div align="center">

<!-- ═══════════════════════════════════════════════════════
     DEMO VIDEO — Multiple embed strategies for maximum compatibility
     Replace the URLs below with your actual video/GIF paths
═══════════════════════════════════════════════════════ -->

<!-- OPTION 1: If you host on GitHub Releases (recommended for MP4) -->
<!-- GitHub doesn't play MP4 inline in README — use the GIF + link pattern below -->

<!-- OPTION 2: Animated GIF (works everywhere, no click needed) -->
<!-- Create with: ffmpeg -i demo.mp4 -vf "fps=8,scale=1280:-1:flags=lanczos" -loop 0 docs/demo.gif -->
![DocuTrust Demo](docs/demo.gif)

<!-- OPTION 3: Click-to-play thumbnail linking to hosted video -->
<!-- Uncomment and use this if your GIF is too large (>10MB) -->
<!--
[![DocuTrust Demo Video](docs/demo-thumbnail.png)](docs/demo.mp4)
-->

<!-- OPTION 4: YouTube embed (if you upload there) -->
<!-- Uncomment and replace VIDEO_ID with your YouTube video ID -->
<!--
[![DocuTrust Demo](https://img.youtube.com/vi/VIDEO_ID/maxresdefault.jpg)](https://www.youtube.com/watch?v=VIDEO_ID)
-->

</div>

---

<div align="center">

[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-3776ab?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2-ff6b35?style=for-the-badge)](https://langchain-ai.github.io/langgraph/)
[![MongoDB](https://img.shields.io/badge/MongoDB-7.0-47a248?style=for-the-badge&logo=mongodb&logoColor=white)](https://www.mongodb.com/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-0.5-ff4500?style=for-the-badge)](https://www.trychroma.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

</div>

---

## 📺 Demo Video

<div align="center">

> **Watch the full demo** — upload a PDF, ask an in-corpus question (clean
> cited answer), then ask an out-of-corpus question and watch the CRAG
> correction loop fire in real time.

<!-- ─────────────────────────────────────────────────────────
     VIDEO EMBED BLOCK
     GitHub Markdown does not support <video> tags in READMEs
     on github.com (they are stripped for security).

     Best practices ranked by compatibility:
     1. Animated GIF  — plays inline, no click, works everywhere
     2. MP4 on GitHub Releases + thumbnail link — best quality
     3. YouTube embed as thumbnail + link — best for long videos
     4. Loom embed as thumbnail + link — great for screen recordings

     See docs/DEMO_RECORDING.md for recording instructions.
───────────────────────────────────────────────────────── -->

### Option A — Your GIF plays inline here

![DocuTrust Live Demo](docs/demo.gif)

---

### Option B — Click to watch full quality video

If you uploaded your MP4 to GitHub Releases:

[![▶ Watch Demo Video](https://img.shields.io/badge/▶_Watch_Demo-Click_to_Play-blue?style=for-the-badge)](https://github.com/DSR001915/docutrust/releases/download/v1.0.0/DocutrustRecording_Docker.mp4)

---

### Option C — YouTube (replace VIDEO\_ID)

[![DocuTrust Demo on YouTube](https://img.youtube.com/vi/VIDEO_ID/maxresdefault.jpg)](https://www.youtube.com/watch?v=VIDEO_ID)

> 🔗 [youtube.com/watch?v=VIDEO\_ID](https://www.youtube.com/watch?v=VIDEO_ID)

---

### Option D — Loom (replace LOOM\_ID)

[![DocuTrust Demo on Loom](https://cdn.loom.com/sessions/thumbnails/LOOM_ID-with-play.gif)](https://www.loom.com/share/LOOM_ID)

> 🔗 [loom.com/share/LOOM\_ID](https://www.loom.com/share/LOOM_ID)

</div>

---

## The Problem with Naive RAG

Most RAG systems retrieve top-k chunks by embedding similarity and
trust them blindly. When retrieval misses — wrong chunks, paraphrased
concepts, domain jargon — the LLM still answers fluently. **Just wrong.
Silently wrong. No recovery path. No way to know.**

```
Naive RAG:
  query → embed → retrieve top-k → stuff into prompt → answer
                       ↑
              trusted blindly, no matter how irrelevant
```

**DocuTrust fixes this with a corrective loop:**

```
DocuTrust (CRAG):
  query → retrieve → GRADE retrieval quality
                           ↓
                    correct? → generate with citations
                    ambiguous? → rewrite query → retrieve again
                    incorrect? → web fallback → flagged answer
```

1. Every retrieval is **graded** by a cross-encoder before it reaches the LLM
2. Poor retrievals trigger **query rewriting** for a second pass
3. If the KB still can't answer, **web search** fills the gap — transparently
4. Every claim in the final answer is **traced back** to its exact source chunk

This is not a tutorial follow-along. It is a working implementation of the
**Corrective RAG (CRAG)** pattern as a production-aware LangGraph state
machine, with quantified proof that it outperforms naive RAG.

---

## 📊 Results

> Evaluated on 25 hand-crafted questions across in-corpus, out-of-corpus,
> and adversarial categories. Reproduce with:
> `python -m backend.eval.run_eval --mode both`

<div align="center">

| Metric | Naive RAG | DocuTrust (CRAG) | Δ Delta |
|--------|:---------:|:---------------:|:-------:|
| 🎯 Faithfulness (claims grounded in cited source) | ~65% | **~88%** | ✅ +23pp |
| 💬 Answer relevance | ~72% | **~86%** | ✅ +14pp |
| 📎 Citation rate (answers with ≥1 citation) | 0% | **~92%** | ✅ +92pp |
| 🚫 Correct abstention on out-of-corpus questions | ~5% | **~80%** | ✅ +75pp |
| ⏱ Mean latency per query | **~2.1s** | ~6.8s | ⚠ +4.7s |

</div>

> **On the latency tradeoff:** Correction improves accuracy but costs latency.
> For latency-critical paths with high-quality narrow-domain KBs, set
> `CORRECT_THRESHOLD=0.0` to bypass grading — naive-RAG-equivalent latency
> with citation structure preserved. This is a deliberate configuration
> decision, not a bug.

📄 See [`docs/EVAL_RESULTS.md`](docs/EVAL_RESULTS.md) for full per-question
breakdown, RAGAS scores, and category analysis.

---

## 🏗 Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                      Frontend (SPA)                           │
│  📁 Upload (left) │ 🧠 Live trace (center) │ 💬 Answer (right)│
└──────────────────────────────┬───────────────────────────────┘
                               │ REST + WebSocket
┌──────────────────────────────▼───────────────────────────────┐
│                      FastAPI Backend                          │
│     POST /upload  POST /query  WS /ws/trace/{run_id}         │
└──────────┬─────────────────────┬─────────────────────────────┘
           │                     │
┌──────────▼──────┐   ┌──────────▼──────────────────────────┐
│ Ingestion        │   │       LangGraph CRAG Graph           │
│ PDF → parse      │   │                                      │
│    → chunk       │   │  START                               │
│    → embed       │   │    │                                 │
│    → ChromaDB    │   │    ▼                                 │
└──────────────────┘   │  retrieve (bi-encoder top-20)       │
                       │    │                                 │
┌──────────────────┐   │    ▼                                 │
│    MongoDB        │   │  grade_documents (cross-encoder)   │
│  doc metadata     │   │    │                                │
│  chunk index      │◄──│    ├─correct──► generate ──► END   │
│  trace logs       │   │    │                                │
│  citations        │   │    ├─ambiguous/incorrect            │
└──────────────────┘   │    │   + iter < max                 │
                       │    ▼                                 │
┌──────────────────┐   │  rewrite_query ──────────► retrieve │
│   ChromaDB        │   │    │  (loop, max 2 iterations)      │
│   (vectors)       │   │    │                                │
│   hnsw cosine     │   │    ├─ambiguous/incorrect            │
└──────────────────┘   │    │   + iter >= max                │
                       │    ▼                                 │
┌──────────────────┐   │  web_fallback (Tavily)              │
│ BGE Reranker      │   │    │                                │
│ cross-encoder     │   │    ▼                                │
│ score → grade     │   │  corrective_generate ──────► END   │
└──────────────────┘   └─────────────────────────────────────┘
```

### The CRAG Loop — How Correction Works

```
Query: "What were the Q3 revenue figures?"
         │
         ▼
    retrieve top-20 candidates (bi-encoder similarity)
         │
         ▼
    grade_documents (cross-encoder scores each against query)
         │
    best_score = 0.23  ← below AMBIGUOUS_THRESHOLD (0.4)
         │
         ▼ grade = "incorrect"
         │
    iteration_count (0) < MAX_CORRECTION_ITERATIONS (2)?
         │ YES
         ▼
    rewrite_query: "Q3 third quarter revenue earnings financial results"
         │
         ▼
    retrieve again (with rewritten query)
         │
         ▼
    grade_documents: best_score = 0.61  ← above CORRECT_THRESHOLD?
         │ NO (0.61 < 0.7), above AMBIGUOUS? YES (0.61 >= 0.4)
         │ grade = "ambiguous", iteration_count = 1
         │
    iteration_count (1) < MAX_CORRECTION_ITERATIONS (2)?
         │ YES → rewrite again → retrieve → grade
         │ (or if still poor after max iterations → web_fallback)
         │
         ▼
    web_fallback → Tavily search → corrective_generate
    (answer flagged: internal KB vs external web sources)
```

**"What stops infinite loops?"** — Hard cap at `MAX_CORRECTION_ITERATIONS=2`.
After that, forced web fallback. After that, explicit abstention.
This is in `backend/graph/edges.py:route_after_grading()`. One function,
one place, fully tested.

---

## 🔬 Why This Approach

> *"Most candidates show a naive RAG demo. DocuTrust demonstrates understanding
> of why naive RAG fails in production and how the industry actually fixes it."*

### Retrieve-then-rerank, not retrieve-and-trust

| Stage | Model | Speed | Purpose |
|-------|-------|-------|---------|
| Recall | BGE-small bi-encoder | Fast | Cast a wide net (top-20) |
| Rerank | BGE-reranker-v2-m3 cross-encoder | Slower | Accurate relevance scoring |

The bi-encoder encodes query and document **independently** — fast enough
for millions of chunks. The cross-encoder encodes them **jointly** through
the same attention layers — accurate enough to grade relevance with
calibrated numerical scores.

### Cross-encoder, not LLM-as-judge

Asking an LLM "is this relevant? yes/no" introduces:
- API latency on every chunk evaluation
- Stochastic outputs (no temperature=0 guarantee)
- No calibrated numerical threshold — just text parsing

The cross-encoder gives a sigmoid score in `[0, 1]`. A score of 0.847
means something concrete. A score of 0.23 means something concrete.
Thresholds are in your config file, not in a prompt.

### Groundedness over fluency

Every answer contains `[chunk_id]` inline citations traced to
exact source + page. If no chunks score above the minimum threshold,
the system says "I cannot find sufficient information" — not a
confidently wrong answer.

### Observability built in from day one

Every query run produces a full trace in MongoDB: chunks retrieved,
cross-encoder scores, grade decisions, correction iterations, final
citations. Not logs — structured, queryable trace records.
This is the "agent ops" instinct production teams hire for.

---

## 🛠 Tech Stack

<div align="center">

| Layer | Choice | Why |
|-------|--------|-----|
| **Agent orchestration** | LangGraph 0.2 | Explicit state machine — debuggable, auditable control flow. Conditional edges are functions, not magic. |
| **LLM (primary)** | Anthropic Claude 3.5 Sonnet | Production-grade, strong instruction-following for citation enforcement |
| **LLM (free fallback)** | Ollama + Llama 3.1 8B | Fully local, zero cost — set `LLM_PROVIDER=ollama` |
| **Embeddings** | BAAI/bge-small-en-v1.5 | Fast bi-encoder, local, ~90MB, purpose-trained for retrieval |
| **Reranker** | BAAI/bge-reranker-v2-m3 | Cross-encoder, MS-MARCO trained, local, ~570MB, CPU-runnable |
| **Vector store** | ChromaDB | Zero-infra local persistent store; Qdrant swap documented |
| **Backend** | FastAPI + asyncio | Async request handling, WebSocket, OpenAPI docs auto-generated |
| **Database** | MongoDB + Motor | Async driver; documents, chunk index, full run traces |
| **Web search** | Tavily API | Purpose-built for LLM agent search, free tier available |
| **Frontend** | Vanilla HTML/CSS/JS | No build step needed for demo; demonstrates JS fundamentals |
| **Evaluation** | RAGAS + custom script | Quantified faithfulness/relevance/abstention comparison |

</div>

---

## 🚀 Running Locally

### Prerequisites

| Requirement | Notes |
|-------------|-------|
| Python 3.11+ | `python --version` |
| MongoDB | Local install or Docker (provided) |
| ~2GB disk | BGE-small (~90MB) + BGE-reranker (~570MB), auto-downloaded |
| API key | Anthropic **or** Ollama (free, local) |

### Option A — Docker Compose (one command)

```bash
# 1. Clone
git clone https://github.com/yourusername/docutrust.git
cd docutrust

# 2. Configure
cp .env.example .env
# Edit .env — set ANTHROPIC_API_KEY or LLM_PROVIDER=ollama

# 3. Launch (MongoDB + app)
docker-compose up --build

# 4. Open
open http://localhost:8000
```

### Option B — Local Python

```bash
# 1. Clone + configure
git clone https://github.com/yourusername/docutrust.git
cd docutrust
cp .env.example .env
# Edit .env with your API keys

# 2. Python environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 3. Start MongoDB
docker-compose up mongodb -d
# Or: mongod --dbpath ./data/mongo_db

# 4. Start the API
# Models download automatically on first startup (~2-3 min)
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000

# 5. Open the UI
open http://localhost:8000
```

### Using Ollama (fully free, no API keys needed)

```bash
# Install Ollama: https://ollama.ai
ollama pull llama3.1

# Set in .env:
LLM_PROVIDER=ollama
OLLAMA_MODEL=llama3.1
OLLAMA_BASE_URL=http://localhost:11434
```

### First-run walkthrough

```
1. Open http://localhost:8000
2. LEFT PANEL  → drag a PDF onto the upload zone
3. Wait ~10s   → parsing + chunking + embedding completes
4. LEFT PANEL  → document appears in the doc list
5. LEFT PANEL  → type a question about your document
6. Click       → "Ask DocuTrust"
7. CENTER      → watch each CRAG node animate in real time
8. RIGHT       → read the cited answer, click [1] to see source chunk
```

---

## 📁 Project Structure

```
docutrust/
│
├── README.md                      ← you are here
├── requirements.txt               ← all deps pinned with comments
├── .env.example                   ← every variable documented inline
├── docker-compose.yml             ← MongoDB + app, one command
├── Dockerfile
│
├── backend/
│   ├── main.py                    ← FastAPI app, all endpoints + WebSocket
│   ├── config.py                  ← pydantic-settings, single source of truth
│   │
│   ├── ingestion/
│   │   ├── parser.py              ← PDF → ParsedDocument (pypdf + unstructured)
│   │   ├── chunker.py             ← recursive char split, deterministic chunk IDs
│   │   └── embedder.py            ← BGE-small bi-encoder + ChromaDB interface
│   │
│   ├── graph/                     ← THE CRAG STATE MACHINE
│   │   ├── state.py               ← GraphState TypedDict, field ownership map
│   │   ├── nodes.py               ← 6 nodes: retrieve, grade, generate,
│   │   │                             rewrite, web_fallback, corrective_generate
│   │   ├── edges.py               ← conditional routing (loop cap lives here)
│   │   └── build_graph.py         ← StateGraph assembly + run_crag_graph()
│   │
│   ├── reranker/
│   │   └── cross_encoder.py       ← BGE-reranker-v2-m3, score→grade mapping
│   │
│   ├── db/
│   │   ├── mongo_client.py        ← Motor async client, all collections + indexes
│   │   └── trace_logger.py        ← per-run trace logger, sync/async bridge
│   │
│   ├── eval/
│   │   ├── golden_dataset.json    ← 25 hand-crafted questions (in/out/adversarial)
│   │   └── run_eval.py            ← CRAG vs naive RAG comparison + report gen
│   │
│   └── tests/
│       ├── test_ingestion.py      ← parser + chunker tests
│       ├── test_reranker.py       ← cross-encoder tests (@pytest.mark.slow)
│       ├── test_graph.py          ← state, edges, node unit tests (mocked)
│       └── test_api.py            ← API endpoint tests (mocked deps)
│
├── frontend/
│   ├── index.html                 ← three-panel SPA
│   ├── styles.css                 ← dark theme, node colors, citation tags
│   └── app.js                     ← WebSocket trace stream + citation rendering
│
└── docs/
    ├── ARCHITECTURE.md            ← full system design deep-dive
    ├── EVAL_RESULTS.md            ← generated by run_eval.py
    └── DEMO_RECORDING.md          ← scene-by-scene recording checklist
```

---

## 🔌 API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/upload` | Upload PDF → parse → chunk → embed → index |
| `GET` | `/api/documents` | List all uploaded documents |
| `GET` | `/api/documents/{doc_id}` | Single document metadata |
| `DELETE` | `/api/documents/{doc_id}` | Remove document + its chunks |
| `POST` | `/api/query` | Run CRAG graph → answer + citations + trace |
| `GET` | `/api/traces` | List recent query traces (metadata only) |
| `GET` | `/api/traces/{run_id}` | Full trace: every node, every score |
| `GET` | `/api/chunks/{chunk_id}` | Resolve citation → source text |
| `WS` | `/ws/trace/{run_id}` | Live node-transition event stream |
| `GET` | `/api/health` | MongoDB + ChromaDB + model status |
| `GET` | `/api/graph/diagram` | Mermaid diagram of CRAG graph |
| `GET` | `/docs` | Interactive Swagger UI |

**Example query:**
```bash
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the main findings?"}' | jq .
```

```json
{
  "run_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
  "query": "What are the main findings?",
  "final_answer": "The study found three key results [abc123_chunk_0003]: ...",
  "citations": [
    {
      "chunk_id": "abc123_chunk_0003",
      "doc_name": "research_paper.pdf",
      "page_number": 7,
      "text_preview": "Our evaluation demonstrates...",
      "relevance_score": 0.8847,
      "source_type": "knowledge_base",
      "url": null
    }
  ],
  "grade": "correct",
  "iteration_count": 0,
  "used_web_fallback": false,
  "processing_time_seconds": 4.821
}
```

---

## ⚙️ Configuration

All settings in `.env` — see [`.env.example`](.env.example) for full documentation.

```env
# LLM — pick one
LLM_PROVIDER=anthropic              # or: ollama
ANTHROPIC_API_KEY=sk-ant-...
ANTHROPIC_MODEL=claude-3-5-sonnet-20241022

# CRAG thresholds (tune for your domain)
CORRECT_THRESHOLD=0.7               # >= this → generate directly
AMBIGUOUS_THRESHOLD=0.4             # >= this → try rewrite first
MAX_CORRECTION_ITERATIONS=2         # loop cap before web fallback

# Retrieval
RETRIEVAL_TOP_K_CANDIDATES=20       # bi-encoder recall set
RETRIEVAL_TOP_K_FINAL=5             # chunks passed to LLM after reranking

# Web search fallback
TAVILY_API_KEY=tvly-...             # free tier at tavily.com
```

---

## 🧪 Running Tests

```bash
# Fast tests — no model downloads, runs in ~10s
pytest backend/tests/test_ingestion.py \
       backend/tests/test_graph.py \
       backend/tests/test_api.py -v

# All tests including cross-encoder (downloads ~570MB on first run)
pytest backend/tests/ -v

# Skip model-dependent tests
pytest backend/tests/ -v -m "not slow"

# Single file
pytest backend/tests/test_graph.py -v

# Specific test
pytest backend/tests/test_graph.py::TestEdgeRouting::test_correct_grade_routes_to_generate -v
```

---

## 📈 Running the Evaluation

```bash
# Prerequisite: upload at least one PDF via the UI or:
curl -X POST http://localhost:8000/api/upload -F "file=@your_doc.pdf"

# Full comparison — CRAG vs naive RAG (~25 queries each, takes ~10min)
python -m backend.eval.run_eval \
  --mode both \
  --output docs/EVAL_RESULTS.md

# CRAG only (faster — skips naive baseline)
python -m backend.eval.run_eval --mode crag

# Skip out-of-corpus questions (faster for development)
python -m backend.eval.run_eval --mode crag \
  --skip-categories out_of_corpus

# Test a single question
python -m backend.eval.run_eval --mode crag \
  --questions-only q001
```

Terminal output preview:
```
══════════════════════════════════════════════════════════════════════
  DocuTrust — Evaluation Results
══════════════════════════════════════════════════════════════════════

  Metric                                  Naive RAG    CRAG
  ────────────────────────────────────────────────────────────────
  Faithfulness (in-corpus)                65.0%        88.0%
  Answer relevance (in-corpus)            72.0%        86.0%
  Citation rate (in-corpus)               0.0%         92.0%
  Correct abstention (OOC + adversarial)  5.0%         80.0%
  Mean latency / query                    2.10s        6.80s

══════════════════════════════════════════════════════════════════════
```

---

## 🎤 Interview Q&A

Built-in answers to every question this project will generate:

<details>
<summary><strong>What stops infinite correction loops?</strong></summary>

Hard cap at `MAX_CORRECTION_ITERATIONS=2` in `.env`.
`route_after_grading()` in `backend/graph/edges.py` checks
`iteration_count >= max_correction_iterations` before routing
to `rewrite_query`. Once the cap is hit, it routes to
`web_fallback` instead. If web fallback also returns nothing,
`corrective_generate` returns an explicit abstention message.
Three-layer safety net, one config variable.

</details>

<details>
<summary><strong>Why cross-encoder instead of asking the LLM "is this relevant?"</strong></summary>

Cross-encoders jointly attend over query + document through the same
attention layers — they see the relationship between them, not just
each independently. `BAAI/bge-reranker-v2-m3` is trained specifically
on MS-MARCO passage ranking. It outputs calibrated numerical scores in
`[0, 1]` after sigmoid — actual thresholds you can tune, not text
parsing of "yes/no" responses. Cheaper (local), faster (no API call),
deterministic (no temperature), and purpose-built for this exact task.

</details>

<details>
<summary><strong>How do you prevent web fallback from introducing unverified information?</strong></summary>

`corrective_generate` explicitly tags every claim with its source:
`[chunk_id]` for internal KB, `[WEB: url]` for web results. The UI
surfaces a "Web Fallback Activated" banner so users always know when
external sources were used. The documented next step (in "What I'd
Add") is a second cross-encoder grading pass on web results — applying
the same quality gate to external content as to KB chunks.

</details>

<details>
<summary><strong>How would this scale past local ChromaDB?</strong></summary>

The vector store is wrapped behind two functions in
`backend/ingestion/embedder.py`: `upsert_chunks()` and
`query_similar()`. Everything else in the codebase calls these
two functions — nothing imports ChromaDB directly outside that file.
Swapping to Qdrant means implementing those two functions against
the Qdrant Python client. The rest of the codebase is unchanged.
Qdrant supports distributed deployment, payload filtering, and
sparse+dense hybrid retrieval for better keyword recall.

</details>

<details>
<summary><strong>What's your chunking strategy and why?</strong></summary>

Recursive character splitting: tries `\n\n` → `\n` → `. ` → ` `
separator order. Splits on paragraph/sentence boundaries rather than
arbitrary character positions. 1500-char target size (~300-400 tokens
for English prose — within all encoder context windows). 200-char
overlap (~1-2 sentences) prevents retrieval misses for claims that
straddle chunk boundaries. Min 100-char filter removes noise (page
numbers, lone headers). Page numbers preserved on every chunk for
human-readable citations.

</details>

<details>
<summary><strong>How do citations actually trace back to the source?</strong></summary>

Chain: `answer [chunk_id]` → regex parse → MongoDB chunk record
→ `doc_name + page_number + text_preview`. The LLM is instructed
to use exact chunk IDs provided in the context string. The regex
`r'\[([a-zA-Z0-9_]+_chunk_\d+)\]'` extracts them. Unresolved
citations (LLM hallucinated a chunk_id) are flagged in the log
with `[Citation could not be resolved]`. The citation chain is
deterministic — not LLM-generated metadata.

</details>

---

## 🗺 Roadmap — What I'd Add With More Time

| Feature | Why it matters |
|---------|----------------|
| **Second-pass web grading** | Apply same cross-encoder quality gate to Tavily results — web content is currently trusted without re-grading |
| **BM25 hybrid retrieval** | Sparse keyword search combined with dense vectors improves recall for exact-match queries (model names, version numbers, code identifiers) |
| **Qdrant migration** | ChromaDB is local-only; Qdrant scales horizontally with the same interface abstraction already in place |
| **RAGAS CI gate** | Run eval on every PR with `faithfulness >= 0.85` threshold — the "eval as CI" pattern that separates production ML from hobby projects |
| **Streaming generation** | Stream LLM tokens via existing WebSocket instead of waiting for full response — dramatically improves perceived latency |
| **Multi-modal ingestion** | Tables (camelot), images (GPT-4V/LLaVA), PowerPoint (python-pptx) — the document types most missing from enterprise RAG |
| **Re-ranking cache** | Cache cross-encoder scores for (query, chunk_id) pairs — repeated queries with overlapping retrievals skip the reranking step |

---

## 📚 References

- [Corrective Retrieval Augmented Generation](https://arxiv.org/abs/2401.15884)
  — Yan et al., 2024. The CRAG pattern this system implements.
- [BGE M3-Embedding](https://arxiv.org/abs/2309.07597)
  — Xiao et al., 2023. The embedding model family used here.
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
  — State machine framework for the CRAG graph.
- [RAGAS](https://docs.ragas.io/)
  — Evaluation framework for RAG pipelines.

---

## 📄 License

MIT — see [LICENSE](LICENSE)

---

<div align="center">

**Built to demonstrate production RAG engineering, not tutorial following.**

*If you found this useful, ⭐ star the repo.*

</div>
