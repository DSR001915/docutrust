# DocuTrust Evaluation Results: Naive RAG vs CRAG

Evaluated on 25 hand-written questions against `sample_data/corporate_policy_handbook.pdf` (20 in-corpus, 5 deliberately out-of-corpus).

**Important caveat:** this run used the project's mock providers (`DOCUTRUST_PROVIDER_MODE=mock`, `LLM_PROVIDER=mock`) for deterministic, offline scoring -- see the notes below for what changes with real embedding/reranker/LLM models.

## Summary

| Metric | Naive RAG | CRAG |
|---|---|---|
| Faithfulness (in-corpus) | 95.0% | 95.0% |
| Correctly handled (out-of-corpus) | 0.0% | 100.0% |
| Avg. latency per query (sec) | 0.005 | 0.0052 |

CRAG's correction loop (rewrite and/or web fallback) triggered on **8/25** queries.

### What "correctly handled" means here

For naive RAG, correctly handling an out-of-corpus question means explicitly saying it doesn't know -- naive RAG has no fallback mechanism, so abstention is its only correct option.

For CRAG, correctly handling an out-of-corpus question means EITHER explicitly abstaining OR escalating to a disclosed web-search fallback rather than silently presenting irrelevant internal document chunks as if they answered the question. Measuring CRAG by literal abstention text alone understates what it actually does: when internal retrieval is graded `incorrect`, CRAG is *designed* to find an alternative source and disclose that switch, not to give up. In mock mode the web search provider is a labeled placeholder (no live network call), so this eval is verifying the control flow takes the disclosed-fallback path correctly, not the quality of the web content itself -- see `backend/graph/web_search.py`.

## Reading these numbers honestly

- **Faithfulness** here is a substring-match proxy, not a full groundedness judge -- it catches missed/wrong facts but not subtler hallucination. A production eval should layer in RAGAS or an LLM-as-judge pass for nuance.
- **Naive RAG has no relevance signal at all.** An earlier version of this baseline synthesized a confidence score from raw vector distance so it could use the same abstention check as CRAG. That was removed: empirically, this project's mock embedder gives even semantically unrelated text pairs a baseline cosine similarity of roughly 0.05-0.2 (verified directly, not assumed), so a distance-derived "confidence" score doesn't reliably separate relevant from irrelevant retrieval without a reranking step. That lack of a trustworthy relevance signal is naive RAG's real architectural gap, not an artifact of this eval -- which is exactly what motivates adding a cross-encoder reranker + grading step in CRAG rather than trying to threshold raw vector distance.
- **Latency is a real cost.** CRAG's number includes the queries that took the rewrite-loop and/or web-fallback path. In a production system, this is the tradeoff you're buying: better abstention/fallback handling, at the cost of extra round-trips on the queries that need it. CRAG's conditional routing means confident retrievals skip the loop entirely and only pay this cost when correction is actually triggered.

## Per-question results

For in-corpus questions, the relevant columns show **faithful** (did the answer contain the expected fact). For out-of-corpus questions, the columns instead show **handled** (did the system correctly decline or disclose a fallback, rather than presenting irrelevant content as a confident answer) -- faithfulness is not a meaningful check on those rows.

| ID | Question | In corpus | Naive | CRAG | CRAG used correction |
|---|---|---|---|---|---|
| q01 | How long must customer transaction records be retained? | ✓ | ✓ faithful | ✓ faithful | — |
| q02 | How long are employee records kept after termination? | ✓ | ✓ faithful | ✓ faithful | ✓ |
| q03 | What method must be used to destroy physical documents after | ✓ | ✗ unfaithful | ✗ unfaithful | — |
| q04 | What standard must be used to wipe digital records? | ✓ | ✓ faithful | ✓ faithful | ✓ |
| q05 | Who must approve a request for expanded data access? | ✓ | ✓ faithful | ✓ faithful | — |
| q06 | Is single-factor authentication allowed for Tier 1 systems? | ✓ | ✓ faithful | ✓ faithful | — |
| q07 | How often are access reviews conducted? | ✓ | ✓ faithful | ✓ faithful | — |
| q08 | After how many days of inactivity is account access automati | ✓ | ✓ faithful | ✓ faithful | — |
| q09 | What is the maximum duration for a privileged administrator  | ✓ | ✓ faithful | ✓ faithful | ✓ |
| q10 | Within how many hours must a suspected security incident be  | ✓ | ✓ faithful | ✓ faithful | — |
| q11 | How quickly must the SOC classify incident severity? | ✓ | ✓ faithful | ✓ faithful | — |
| q12 | If an incident involves unauthorized PII access, when must L | ✓ | ✓ faithful | ✓ faithful | — |
| q13 | How long after incident closure must the post-incident revie | ✓ | ✓ faithful | ✓ faithful | — |
| q14 | What security software must be installed on company laptops  | ✓ | ✓ faithful | ✓ faithful | — |
| q15 | Can remote employees use personal devices to access company  | ✓ | ✓ faithful | ✓ faithful | — |
| q16 | Is split-tunneling permitted on the company VPN? | ✓ | ✓ faithful | ✓ faithful | — |
| q17 | Within how many days must business travel expenses be submit | ✓ | ✓ faithful | ✓ faithful | — |
| q18 | What is the daily meal reimbursement cap for international t | ✓ | ✓ faithful | ✓ faithful | — |
| q19 | What receipt documentation is required for expenses over $25 | ✓ | ✓ faithful | ✓ faithful | — |
| q20 | For flights under six hours, what airfare class must be book | ✓ | ✓ faithful | ✓ faithful | — |
| q21 | What is the company's policy on pet adoption leave? | ✗ | ✗ not handled | ✓ handled | ✓ |
| q22 | What is the boiling point of liquid nitrogen? | ✗ | ✗ not handled | ✓ handled | ✓ |
| q23 | What is the company's maternity leave policy? | ✗ | ✗ not handled | ✓ handled | ✓ |
| q24 | Recommend a good recipe for chocolate chip cookies | ✗ | ✗ not handled | ✓ handled | ✓ |
| q25 | What stock trading restrictions apply to employees during bl | ✗ | ✗ not handled | ✓ handled | ✓ |

## What this eval does NOT cover

- Real cross-encoder relevance judgments (mock mode uses a lexical-overlap approximation -- see `backend/reranker/cross_encoder.py`)
- Real LLM-generated answers (mock mode uses extractive sentence-selection -- see `backend/llm/provider.py`)
- Multi-document corpora or adversarial/ambiguous phrasing beyond what's in the golden dataset
- Citation-level precision (whether each individual citation marker points to the single best-supporting chunk, vs. just *a* relevant chunk)

Re-running this script with `DOCUTRUST_PROVIDER_MODE=local LLM_PROVIDER=anthropic` on a machine with HuggingFace + Anthropic API access will produce results reflecting the real model stack.