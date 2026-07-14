# Design: thorough mode v1

## Context

ADR-0010 fixed the shape: per-claim queries with deterministic context
expansion, recall-then-rerank per ADR-0002, pooled evidence, one joint
scoring pass identical to fast mode. This document decides the code-level
questions the ADR left to implementation: module boundaries, the Postgres
adapter's contract, how the pool is budgeted, how caller context and the
store connection are handled, and where each knob lives.

Current state: `api/app.py` scores `request.context` directly
(`scorer.score(text, passages)`), then `decompose()` cuts per-claim scores.
Segmentation is rule-based on `text` alone, so claims are available before
scoring — retrieval slots in front of the existing scoring call without
touching it.

## Goals / Non-Goals

**Goals:**

- Retrieval as a separate engine package behind a recall-then-rerank
  interface; the Postgres adapter is one implementation, not the interface.
- Scoring semantics byte-identical to fast mode: same prompt rendering, same
  joint pass, same reduction. Mode changes only what fills the passage list.
- Every knob in versioned config; every truncation logged; every failure
  loud.

**Non-Goals:**

- Qdrant/OpenSearch adapters, the opt-in dense index (ADR-0002 option 3),
  ingest or chunking — we read the tenant's existing chunks as they are.
- Request-level multi-tenancy, NLI/self-consistency signals, reranker or
  embedding benchmarks (#58).
- Cross-request evidence caching. The ADR allows a cache inside the
  deployment boundary; v1 dedupes identical queries within a request only.

## Decisions

### Module layout

`engine/retrieval/` with four seams, mirroring the signal package's
protocol-plus-implementation pattern:

- `expansion.py` — claim → query. Pure function over the segmentation
  output.
- `store.py` — `EvidenceStore` protocol: `recall(query, k) -> list[Chunk]`,
  read-only by construction (no write methods exist to misuse). `Chunk`
  carries text, source identity, chunk identity, store snapshot identity
  (optional — only where the store exposes one).
- `postgres.py` — the first adapter.
- `rerank.py` — `Reranker` protocol: `score(query, chunks) -> list[float]`;
  transformers-backed cross-encoder implementation loaded like
  `LettuceDetectScorer.load` (model extra required, pinned revision, no
  baked weights).
- `pool.py` — dedupe, quota packing, global fill, truncation logging.

`api/app.py` dispatches on mode: fast keeps its exact current path; thorough
builds the pooled passage list, then calls the same
`scorer.score`/`decompose`/calibrate/judge sequence.

### Query expansion (deterministic, config-pinned)

Query for claim *i* = leading sentence of the answer + up to
`expansion_window` sentences immediately preceding claim *i* + claim *i*
itself, deduplicated preserving order, joined with single spaces. No model,
no rewriting; queries are ephemeral and never appear in the response. Golden
tests pin exact query strings for fixed inputs and window sizes, so a
window-size or joining change is a visible diff plus a config version bump.

### Postgres adapter

Targets the pgvector deployment shape: chunk text in a table beside the
vectors. Recall is Postgres full-text search:

- rank: `ts_rank_cd` over `to_tsvector(<regconfig>, <text_column>)` matched
  against `websearch_to_tsquery(<regconfig>, query)`, `LIMIT recall_depth`.
- terms are OR-joined before `websearch_to_tsquery` (found during
  implementation: websearch semantics AND all terms, which would demand every
  content word of an expanded multi-sentence query in one chunk — the
  opposite of recall). Bag-of-words OR recall with `ts_rank_cd` density
  ranking; the reranker restores precision, per ADR-0002's division of labor.
- The deployment config names table, id column, text column, optional source
  column, optional snapshot column, and the FTS regconfig (default
  `simple` — language analyzers are a tenant choice, not a guess).
- Read-only is enforced by the tenant granting a read-only role; the adapter
  additionally sets the session `default_transaction_read_only`, so a
  misconfigured role still cannot write.
- Chunk identity = table + id column value; source identity = source column
  value (or the table name when unconfigured); snapshot identity = snapshot
  column value when configured, otherwise absent — never invented.
- Connection, query, or schema errors surface as 502 with context. No empty
  result fallback: a store that errors is a failed verification, not a
  degraded one.

Alternative considered: requiring a tenant-side search endpoint (OpenSearch
first). Better BM25, but a heavier first dependency, a much heavier e2e
(kind cluster on developer hardware), and pgvector-next-to-app-data is the
more common stack we can read without any tenant-side work.

### Reranker

`BAAI/bge-reranker-v2-m3`, pinned by revision in the versioned config —
still the standard open multilingual cross-encoder default, and multilingual
matches the mmBERT scorer. The pin is provisional: #58's benchmark treatment
owns selection, and a swap is a config-version bump against the `Reranker`
protocol. All (query, chunk) pairs across claims are reranked in one batch;
rerank happens once per unique query (duplicate sentences share a query and
its results).

### Pooling and the window budget

1. Dedupe candidates across claims by chunk identity.
2. Caller `context` passages (thorough mode, when present) enter the pool
   first, labeled `source: "caller"` — the caller asked for them explicitly,
   so they are never displaced by retrieved chunks.
3. Every claim's top-reranked chunk gets a guaranteed slot
   (`per_claim_quota`, default 1).
4. Remaining budget fills by global rerank score.

Budget (`pool_budget_tokens`) is counted in scoring-tokenizer tokens of the
rendered passage lines, reserving room for the prompt template and the
answer, so pooling — not the scorer's silent window — decides what is
scored. Anything dropped at any step is logged with chunk identity and
reason; `truncated=True` from the scorer in thorough mode indicates a
budgeting bug, not normal operation.

Known trade-off, accepted in the ADR: rerank scores from different queries
are not strictly comparable, so global fill is an approximation; the
per-claim guaranteed slot is what protects minority claims.

### Evidence provenance in the response

Per the ADR, `evidence` on each claim lists the chunks *its query retrieved*
that made the scoring window: source identity, chunk identity, retrieval
rank (the claim-local rerank rank), and snapshot identity where present.
Caller passages made the window but were retrieved by no query, so they
appear in no claim's `evidence` — the caller already holds them. Documented
as retrieval provenance, not support attribution: joint inference cannot
name a supporting passage (ADR-0007), and the docs say so plainly.

### Configuration split

- `config/verifier.yaml` (versioned behavior): new `retrieval` section —
  `expansion_window`, `recall_depth`, `per_claim_quota`,
  `pool_budget_tokens`, reranker `model` + `revision`. Loaded through
  `engine/config.py` with the same fail-loud validation; section optional
  (fast-only deployments stay valid), but thorough requests against a
  config without it are a 400.
- Deployment config (chart values → env): store DSN (from a Kubernetes
  secret), table/column names, regconfig. Deployment identity, not engine
  behavior — it does not bump `config_version`.

### Store lifecycle and readiness

When a store is configured, startup validates the connection and the
configured table/columns (one probe query) and fails loud on mismatch —
same posture as the calibration artifact. `readyz` gains nothing
per-request: a mid-flight store outage fails the requests that need it with
502 rather than flapping the pod. Thorough requests on a deployment with no
store configured are 400, stating the deployment is fast-only.

## Risks / Trade-offs

- [FTS recall bounded by lexical match] → ADR-0002's known cost; the evals
  measure it, and the recorded fallback for insufficient expansion recall is
  per-answer-retrieve-then-attribute via a superseding ADR, not a
  coreference model.
- [CPU rerank latency vs p95 ≤ 10 s] → recall_depth × claims pairs per
  request, single batched pass; knobs are config, measured end to end in
  the evals before the number is published. If CPU misses, the same
  measurement prices the GPU recommendation.
- [Calibrator fitted on provided-context traffic] → no refit is triggered
  (inference mode unchanged), but thorough-mode ECE is measured on
  retrieved-evidence traffic and published — never assumed equal.
- [Global fill compares scores across queries] → accepted approximation;
  per-claim quota guarantees representation; truncation logged.
- [Tenant schema diversity] → explicit table/column config rather than
  discovery; startup probe fails loud on mismatch.

## Migration Plan

Pre-1.0 breaking change, shipped at once: thorough accepted, `context`
optional there, `evidence` added. Fast mode's contract does not move — every
existing fast-mode test passes unmodified. Chart values are additive and
optional; existing fast-only deployments upgrade with no value changes.
Rollback is redeploying the previous image/chart; no data migrations exist.

## Open Questions

- None blocking. The reranker pin is provisional by construction (#58 owns
  selection).
