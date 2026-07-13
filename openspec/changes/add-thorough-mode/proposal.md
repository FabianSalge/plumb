# Thorough mode v1: per-claim retrieval into one scoring pass

Issue: #66 · ADR-0010 (thorough-mode retrieval), ADR-0002 (recall-then-rerank)

## Why

The one-liner promises verification against a tenant's knowledge base, but
`/v1/verify` still checks only caller-provided passages — the context-native
blind spot ADR-0003 recorded as fast mode's known cost. ADR-0010 is accepted
and defines exactly how thorough mode closes it; this change implements that
decision.

## What Changes

- New retrieval stage in the engine: each claim (the verbatim sentence of
  ADR-0009) becomes one query, expanded deterministically with neighboring
  answer sentences (window size in versioned config), run through ADR-0002's
  recall-then-rerank interface read-only against the tenant's store.
- First store adapter: **Postgres full-text search** — the pgvector
  deployment shape, where chunk text lives beside the vectors. Lexical recall
  via `tsvector`/`websearch_to_tsquery` needs nothing but a read-only role,
  it is the store most RAG stacks already run, and it is light enough that
  the kind-based e2e exercises the real adapter. Qdrant/OpenSearch adapters
  are later work against the same interface.
- Pooling: results deduplicated across claims and packed into the model
  window by quota — every claim's top chunk is guaranteed a slot, remaining
  space fills by global rerank score. Budget and quota in versioned config;
  truncation logged, never silent.
- One joint forward pass over the pooled evidence, identical to fast mode's
  scoring path; segment-after-score cuts per-claim scores. Mode changes what
  fills the context window, never the scoring semantics — no calibration
  refit is triggered.
- **BREAKING** (pre-1.0): `mode: "thorough"` accepted; `context` optional in
  that mode (when present it joins the pool with caller provenance); each
  claim gains `evidence` — retrieval provenance (source/chunk identity,
  retrieval rank, store snapshot identity where exposed), documented as
  "retrieved for this claim", not "supports this claim". Fast mode's
  contract does not move.
- Deployment config carries the tenant store connection, read-only,
  versioned like the rest of the chart values.
- Evals: thorough-mode p50/p95 measured end to end against ADR-0010's
  p95 ≤ 10 s target, and thorough-mode calibration error measured on
  retrieved-evidence traffic — both published in `evals/RESULTS.md`.

## Capabilities

### New Capabilities

- `evidence-retrieval`: per-claim query construction (deterministic context
  expansion), recall-then-rerank against a read-only store adapter, pooling
  with dedupe/quota/global fill, and the provenance each pooled chunk
  carries.

### Modified Capabilities

- `verify-api`: `mode: "thorough"` accepted (the "only fast mode" requirement
  is replaced); `context` becomes optional in thorough mode; claims carry
  `evidence` provenance; gate parity holds in both modes.
- `helm-deploy`: chart values gain the tenant store connection (read-only,
  versioned); readiness accounts for the store dependency in thorough-mode
  deployments.

## Impact

- `engine/`: new `retrieval/` package (query expansion, adapter interface,
  Postgres adapter, pooling); pipeline wiring for mode dispatch.
- `api/`: request/response schemas (`mode`, optional `context`, `evidence`),
  validation errors.
- `config/verifier.yaml`: new versioned `retrieval` section (expansion
  window, recall depth, rerank model + revision, pool budget, per-claim
  quota).
- `charts/`: store connection values, secret handling for credentials,
  readiness.
- `evals/`: latency + calibration measurement on retrieved-evidence traffic,
  `RESULTS.md` update.
- New dependencies: a Postgres driver (engine), a cross-encoder reranker
  model pinned by revision in config (weights not baked into the image, same
  policy as the scorer). Reranker model *selection* benchmarks stay out of
  scope per #58 — the pin is provisional and swappable by config bump.
- Out of scope: request-level multi-tenancy (deferred to its own ADR), NLI
  and self-consistency signals, reranker/embedding benchmarks.
