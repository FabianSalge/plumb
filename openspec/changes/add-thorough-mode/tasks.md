# Tasks: add-thorough-mode

Checks come first throughout: each group starts with failing tests.

## 1. Retrieval config

- [x] 1.1 Failing tests: `retrieval` section parsed from `config/verifier.yaml` (expansion_window, recall_depth, per_claim_quota, pool_budget_tokens, reranker model + revision), fail-loud on partial section, section optional for fast-only configs
- [x] 1.2 Extend `engine/config.py` with `RetrievalConfig`; add the section to `config/verifier.yaml` with a version bump

## 2. Query expansion

- [x] 2.1 Failing golden tests pinning exact query strings: leading sentence + preceding window + claim, dedupe preserving order, single-space join, window 0/1/N, leading-claim case, duplicate-sentence sharing
- [x] 2.2 Implement `engine/retrieval/expansion.py` over the existing segmentation output

## 3. Store adapter

- [x] 3.1 Failing tests: `EvidenceStore` protocol and `Chunk` (text, source identity, chunk identity, optional snapshot identity — never invented); fake-store unit coverage
- [x] 3.2 Failing tests (marked, real Postgres): FTS recall via `websearch_to_tsquery`/`ts_rank_cd`, configurable table/columns/regconfig, read-only session, loud errors on unreachable store and schema mismatch
- [x] 3.3 Implement `engine/retrieval/store.py` and `engine/retrieval/postgres.py`; add the Postgres driver dependency; startup probe validating connection and schema

## 4. Reranker

- [x] 4.1 Failing tests: `Reranker` protocol, fake-reranker unit coverage; model-marked test loading the pinned bge-reranker-v2-m3 revision and scoring one batch
- [x] 4.2 Implement `engine/retrieval/rerank.py` mirroring the scorer's load pattern (model extra, pinned revision, no baked weights); one batched pass per request, one rerank per unique query

## 5. Pooling

- [x] 5.1 Failing tests: dedupe by chunk identity; caller passages first and never displaced; per-claim quota guarantees the minority claim's slot; global fill by rerank score; budget counted in scoring-tokenizer tokens reserving prompt + answer; every drop logged with identity and reason
- [x] 5.2 Implement `engine/retrieval/pool.py`

## 6. API surface

- [x] 6.1 Failing tests: `mode: "thorough"` accepted; unknown modes 400 naming both supported modes; `context` optional in thorough / required in fast; thorough on a store-less deployment 400 fast-only; store failure mid-request 502, no partial-evidence verdict
- [x] 6.2 Failing tests: per-claim `evidence` (source/chunk identity, retrieval rank, snapshot identity only where exposed); caller passages absent from `evidence`; fast-mode responses unchanged byte-for-byte
- [x] 6.3 Failing property test: gate parity in thorough mode over a fixed pool; scorer-truncation-in-thorough logged as error
- [x] 6.4 Implement schema changes and mode dispatch in `api/schemas.py` / `api/app.py`; thorough path pools then calls the unchanged scoring/decomposition/calibration sequence

## 7. Chart and deployment

- [x] 7.1 Failing chart tests: store values (enabled flag off by default, DSN via Secret reference only, table/columns/regconfig); NetworkPolicy egress to the store endpoint only when enabled; fast-only default renders no store config
- [x] 7.2 Implement chart values, templates, and startup wiring; document values in the README (the chart has no README of its own — chart docs live in the main README's Helm section)
- [x] 7.3 Extend kind e2e: Postgres pod with seeded chunks, golden thorough request through the Service asserting verdicts and `evidence` provenance (`make deploy-thorough` + `make e2e-thorough`)

## 8. Evals

- [ ] 8.1 Measure thorough-mode p50/p95 end to end on stated hardware against the p95 ≤ 10 s target; publish in `evals/RESULTS.md` with knob settings
- [ ] 8.2 Measure thorough-mode calibration error (ECE) on retrieved-evidence traffic alongside the provided-context number; publish in `evals/RESULTS.md`

## 9. Docs and closeout

- [ ] 9.1 Update README and docs: thorough mode contract, `evidence` documented as retrieval provenance not support attribution, store onboarding (read-only role), fast mode unmoved
- [ ] 9.2 Full local gate: `make test`, `make lint`, `make typecheck`, chart lint; OpenSpec validate; PR describes how each acceptance criterion was verified
