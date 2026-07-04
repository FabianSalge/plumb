## 1. Contract tests first (must fail before implementation)

- [ ] 1.1 Contract tests for `POST /v1/verify`: supported, unsupported, invalid request (400), unknown mode (400), version fields present, gate semantics
- [ ] 1.2 Unit tests for the scoring wrapper: score range, max-over-passages, evidence_index selection
- [ ] 1.3 Unit tests for config loading: threshold read from file, config_version echoed, missing/invalid config fails loudly

## 2. Engine

- [ ] 2.1 Scoring wrapper around HHEM-2.1-open: (claim, passages) → per-passage scores
- [ ] 2.2 Versioned threshold config file + loader (no hardcoded constants)
- [ ] 2.3 Verdict mapping and conjunctive gate decision

## 3. API surface

- [ ] 3.1 `POST /v1/verify` endpoint wired to the engine, request validation per spec
- [ ] 3.2 `/healthz` and `/readyz` (readiness gated on model load)
- [ ] 3.3 Structured JSON logging middleware with X-Request-ID propagation

## 4. Close out

- [ ] 4.1 Coverage floor configured in CI for `engine/` (agree number in the PR, e.g. 80%)
- [ ] 4.2 All acceptance criteria on issue #8 checked; OpenSpec change validated and archived
