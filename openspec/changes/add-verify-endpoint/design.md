# Design: add-verify-endpoint

## Context

First feature after repo bootstrap. Nothing exists yet: no API framework, no engine code. The language decision (ADR-0001, issue #7) lands before implementation starts, informed by two spikes (#5, #6). CI (#4) must be green first so the tests written here gate the PR.

## Goals / Non-Goals

**Goals:**
- The smallest honest slice: one claim, inline evidence, one signal, a verdict with a score and a citation index.
- Prove the delivery discipline end-to-end: spec → failing tests → implementation → green CI.
- Seed the contracts that later work extends (verdict vocabulary, version stamping, gate semantics).

**Non-Goals:**
- Claim decomposition, retrieval, reranking, multi-signal aggregation, calibration, tenancy, thorough mode, `contradicted` verdicts.

## Decisions

- **Single signal, HHEM-2.1-open**: a small cross-encoder that runs on laptop CPU; scored per (claim, passage) pair, max over passages wins. Alternatives (NLI model, LLM judge) are deferred to the multi-signal work in phase 2 — orchestrating them is the product, but not yet.
- **Threshold in a versioned config file**, injected at startup; the file carries its own `version` field which is echoed as `config_version`. Alternative (env var) rejected: not versionable, not auditable.
- **Gate rule is conjunctive**: all claims supported → `pass`, else `block`. `flag` is deferred until per-tenant policy exists.
- **Reject unknown modes** rather than falling back to fast: silent degradation would lie about what was verified.
- **API framework**: decided by ADR-0001; the spec and tests are written against the HTTP contract so they survive the decision either way.

## Risks / Trade-offs

- [Model load time makes cold starts slow] → `/readyz` gates traffic until loaded; probe configuration in the chart (#10) uses it.
- [CPU latency per check is unknown] → the spikes (#5, #6) measure it before this is implemented; numbers go in the spike notes.
- [Single-signal verdicts are weak] → acceptable and documented: this is the tracer bullet, not the verifier.

## Open Questions

- ADR-0001 (issue #7): API language — closes before implementation.
