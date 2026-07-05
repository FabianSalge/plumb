# Proposal: add-verify-endpoint

## Why

Plumb has no verification capability yet. The tracer bullet (issue #8) proves the whole delivery path — spec, tests-first, API, container, chart — around the smallest honest slice of the product: one claim checked against caller-provided evidence.

## What Changes

- New `POST /v1/verify` endpoint: Tier-1 (inline-context) verification of a single claim against caller-provided evidence passages, fast mode only.
- Single grounding signal: HHEM-2.1-open cross-encoder score, mapped to a verdict via a threshold from a versioned config file. The config names the signal model (name + pinned revision hash) and defines the threshold per-model, so the model swap decided in #18 is a config-version bump, not a silent verdict change.
- Verdicts limited to `supported` / `unsupported` — `contradicted` is deliberately absent until the NLI signal lands.
- Every response carries `engine_version` and `config_version` (verdict-pinning seed).
- `/healthz` and `/readyz` endpoints; structured JSON logging with request IDs.
- Out of scope: claim decomposition, retrieval, tenancy, calibration, thorough mode.

## Capabilities

### New Capabilities

- `verify-api`: the HTTP contract of `/v1/verify` — request/response shapes, verdict semantics, gate decision, version stamping, health probes.

### Modified Capabilities

<!-- none — this is the first spec -->

## Impact

- `engine/`: scoring wrapper around HHEM, threshold config loading.
- `api/`: HTTP surface, request validation, logging middleware.
- New runtime dependency on the HHEM-2.1-open model weights.
- Blocked by issues #4 (CI) and #7 (language decision); implemented in issue #8.
