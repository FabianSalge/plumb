# Joint inference and span attribution

## Why

ADR-0007 is accepted: per-passage scoring costs one forward pass per passage, under-scores
claims grounded only in the union of passages, and deviates from the configuration
benchmarked in `evals/RESULTS.md` — and it becomes M×N once decomposition lands. The
calibrator (#34/#32) must be fitted to the scoring mode that will actually serve, so this
lands first (issue #39).

## What Changes

- The scorer moves to joint all-passages inference: one forward pass per claim, all
  passages rendered into the vendored multi-passage context format (verified against
  lettucedetect 0.2.1, the package version the pinned v2 model ships with:
  `passage N: <text>` lines joined by newline into the summary template). The rendered
  format is pinned by a golden regression test.
- The claim `score` changes meaning: support by the union of passages
  (1 − max token hallucination probability, reduction unchanged). Out-of-range still
  fails loudly.
- **BREAKING**: `evidence_index` leaves the response. Each claim instead carries
  `spans` — `start`/`end` (Unicode code-point offsets into the claim's `text`) and
  `text` — marking its unsupported regions, with no confidence field until #32.
- The span-flagging threshold moves from a module constant into versioned config;
  `config_version` bumps.
- Context-truncation logging now notes the total passage count.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `groundedness-scoring`: `score(claim, passages)` returns one union-support score from a
  single joint forward pass instead of per-passage scores; the vendored format requirement
  covers multi-passage rendering; spans move from log-only detail to scorer output; the
  span-flagging threshold comes from versioned config.
- `verify-api`: claims carry `spans` instead of `evidence_index`; verdict semantics
  (threshold from versioned config) and gate semantics unchanged.

## Impact

- `engine/scoring.py` (prompt rendering, scorer interface, span threshold), `engine/verdict.py`
  (no more best-index selection), `engine/config.py` + `config/verifier.yaml` (span threshold,
  version bump), `api/schemas.py` + `api/app.py` (response shape).
- API breaks once, pre-1.0, no deprecation window (ADR-0007).
- README/docs wherever the response shape is shown.
- Downstream: #32 fits the calibrator to this scoring mode; #36 measures its latency;
  span confidences stay out until #32.
