# Calibrated span confidences (issue #40)

## Why

ADR-0007 shipped spans with positions only: the per-span numbers the model produces are
raw max-token probabilities, and the response must not carry a value people will read as
a probability. Calibration v0 (#32, ADR-0008) has landed the machinery — a versioned
Platt artifact bound to the running model, revision, inference mode, and claim unit — so
span confidences can now follow, provided the calibration applied to span-level scores
is shown to be valid at span level, not assumed. Today that detail lives only in
structured logs; a `block` verdict points at a region without saying how sure the engine
is about it.

## What Changes

- Each span in the `/v1/verify` response gains a `confidence` field: the calibrated
  probability that the flagged region is genuinely unsupported by the supplied passages.
  Additive to the response shape — span geometry and the flagging threshold are
  untouched (fixed by ADR-0007).
- The evals harness gains a span-level reliability measurement on RAGTruth's span
  annotations: first checking whether the claim-level calibrator transfers to span
  scores, and fitting a span-level Platt map per the ADR-0008 methodology if it does
  not. Either outcome lands as reliability data in `evals/RESULTS.md`.
- The calibration artifact schema is extended to carry the span-level calibration
  (coefficients, fit identity, metrics) alongside the claim-level one, under the same
  bindings; the binding validation from #32 (model revision + inference mode, fail
  loudly on mismatch) covers the span path by construction. The span path additionally
  binds the span-flagging threshold it was fitted at, since the flagged-span population
  is defined by that knob.
- A new artifact file ships with a verifier config-version bump, per the
  no-in-place-edit rule.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `verify-api`: spans gain a calibrated `confidence` field; "spans carry no confidence
  field" retires.
- `calibration`: span-level calibrated confidence requirement; artifact carries the span
  calibration under the same bindings plus the span-flagging threshold; refusal
  behaviour extends to the span path.
- `groundedness-scoring`: the "confidences stay in the logs" constraint on spans is
  replaced — spans carry the calibrated confidence toward the API, raw token risks stay
  in structured logs.

## Impact

- `engine/calibration/` (protected zone — this proposal is the plan): artifact schema
  version bump, span-map loading and validation.
- `engine/decomposition/reduction.py`: `Span.confidence` currently holds the raw max
  token risk; the raw value must stay distinguishable from the calibrated one on the way
  to the API.
- `api/schemas.py`, `api/app.py`: span response shape and calibration application.
- `config/`: new calibration artifact file + config version bump.
- `evals/bench/`: span-label derivation from RAGTruth annotations, transfer check,
  optional span-level fit, reliability output; `evals/RESULTS.md` gains the span
  section.
- Docs: README/docs statements that span confidences are log-only get updated in the
  same PR.
