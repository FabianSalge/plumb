# Calibration v0: from raw signal score to calibrated confidence

GitHub issue: #32 · ADR-0008 (methodology), ADR-0007 (scoring mode), ADR-0009 (claim unit)

## Why

The one-liner promises one *calibrated* verifier; today `/v1/verify` returns the raw
LettuceDetect support score. That number is `1 − max` over per-token hallucination
probabilities — an extreme-value statistic, not a probability: thresholds picked against
it don't survive a domain shift or a model swap in a slot ADR-0006 deliberately made
swappable. ADR-0008 fixed the methodology (Platt scaling, fitted on held-out RAGTruth)
and both of its sequencing dependencies have landed: joint inference (ADR-0007, #44) and
sentence decomposition (ADR-0009, #45). This implements it.

One thing has moved since ADR-0008 was written: the claim unit is now the sentence, not
the whole response. ADR-0008 anticipated exactly this — RAGTruth's span annotations
derive sentence-level labels, and the sentence-overlap labelling already exists in the
evals harness (#45). The calibrator is therefore fitted at the sentence claim unit the
engine actually serves, and that binding is recorded in the artifact.

## What Changes

- **`engine/calibration/`** — new module applying a Platt map (two floats) to the raw
  per-claim support score, loaded from a portable versioned artifact file that records
  the bindings it was fitted against: model id and revision, inference mode, claim unit,
  fit-set identity and hash, and measured metrics. On startup the engine validates the
  artifact's bindings against the running config and the engine's own scoring protocol;
  a mismatch is a loud startup failure, never a silent fallback to raw scores.
- **Fitting and evaluation in `evals/`** — a fit script over the ~2,100 RAGTruth
  test-split responses outside the seed-18 benchmark slice (sentence-level labels by
  span overlap), validated in-domain on the seed-18 slice and out-of-domain on an
  LLM-AggreFact slice (RAGTruth subset excluded; chosen subsets verified absent from the
  pinned model's training mix). ECE and reliability-diagram data land in
  `evals/RESULTS.md` alongside AUROC. Nothing the calibrator is fitted on evaluates it.
- **BREAKING** — `/v1/verify` claims carry `confidence` (the calibrated probability of
  support) instead of the raw `score`; the verdict thresholds the calibrated confidence,
  with the threshold in versioned config. Raw scores move to structured logs. Pre-1.0,
  one break, no deprecation window — same policy as ADR-0007.
- **`config/verifier.yaml`** — the groundedness signal gains a calibration artifact
  reference; the verdict threshold is re-read as a threshold on calibrated confidence.
  Config version bumps.

## Capabilities

### New Capabilities

- `calibration`: the calibration artifact contract — what it records, how the engine
  binds it to the running model/mode/claim-unit, the refusal behaviour on mismatch, and
  how the calibrated confidence is computed from the raw support score.

### Modified Capabilities

- `verify-api`: claims return calibrated `confidence` in place of raw `score`; the
  verdict is derived by thresholding the calibrated confidence; config identifies the
  calibration artifact alongside the model and thresholds.

## Impact

- `engine/calibration/` (new, protected zone — trust-critical math), `engine/verdict.py`
  (thresholds calibrated confidence), `api/schemas.py` / `api/app.py` (response shape),
  `engine/config.py` + `config/verifier.yaml` (artifact reference, version bump).
- `evals/bench/` — ECE/reliability metrics, fit script, LLM-AggreFact loader;
  `evals/RESULTS.md` gains a calibration section.
- Docs: README status line ("not yet calibrated") comes out via #33 after this lands;
  the confidence's meaning is documented docs-plain per ADR-0008.
- Out of scope: span confidences in the response (#40), retrieval and decomposition
  changes, and the methodology decision itself (ADR-0008, decided).
