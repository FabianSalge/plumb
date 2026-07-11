# Design — calibration v0

## Context

ADR-0008 fixed the methodology: Platt scaling on the raw joint-inference support score,
fitted on the ~2,100 RAGTruth test-split responses outside the seed-18 benchmark slice,
validated in-domain on that slice and out-of-domain on an LLM-AggreFact slice. The
artifact is a portable versioned file carrying coefficients plus the bindings it was
fitted against, and the engine refuses to serve through a mismatched calibrator.

Since that ADR, sentence decomposition landed (ADR-0009, #45): the engine's claim unit
is now the verbatim sentence, and each claim's raw support is `1 − max` token risk over
the tokens overlapping it. ADR-0008 anticipated the unit change and named the labelling
mechanism — RAGTruth's span annotations — which `evals/bench/sentence.py` already uses
(a sentence is hallucinated iff its range overlaps an annotated span). The raw score's
pathologies (length drift, clumping near the ends of [0, 1]) are why calibration exists.

## Goals / Non-Goals

**Goals:**

- Fit one Platt calibrator at the shipping configuration: pinned model revision, joint
  question-less inference, sentence claim unit, max-risk-overlap reduction.
- Ship it as a versioned artifact the engine validates at startup and applies per claim.
- Report reliability honestly: in-domain and out-of-domain ECE with reliability-diagram
  data in `evals/RESULTS.md`.
- `/v1/verify` returns calibrated confidence; the verdict thresholds it.

**Non-Goals:**

- Span confidences in the response (#40 owns promoting those once calibrated).
- Per-tenant or per-domain refit; multiple concurrent artifacts.
- Any change to segmentation, reduction, or gate conjunction semantics.
- Revisiting the method — a bad sigmoid fit is recorded evidence for a superseding ADR,
  not a silent swap.

## Decisions

### 1. Fit at the sentence claim unit, labels by span overlap

The calibrator must be fitted to the unit it serves. Fit examples are the sentences of
the ~2,100 RAGTruth test responses **outside** the seed-18 slice (exclusion by response
id, computed by the same `stratified_slice` call the benchmark uses — same code path, no
re-derivation drift). Each sentence is scored exactly as `/v1/verify` scores it (one
whole-answer joint pass, `engine.decomposition` reduction) and labelled supported iff it
overlaps no annotated span — the #45 labelling convention, unchanged. Roughly 15k
sentences at ~9% hallucinated. Alternative — fit response-level as ADR-0008 originally
sketched — rejected: it binds the calibrator to a claim unit the engine no longer serves,
which the artifact's own binding check would have to refuse.

### 2. Calibrated confidence is P(supported), via a clamped-logit Platt map

`confidence = sigmoid(a · logit(s) + b)` where `s` is the raw support score clamped to
`[ε, 1−ε]`, ε = 1e−6, before the logit. The clamp is part of the artifact schema
semantics, not a tunable: raw supports can round to exactly 0.0 or 1.0 in float, the
logit must not blow up, and the output consequently never prints exact certainty —
which ADR-0008 requires anyway. The map is monotone, so AUROC, ranking, and the gate's
decision structure are untouched.

### 3. Fitting lives in evals, hand-rolled and unit-tested; the engine only applies

The engine's runtime job is two floats and a sigmoid — `engine/calibration/` gets no
fitting code and no new dependencies. The fit script lives in `evals/bench/` beside the
benchmarks, and the logistic MLE (two-parameter Newton–Raphson on log-loss) is
implemented directly, matching the evals convention that metrics are unit-testable and
free of silent library defaults. At ~15k points and two parameters, Platt's
target-smoothing refinement buys nothing; plain MLE, noted here so the omission is a
decision, not an oversight.

### 4. Artifact: one YAML file under `config/`, referenced from the verifier config

`config/calibration/<model-shortname>-<claim-unit>-v1.yaml`, referenced by path from
`signals.groundedness.calibration` in `config/verifier.yaml`. Contents:

- `schema`: artifact schema version (integer; the ε clamp and coefficient meaning are
  pinned to it)
- `method: platt`, `coefficients: {a, b}`
- `bindings`: `model`, `revision`, `inference_mode`, `claim_unit`
- `fit`: dataset identity, exclusion rule, fit-set SHA-256 (over the ordered
  `(response_id, sentence_start, sentence_end, label)` tuples), sentence count, date
- `metrics`: in-domain and out-of-domain ECE with slice identities

Portable and diff-reviewable; committed to the repo, baked into the image the same way
`verifier.yaml` is, and shipped by the chart. A refit is a new artifact file plus a
config-version bump — never an in-place coefficient edit.

### 5. Bindings validate against engine-declared protocol identifiers

Model and revision validate against the running config. Inference mode and claim unit
validate against constants the engine itself declares: `engine.scoring` names its
protocol (joint question-less all-passages, the vendored template) and
`engine.decomposition` names its unit (rule-based sentence segmentation +
max-risk-overlap reduction). Anyone changing those code paths must bump the constant —
the golden prompt/segmentation tests changing in the same diff is the reviewer's signal
— and a stale artifact then fails the startup check by construction. Alternative —
hashing the prompt template and segmenter source — rejected: false refits on comment
edits, and it can't cover behaviour that lives outside the hashed text.

### 6. Refusal is a startup failure, and calibration is mandatory

A missing artifact reference, an unreadable/invalid artifact, or any binding mismatch
fails startup with an error naming every mismatched field (expected vs found). No
serve-raw-scores fallback, no degraded mode: the product's claim is calibrated
confidence, so an uncalibrated engine is not ready. `/readyz` stays non-200 because the
app never finishes starting.

### 7. API: `confidence` replaces `score`; the threshold moves to confidence space

Claims carry `confidence` (calibrated P(supported)); the raw support score moves to
structured logs next to the span confidences that already live there. Keeping both
fields was considered and rejected: two numbers invite thresholding the raw one, which
is the exact failure mode calibration retires. The verdict threshold in
`config/verifier.yaml` is re-read as a threshold on calibrated confidence and re-picked
against the reliability data (0.5 now means "more likely supported than not"); the
config version bump makes the verdict shift visible per the verdict-pinning contract.
Monotonicity keeps gate parity with the whole-text decision intact — thresholding
confidence at t equals thresholding raw support at the inverse-mapped t.

### 8. Reliability metrics: ECE over 10 equal-width bins, diagram data published

`evals/bench/metrics.py` gains ECE and reliability-bin computation (10 equal-width bins;
per-bin mean confidence, empirical support rate, count), unit-tested like the existing
metrics. In-domain: the seed-18 slice's sentences. Out-of-domain: an LLM-AggreFact slice
with its RAGTruth subset excluded and the chosen subsets checked against the pinned
model card's training mix at fit time; the check's outcome is recorded in the artifact
and RESULTS.md. LLM-AggreFact is (document, claim, label) at its own claim granularity —
each claim is scored as a single unit and the mismatch with our sentence unit is stated
in RESULTS.md rather than papered over. Both ECEs plus diagram data land in RESULTS.md;
the OOD number will be worse, and publishing it is the point.

## Risks / Trade-offs

- **The sigmoid assumption fails visibly** → reliability diagrams are a required output;
  a systematically bad fit becomes evidence for a superseding ADR (isotonic/beta), per
  ADR-0008.
- **Protocol constants rely on humans bumping them** → the golden prompt and
  segmentation tests change in the same diff whenever behaviour changes; review checks
  the constant moved with them. Accepted over source-hashing's false positives.
- **LLM-AggreFact subsets may overlap the model's training mix** → subsets verified
  against the model card at fit time; anything unverifiable is excluded and the
  exclusion recorded.
- **Sentence labels inherit span-boundary noise** → same convention as the #45
  benchmark, so fit and evaluation are at least consistent; noise dilutes fit sharpness
  rather than biasing it directionally.
- **~2,100-response scoring run on a 16 GB M4 laptop** → ~10 minutes at measured
  latencies; run under caffeinate, results JSON committed so the fit is reproducible
  without re-scoring.
- **Verdicts shift at the flip of the config version** → intended and visible: the bump
  plus RESULTS.md give the before/after story; there are no production tenants pre-1.0.

## Migration Plan

One breaking API change (score → confidence), shipped with the config-version bump and
the artifact file in the same PR; chart configmap and image pick up `config/` as they do
today. Rollback is reverting the PR — the artifact and config travel together, so there
is no state to unwind.

## Open Questions

None — the methodology is decided (ADR-0008); the remaining unknowns (fit coefficients,
measured ECEs, the exact LLM-AggreFact subset list) are outputs of the work, not inputs.
