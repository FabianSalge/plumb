# ADR-0008: Calibration methodology — Platt scaling on held-out RAGTruth

2026-07-06 · Status: Accepted

## Context

ADR-0004 rests the product's accuracy claim on aggregation and
calibration, yet `/v1/verify` still returns the raw LettuceDetect support
score. That number is `1 − max` over per-token hallucination
probabilities — an extreme-value statistic, not a probability: it drifts
with claim length, clumps near the ends of [0, 1], and any threshold
picked against it dies with the next model swap in a slot ADR-0006
deliberately made swappable. ADR-0007 has settled the scoring mode the
calibrator must be fitted to — joint all-passages inference (#39) — so
the methodology can now be fixed ahead of its implementation (#32).

The fitting data in hand is the RAGTruth test split: 2,700 responses
with human response-level support labels, which match tier-1's
whole-response claim unit exactly. The pinned model's card confirms the
v2 line trains on KR Labs' unified prose+code sets and holds RAGTruth
test out as its own benchmark, so fitting on it is legitimate — but it
remains in-domain for the model family, and whatever the calibrated
number claims must say so. The stratified 600-example slice of that
split is already the discrimination benchmark in `evals/RESULTS.md`; the
remaining ~2,100 examples are untouched. Methods considered:

- **Temperature scaling** — one parameter, but built for logits; the
  raw score is already a bounded probability-shaped max, and a fixed
  slope buys nothing Platt doesn't.
- **Isotonic regression** — nonparametric, and ~2,100 examples would
  feed it, but it produces a step function, overfits at the extremes,
  and emits exact 0.0 and 1.0 — a certainty this product must never
  print. The artifact becomes a lookup table instead of two numbers.
- **Beta calibration** — native to [0, 1] scores and strictly more
  flexible than Platt at three parameters, but the flexibility is
  marginal and the method is harder to explain in user-facing docs.
- **Platt scaling** — a two-parameter logistic map on the log-odds of
  the raw score. Monotone, so AUROC and ranking are untouched; nearly
  impossible to overfit at this data size; the artifact is two floats.
  Its one real assumption — sigmoid-shaped miscalibration — is checkable
  by eye on a reliability diagram. Chosen.
- **Defer the choice to #32** — letting held-out ECE pick the winner at
  implementation time leaves trust-critical methodology implicit in
  whichever run happened last. Rejected; that is what this ADR is for.

## Decision

The engine calibrates the joint-inference support score with Platt
scaling: fitted on the ~2,100 RAGTruth test-split examples outside the
seed-18 benchmark slice, validated in-domain on that slice and
out-of-domain on an LLM-AggreFact slice (its RAGTruth subset excluded,
and the chosen subsets verified absent from the pinned model's training
mix at fit time), with ECE and reliability diagrams for both landing in
`evals/RESULTS.md`. Nothing the calibrator is fitted on is ever used to
evaluate it. The calibrated confidence claims exactly this, docs-plain:
*among claims the engine scores c, about a fraction c are fully
supported by the supplied passages, as measured on RAGTruth-style RAG
traffic; out-of-domain calibration error is published alongside.* The
artifact is a portable versioned file — coefficients plus the bindings
it was fitted against: model id and revision, inference mode, claim
unit, fit-set identity and hash, and its measured metrics. A refit is
forced by anything that moves the raw score distribution: a model or
revision change, an inference-mode or prompt-template change, a change
to the token-to-claim aggregation, or a claim-unit change when
decomposition (#35) lands. The engine refuses to serve a score through
a calibrator whose bindings mismatch the running config.

## Consequences

- The gate thresholds a probability instead of a model-specific raw
  score, which is the difference between "0.5 worked on our benchmark"
  and a number a tenant can set policy against.
- The sigmoid form is a bet made visible: reliability diagrams are a
  required eval output, and a systematically bad fit is recorded
  evidence for a superseding ADR (isotonic or beta), not a silent swap
  inside `engine/calibration/`.
- The out-of-domain ECE will be worse than the in-domain number, and
  publishing both is the point — it is the honest answer to "what
  happens on my domain" and the standing case for per-tenant refit work
  later.
- Swapping the groundedness slot is no longer only a config-and-adapter
  change plus a benchmark run (ADR-0004): it now also requires fitting
  and shipping a calibration artifact for the new occupant. The refusal
  behaviour turns a forgotten refit from a silent miscalibration into a
  loud failure, per #32's acceptance criteria as written.
- Fitting waits on #39 — calibrating per-passage scores that ADR-0007
  retires would be wasted work. #32 inherits that sequencing.
- Decomposition (#35) breaks the claim unit the calibrator is bound to
  and will need claim-level labels; RAGTruth's span annotations can
  derive them, but that fit is new work the #35/#32 follow-ups own.
- All of it is calibrated to one distribution, honestly labeled. The
  claim the product makes is calibration *transparency*, not domain
  invariance.
