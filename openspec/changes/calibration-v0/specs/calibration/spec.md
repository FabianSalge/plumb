# calibration

## Purpose

The engine-side contract of score calibration: how a raw per-claim support score
becomes a calibrated confidence, what the versioned calibration artifact records, how
the engine binds it to the running model, inference mode, and claim unit, and the
refusal behaviour when the bindings do not match. Methodology per ADR-0008.

## ADDED Requirements

### Requirement: Calibrated confidence from the raw support score
The engine SHALL map each claim's raw support score to a calibrated confidence — the
probability that the claim is fully supported by the supplied passages — using the
method and coefficients carried by the loaded calibration artifact. For the Platt
method the confidence SHALL be `sigmoid(a · logit(s) + b)` where `s` is the raw support
clamped to `[ε, 1−ε]` with ε fixed by the artifact schema version, so the confidence is
strictly inside (0, 1) — the engine MUST NOT emit a confidence of exactly 0.0 or 1.0.
The map SHALL be strictly monotone in the raw score, so ranking and discrimination are
unchanged. The raw support score MUST NOT appear in the API response; it SHALL be
carried in structured logs alongside the claim's calibrated confidence.

#### Scenario: Monotone mapping preserves ranking
- **WHEN** two claims carry raw supports `s1 < s2`
- **THEN** their calibrated confidences satisfy `c1 < c2`

#### Scenario: Saturated raw scores stay inside the open interval
- **WHEN** a claim's raw support is exactly 0.0 or exactly 1.0
- **THEN** the calibrated confidence is finite and strictly inside (0, 1)

#### Scenario: Raw score goes to logs, not the response
- **WHEN** a claim is scored and calibrated
- **THEN** a structured log line carries both the raw support and the calibrated confidence, and the API response carries only the calibrated confidence

### Requirement: Calibration artifact is a versioned portable file
The calibration artifact SHALL be a standalone versioned file referenced by path from
the versioned verifier config, carrying: an artifact schema version; the method and its
coefficients; the bindings it was fitted against — model id, model revision, inference
mode, and claim unit; the fit-set identity — dataset name, exclusion rule, a hash over
the ordered fit examples, example count, and fit date; and its measured calibration
metrics (in-domain and out-of-domain ECE with their slice identities). Loading SHALL
fail loudly if any of these fields is missing or the schema version is unknown. A refit
SHALL ship as a new artifact file together with a verifier config-version bump, never as
an in-place coefficient edit.

#### Scenario: Complete artifact loads
- **WHEN** the artifact file carries schema version, method, coefficients, all four bindings, fit-set identity, and metrics
- **THEN** it loads and its bindings are available for validation

#### Scenario: Incomplete artifact fails loudly
- **WHEN** the artifact file is missing any binding, coefficient, fit-identity, or metrics field, or names an unknown schema version
- **THEN** loading fails with an error naming the missing or unknown field, and the engine does not serve

### Requirement: Engine refuses a mismatched calibrator
At startup the engine SHALL validate the artifact's bindings: model id and revision
against the running verifier config, and inference mode and claim unit against protocol
identifiers the engine itself declares — the scoring module names its inference
protocol, the decomposition module names its claim unit, and each identifier SHALL be
bumped whenever the behaviour it names changes. Any mismatch SHALL fail startup with an
error naming every mismatched field with expected and found values. A verifier config
with no calibration artifact reference SHALL equally fail startup. The engine MUST NOT
fall back to serving raw scores; readiness SHALL stay non-200 because startup never
completes.

#### Scenario: Revision mismatch refuses to serve
- **WHEN** the verifier config pins a model revision different from the artifact's `revision` binding
- **THEN** startup fails with an error naming the `revision` field and both values, and no verify request is served

#### Scenario: Claim-unit mismatch refuses to serve
- **WHEN** the engine's declared claim-unit identifier differs from the artifact's `claim_unit` binding
- **THEN** startup fails naming the `claim_unit` field, expected and found

#### Scenario: Missing calibration is not a degraded mode
- **WHEN** the verifier config carries no calibration artifact reference, or the referenced file is absent or unreadable
- **THEN** startup fails loudly and the engine serves nothing — there is no uncalibrated fallback

### Requirement: Fit and evaluation data are disjoint
The fit tooling SHALL fit the calibrator only on examples outside every slice used to
evaluate it: the in-domain evaluation slice SHALL be excluded from the fit set by the
same deterministic slice construction the benchmark uses, and the out-of-domain
evaluation SHALL come from a dataset the calibrator was not fitted on, with subsets
verified absent from the pinned model's training mix at fit time. The artifact's
fit-set hash SHALL make the separation checkable after the fact.

#### Scenario: Benchmark slice is excluded from the fit set
- **WHEN** the calibrator is fitted on the RAGTruth test split
- **THEN** every response in the seed-18 stratified benchmark slice is excluded from the fit set, by the same slice construction the benchmark runs

#### Scenario: Out-of-domain slice never fed the fit
- **WHEN** out-of-domain calibration error is measured
- **THEN** the measured slice comes from a dataset disjoint from the fit set, with its subsets checked against the pinned model's documented training mix and the check recorded
