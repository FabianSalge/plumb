# calibration delta — calibrated span confidences

## ADDED Requirements

### Requirement: Calibrated span confidence from the raw span risk
The engine SHALL map each span's raw maximum token risk to a calibrated confidence — the
probability that the flagged region is genuinely unsupported by the supplied passages —
using the span coefficients carried by the loaded calibration artifact, applied through
the same arithmetic path as the claim map: for a span with raw risk `r`, the confidence
SHALL be `1 − sigmoid(a_span · logit(s) + b_span)` where `s = 1 − r` is clamped to
`[ε, 1−ε]` with ε fixed by the artifact schema version. The confidence SHALL be strictly
inside (0, 1) — the engine MUST NOT emit a span confidence of exactly 0.0 or 1.0 — and
strictly monotone increasing in the raw risk, so span ranking is unchanged. The raw
token risk MUST NOT appear in the API response; it SHALL be carried in structured logs
alongside the span's calibrated confidence, under a name that cannot be mistaken for the
calibrated value.

#### Scenario: Monotone mapping preserves span ranking
- **WHEN** two spans carry raw maximum token risks `r1 < r2`
- **THEN** their calibrated confidences satisfy `c1 < c2`

#### Scenario: Saturated raw risks stay inside the open interval
- **WHEN** a span's raw maximum token risk is exactly 0.0 or exactly 1.0
- **THEN** the calibrated span confidence is finite and strictly inside (0, 1)

#### Scenario: Raw risk goes to logs, not the response
- **WHEN** a span is derived and calibrated
- **THEN** a structured log line carries both the raw maximum token risk and the calibrated span confidence, and the API response carries only the calibrated confidence

### Requirement: Span fit data are disjoint from span evaluation data
The span calibration SHALL be fitted and evaluated on span populations derived by the
serve path itself — spans produced at the configured span-flagging threshold from
responses scored exactly as `/v1/verify` scores them — with each span labeled
unsupported iff it overlaps at least one character of a human-annotated hallucination
span. The fit population SHALL come only from responses outside every slice used to
evaluate it, by the same deterministic slice construction the benchmark uses. Whether
the served span coefficients are transferred from the claim map or fitted at span level
SHALL be decided by a rule fixed before fitting, and the held-out span-level reliability
of both candidates SHALL be published in the benchmark results. Where no span-annotated
out-of-domain data exists, the artifact and the benchmark results SHALL state that
span-level out-of-domain error is unmeasured rather than substituting a claim-level
number.

#### Scenario: Benchmark slice is excluded from the span fit population
- **WHEN** the span calibration is fitted on spans derived from the RAGTruth test split
- **THEN** every response in the seed-18 stratified benchmark slice is excluded from the fit population, by the same slice construction the benchmark runs

#### Scenario: Transfer versus fit is evidence, not assumption
- **WHEN** the served span coefficients are chosen
- **THEN** the held-out span-level calibration error of both the transferred claim map and the span-level fit is recorded in the benchmark results, together with the pre-registered rule that selected the winner

#### Scenario: Unmeasured out-of-domain error is stated, not proxied
- **WHEN** the artifact records span calibration metrics and no span-annotated out-of-domain dataset exists
- **THEN** the span metrics state that out-of-domain error is unmeasured and why, and no claim-level number stands in for it

## MODIFIED Requirements

### Requirement: Calibration artifact is a versioned portable file
The calibration artifact SHALL be a standalone versioned file referenced by path from
the versioned verifier config, carrying: an artifact schema version; the claim-level
method and its coefficients; the bindings it was fitted against — model id, model
revision, inference mode, and claim unit; the fit-set identity — dataset name, exclusion
rule, a hash over the ordered fit examples, example count, and fit date; its measured
claim-level calibration metrics (in-domain and out-of-domain ECE with their slice
identities); and a span section carrying the span method and coefficients, the
span-flagging threshold the span population was derived at, the span fit provenance —
whether the coefficients are transferred from the claim map or fitted at span level,
with the span population's dataset, label convention, span count, hash, and fit date —
and the span-level in-domain metrics with an explicit statement of whether
out-of-domain error was measured. Loading SHALL fail loudly if any of these fields is
missing or the schema version is unknown; an artifact without a span section SHALL be
refused, not served span-uncalibrated. A refit SHALL ship as a new artifact file
together with a verifier config-version bump, never as an in-place coefficient edit.

#### Scenario: Complete artifact loads
- **WHEN** the artifact file carries schema version, claim method, coefficients, all four bindings, fit-set identity, metrics, and a span section with coefficients, threshold, provenance, and metrics
- **THEN** it loads and its bindings are available for validation

#### Scenario: Incomplete artifact fails loudly
- **WHEN** the artifact file is missing any binding, coefficient, fit-identity, metrics, or span-section field, or names an unknown schema version
- **THEN** loading fails with an error naming the missing or unknown field, and the engine does not serve

#### Scenario: Pre-span artifact is refused
- **WHEN** the artifact file carries a schema version that predates the span section
- **THEN** loading fails with an error naming the found and the served schema versions, and the engine does not serve

### Requirement: Engine refuses a mismatched calibrator
At startup the engine SHALL validate the artifact's bindings: model id and revision
against the running verifier config, inference mode and claim unit against protocol
identifiers the engine itself declares — the scoring module names its inference
protocol, the decomposition module names its claim unit, and each identifier SHALL be
bumped whenever the behaviour it names changes — and the span section's span-flagging
threshold against the running config's span-flagging threshold, because the span
population the calibration was fitted on is defined by that threshold. Any mismatch
SHALL fail startup with an error naming every mismatched field with expected and found
values. A verifier config with no calibration artifact reference SHALL equally fail
startup. The engine MUST NOT fall back to serving raw scores; readiness SHALL stay
non-200 because startup never completes.

#### Scenario: Revision mismatch refuses to serve
- **WHEN** the verifier config pins a model revision different from the artifact's `revision` binding
- **THEN** startup fails with an error naming the `revision` field and both values, and no verify request is served

#### Scenario: Claim-unit mismatch refuses to serve
- **WHEN** the engine's declared claim-unit identifier differs from the artifact's `claim_unit` binding
- **THEN** startup fails naming the `claim_unit` field, expected and found

#### Scenario: Span-threshold mismatch refuses to serve
- **WHEN** the running config's span-flagging threshold differs from the threshold recorded in the artifact's span section
- **THEN** startup fails naming the span-threshold field with expected and found values, and no verify request is served

#### Scenario: Missing calibration is not a degraded mode
- **WHEN** the verifier config carries no calibration artifact reference, or the referenced file is absent or unreadable
- **THEN** startup fails loudly and the engine serves nothing — there is no uncalibrated fallback
