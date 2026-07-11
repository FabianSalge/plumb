# verify-api

## Purpose

The HTTP contract of `/v1/verify` — request/response shapes, verdict semantics,
gate decision, version stamping, and health probes. Tier-1 scope: the input
text is one claim, checked against caller-provided evidence.
## Requirements
### Requirement: Verify a claim against inline evidence
The system SHALL expose `POST /v1/verify` accepting a JSON body with `text` (string, the
answer), `context` (non-empty array of evidence passage strings), and `mode` (string). The
system SHALL decompose `text` into claims, one per verbatim sentence, by deterministic
rule-based segmentation with no model. Each claim SHALL carry answer-relative `start` and
`end` (Unicode code-point offsets into `text`) with the invariant `claim.text == text[start:end]`
enforced fail-loud; the claims SHALL partition `text` with no gaps. Text with no detectable
sentence boundary SHALL yield exactly one claim spanning the whole `text`. The response SHALL
contain a `claims` array with one entry per claim, each carrying `text`, `start`, `end`,
`verdict`, `confidence` (strictly inside (0, 1) — the calibrated probability that the claim is
fully supported by the union of all passages, per the loaded calibration artifact), and
`spans` — an array of unsupported regions of the claim, each with `start` and `end` (Unicode
code-point offsets into that claim's `text`, claim-relative) and `text` (the flagged
substring). The raw support score MUST NOT appear in the response. Spans carry no confidence
field. Spans are localization, not the verdict's proof: the verdict threshold and the
span-flagging threshold are distinct configured knobs, so an `unsupported` claim with zero
spans is possible.

#### Scenario: Multi-sentence answer decomposes into one claim per sentence
- **WHEN** `text` contains more than one sentence
- **THEN** the response `claims` array carries one entry per sentence, each with `start`/`end` satisfying `claim.text == text[start:end]`, and the claims tile `text` with no gaps

#### Scenario: No sentence boundary yields one whole-text claim
- **WHEN** `text` has no detectable sentence boundary
- **THEN** the response contains exactly one claim whose `start`/`end` span the whole `text`

#### Scenario: Supported claim
- **WHEN** a claim is supported by the union of the passages in `context` and its calibrated confidence is at or above the configured threshold
- **THEN** the response contains that claim with verdict `supported` and its calibrated confidence

#### Scenario: Unsupported claim carries spans
- **WHEN** a claim's calibrated confidence falls below the configured threshold and the model flags tokens at or above the span-flagging threshold
- **THEN** the response contains that claim with verdict `unsupported`, its calibrated confidence, and `spans` marking the unsupported regions with claim-relative code-point offsets into the claim's `text`

#### Scenario: Unsupported claim with no spans
- **WHEN** a claim's calibrated confidence falls below the verdict threshold but no token reaches the span-flagging threshold
- **THEN** the response contains that claim with verdict `unsupported` and an empty `spans` array — the two thresholds are independent knobs

#### Scenario: Invalid request
- **WHEN** `text` is missing or empty, or `context` is missing or empty
- **THEN** the API responds 400 with a JSON error explaining which field is invalid

### Requirement: Verdicts map from score via versioned config
The verdict SHALL be derived by comparing the claim's calibrated confidence against a
threshold read from a versioned config file, never a hardcoded constant — the raw score
reaches the verdict only through the calibration artifact the config references. The
span-flagging threshold SHALL likewise come from the same versioned config, as a separate
per-model value (it applies to raw token risks, which remain uncalibrated until span
confidences ship). The config SHALL identify the signal model by name and pinned revision
hash, the calibration artifact by path, and both thresholds SHALL be defined per-model — a
later model swap is a config-version bump plus a refitted artifact by construction, never a
silent verdict change. The verdict vocabulary in Tier-1 is exactly `supported` and
`unsupported`; `contradicted` MUST NOT appear until an NLI signal exists.

#### Scenario: Threshold comes from config
- **WHEN** the threshold in the config file changes and the service reloads
- **THEN** the same request can yield a different verdict without any code change, and `config_version` in the response reflects the new config

#### Scenario: Config names the model it calibrates
- **WHEN** the config file is loaded
- **THEN** it carries the signal model's name and revision hash, the calibration artifact path, and the verdict and span-flagging thresholds, and loading fails loudly if any is missing

### Requirement: Gate decision
The response SHALL include a `gate` field. The gate is `pass` when every claim's verdict is
`supported`, and `block` otherwise. Decomposition SHALL NOT move the gate's decision boundary:
the decomposed gate SHALL equal the gate computed over the whole `text` as a single claim at the
same threshold, because the whole-answer risk equals the maximum over per-claim risks.

#### Scenario: Gate blocks on unsupported claim
- **WHEN** any claim in the response has verdict `unsupported`
- **THEN** `gate` is `block`

#### Scenario: Gate parity with the whole-text decision
- **WHEN** the same `text` and `context` are scored decomposed and as a single whole-text claim, at the same threshold
- **THEN** the two `gate` decisions are identical

### Requirement: Only fast mode is accepted
Tier-1 supports `mode: "fast"` only. Any other mode value SHALL be rejected, not silently degraded.

#### Scenario: Thorough mode requested too early
- **WHEN** the request carries `mode: "thorough"` (or any unknown mode)
- **THEN** the API responds 400 stating that only `fast` is currently supported

### Requirement: Version stamping
Every successful response SHALL carry `engine_version` (the build/release identifier of the engine) and `config_version` (the version of the threshold config used). This is the seed of verdict pinning.

#### Scenario: Versions present
- **WHEN** any verify request succeeds
- **THEN** the response contains non-empty `engine_version` and `config_version` fields

### Requirement: Health probes
The service SHALL expose `GET /healthz` (liveness: 200 whenever the process is serving) and `GET /readyz` (readiness: 200 only once the scoring model is loaded and a verification can succeed).

#### Scenario: Not ready before model load
- **WHEN** the service has started but the model is not yet loaded
- **THEN** `/healthz` returns 200 and `/readyz` returns a non-200 status

### Requirement: Structured request logging
Every request SHALL emit structured JSON log lines carrying a request ID. If the caller provides an `X-Request-ID` header it SHALL be propagated; otherwise one is generated.

#### Scenario: Request ID propagation
- **WHEN** a request arrives with header `X-Request-ID: abc-123`
- **THEN** all log lines for that request carry `abc-123` as the request ID
