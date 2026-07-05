# verify-api — delta for add-verify-endpoint

## ADDED Requirements

### Requirement: Verify a claim against inline evidence
The system SHALL expose `POST /v1/verify` accepting a JSON body with `text` (string, the answer or claim), `context` (non-empty array of evidence passage strings), and `mode` (string). In Tier-1 the whole `text` is treated as a single claim; no decomposition occurs. The response SHALL contain a `claims` array with one entry per claim, each carrying `text`, `verdict`, `score` (0.0–1.0), and `evidence_index` (the index into `context` of the passage that produced the highest score).

#### Scenario: Supported claim
- **WHEN** `text` is entailed by at least one passage in `context` and its top score is at or above the configured threshold
- **THEN** the response contains one claim with verdict `supported`, its score, and the `evidence_index` of the best-scoring passage

#### Scenario: Unsupported claim
- **WHEN** no passage in `context` yields a score at or above the configured threshold
- **THEN** the response contains one claim with verdict `unsupported` and the best score found

#### Scenario: Invalid request
- **WHEN** `text` is missing or empty, or `context` is missing or empty
- **THEN** the API responds 400 with a JSON error explaining which field is invalid

### Requirement: Verdicts map from score via versioned config
The verdict SHALL be derived by comparing the grounding score against a threshold read from a versioned config file, never a hardcoded constant. The config SHALL identify the signal model by name and pinned revision hash, and the threshold SHALL be defined per-model — a later model swap is a config-version bump by construction, never a silent verdict change. The verdict vocabulary in Tier-1 is exactly `supported` and `unsupported`; `contradicted` MUST NOT appear until an NLI signal exists.

#### Scenario: Threshold comes from config
- **WHEN** the threshold in the config file changes and the service reloads
- **THEN** the same request can yield a different verdict without any code change, and `config_version` in the response reflects the new config

#### Scenario: Config names the model it calibrates
- **WHEN** the config file is loaded
- **THEN** it carries the signal model's name and revision hash alongside the threshold, and loading fails loudly if either is missing

### Requirement: Gate decision
The response SHALL include a `gate` field. The gate is `pass` when every claim's verdict is `supported`, and `block` otherwise.

#### Scenario: Gate blocks on unsupported claim
- **WHEN** any claim in the response has verdict `unsupported`
- **THEN** `gate` is `block`

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
