# verify-api — delta

## MODIFIED Requirements

### Requirement: Verify a claim against inline evidence
The system SHALL expose `POST /v1/verify` accepting a JSON body with `text` (string, the
answer or claim), `context` (non-empty array of evidence passage strings), and `mode`
(string). In Tier-1 the whole `text` is treated as a single claim; no decomposition
occurs. The response SHALL contain a `claims` array with one entry per claim, each
carrying `text`, `verdict`, `score` (0.0–1.0, the claim's support by the union of all
passages), and `spans` — an array of unsupported regions of the claim, each with `start`
and `end` (Unicode code-point offsets into that claim's `text`) and `text` (the flagged
substring). Spans carry no confidence field. Spans are localization, not the verdict's
proof: the verdict threshold and the span-flagging threshold are distinct configured
knobs, so an `unsupported` claim with zero spans is possible.

#### Scenario: Supported claim
- **WHEN** `text` is supported by the union of the passages in `context` and its score is at or above the configured threshold
- **THEN** the response contains one claim with verdict `supported` and its score

#### Scenario: Unsupported claim carries spans
- **WHEN** the claim's score falls below the configured threshold and the model flags tokens at or above the span-flagging threshold
- **THEN** the response contains one claim with verdict `unsupported`, its score, and `spans` marking the unsupported regions with code-point offsets into the claim's `text`

#### Scenario: Unsupported claim with no spans
- **WHEN** the claim's score falls below the verdict threshold but no token reaches the span-flagging threshold
- **THEN** the response contains one claim with verdict `unsupported` and an empty `spans` array — the two thresholds are independent knobs

#### Scenario: Invalid request
- **WHEN** `text` is missing or empty, or `context` is missing or empty
- **THEN** the API responds 400 with a JSON error explaining which field is invalid

### Requirement: Verdicts map from score via versioned config
The verdict SHALL be derived by comparing the grounding score against a threshold read
from a versioned config file, never a hardcoded constant. The span-flagging threshold
SHALL likewise come from the same versioned config, as a separate per-model value. The
config SHALL identify the signal model by name and pinned revision hash, and both
thresholds SHALL be defined per-model — a later model swap is a config-version bump by
construction, never a silent verdict change. The verdict vocabulary in Tier-1 is exactly
`supported` and `unsupported`; `contradicted` MUST NOT appear until an NLI signal exists.

#### Scenario: Threshold comes from config
- **WHEN** the threshold in the config file changes and the service reloads
- **THEN** the same request can yield a different verdict without any code change, and `config_version` in the response reflects the new config

#### Scenario: Config names the model it calibrates
- **WHEN** the config file is loaded
- **THEN** it carries the signal model's name and revision hash alongside the verdict and span-flagging thresholds, and loading fails loudly if any is missing
