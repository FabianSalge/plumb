# verify-api — delta

## MODIFIED Requirements

### Requirement: Verify a claim against inline evidence
The system SHALL expose `POST /v1/verify` accepting a JSON body with `text` (string, the
answer), `context` (non-empty array of evidence passage strings), and `mode` (string). The
system SHALL decompose `text` into claims, one per verbatim sentence, by deterministic
rule-based segmentation with no model. Each claim SHALL carry answer-relative `start` and
`end` (Unicode code-point offsets into `text`) with the invariant `claim.text == text[start:end]`
enforced fail-loud; the claims SHALL partition `text` with no gaps. Text with no detectable
sentence boundary SHALL yield exactly one claim spanning the whole `text`. The response SHALL
contain a `claims` array with one entry per claim, each carrying `text`, `start`, `end`,
`verdict`, `score` (0.0–1.0, the claim's support by the union of all passages), and `spans` —
an array of unsupported regions of the claim, each with `start` and `end` (Unicode code-point
offsets into that claim's `text`, claim-relative) and `text` (the flagged substring). Spans
carry no confidence field. Spans are localization, not the verdict's proof: the verdict
threshold and the span-flagging threshold are distinct configured knobs, so an `unsupported`
claim with zero spans is possible.

#### Scenario: Multi-sentence answer decomposes into one claim per sentence
- **WHEN** `text` contains more than one sentence
- **THEN** the response `claims` array carries one entry per sentence, each with `start`/`end` satisfying `claim.text == text[start:end]`, and the claims tile `text` with no gaps

#### Scenario: No sentence boundary yields one whole-text claim
- **WHEN** `text` has no detectable sentence boundary
- **THEN** the response contains exactly one claim whose `start`/`end` span the whole `text`

#### Scenario: Supported claim
- **WHEN** a claim is supported by the union of the passages in `context` and its score is at or above the configured threshold
- **THEN** the response contains that claim with verdict `supported` and its score

#### Scenario: Unsupported claim carries spans
- **WHEN** a claim's score falls below the configured threshold and the model flags tokens at or above the span-flagging threshold
- **THEN** the response contains that claim with verdict `unsupported`, its score, and `spans` marking the unsupported regions with claim-relative code-point offsets into the claim's `text`

#### Scenario: Unsupported claim with no spans
- **WHEN** a claim's score falls below the verdict threshold but no token reaches the span-flagging threshold
- **THEN** the response contains that claim with verdict `unsupported` and an empty `spans` array — the two thresholds are independent knobs

#### Scenario: Invalid request
- **WHEN** `text` is missing or empty, or `context` is missing or empty
- **THEN** the API responds 400 with a JSON error explaining which field is invalid

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
