# verify-api delta — calibrated span confidences

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
`verdict`, `confidence` (strictly inside (0, 1) — the calibrated probability that the claim is
fully supported by the union of all passages, per the loaded calibration artifact), and
`spans` — an array of unsupported regions of the claim, each with `start` and `end` (Unicode
code-point offsets into that claim's `text`, claim-relative), `text` (the flagged substring),
and `confidence` (strictly inside (0, 1) — the calibrated probability that the flagged region
is genuinely unsupported by the supplied passages, per the span calibration carried by the
loaded artifact). Raw scores — the claim's raw support and any span's raw token risk — MUST
NOT appear in the response. Spans are localization, not the verdict's proof: the verdict
threshold and the span-flagging threshold are distinct configured knobs, so an `unsupported`
claim with zero spans is possible.

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
- **THEN** the response contains that claim with verdict `unsupported`, its calibrated confidence, and `spans` marking the unsupported regions with claim-relative code-point offsets into the claim's `text`, each span carrying its calibrated confidence

#### Scenario: Span confidence is calibrated and inside the open interval
- **WHEN** a response span was flagged from a raw maximum token risk of exactly 0.0 or exactly 1.0
- **THEN** the span's `confidence` is finite, strictly inside (0, 1), and is the calibrated value — the raw token risk appears only in structured logs

#### Scenario: Unsupported claim with no spans
- **WHEN** a claim's calibrated confidence falls below the verdict threshold but no token reaches the span-flagging threshold
- **THEN** the response contains that claim with verdict `unsupported` and an empty `spans` array — the two thresholds are independent knobs

#### Scenario: Invalid request
- **WHEN** `text` is missing or empty, or `context` is missing or empty
- **THEN** the API responds 400 with a JSON error explaining which field is invalid
