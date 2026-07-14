# verify-api (delta)

## MODIFIED Requirements

### Requirement: Verify a claim against inline evidence
The system SHALL expose `POST /v1/verify` accepting a JSON body with `text` (string, the
answer), `context` (array of evidence passage strings — required non-empty in fast mode,
optional in thorough mode), and `mode` (string). The system SHALL decompose `text` into
claims, one per verbatim sentence, by deterministic rule-based segmentation with no model.
Each claim SHALL carry answer-relative `start` and `end` (Unicode code-point offsets into
`text`) with the invariant `claim.text == text[start:end]` enforced fail-loud; the claims
SHALL partition `text` with no gaps. Text with no detectable sentence boundary SHALL yield
exactly one claim spanning the whole `text`. The response SHALL contain a `claims` array
with one entry per claim, each carrying `text`, `start`, `end`, `verdict`, `confidence`
(strictly inside (0, 1) — the calibrated probability that the claim is fully supported by
the union of all passages, per the loaded calibration artifact), and `spans` — an array of
unsupported regions of the claim, each with `start` and `end` (Unicode code-point offsets
into that claim's `text`, claim-relative), `text` (the flagged substring), and `confidence`
(strictly inside (0, 1) — the calibrated probability that the flagged region is genuinely
unsupported by the supplied passages, per the span calibration carried by the loaded
artifact). Raw scores — the claim's raw support and any span's raw token risk — MUST NOT
appear in the response. Spans are localization, not the verdict's proof: the verdict
threshold and the span-flagging threshold are distinct configured knobs, so an `unsupported`
claim with zero spans is possible.

#### Scenario: Multi-sentence answer decomposes into one claim per sentence
- **WHEN** `text` contains more than one sentence
- **THEN** the response `claims` array carries one entry per sentence, each with `start`/`end` satisfying `claim.text == text[start:end]`, and the claims tile `text` with no gaps

#### Scenario: No sentence boundary yields one whole-text claim
- **WHEN** `text` has no detectable sentence boundary
- **THEN** the response contains exactly one claim whose `start`/`end` span the whole `text`

#### Scenario: Supported claim
- **WHEN** a claim is supported by the union of the passages in the scored evidence and its calibrated confidence is at or above the configured threshold
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

#### Scenario: Invalid fast-mode request
- **WHEN** `mode` is `fast` and `text` is missing or empty, or `context` is missing or empty
- **THEN** the API responds 400 with a JSON error explaining which field is invalid

#### Scenario: Invalid thorough-mode request
- **WHEN** `mode` is `thorough` and `text` is missing or empty
- **THEN** the API responds 400 with a JSON error explaining which field is invalid — a missing `context` alone is not an error in thorough mode

### Requirement: Gate decision
The response SHALL include a `gate` field. The gate is `pass` when every claim's verdict is
`supported`, and `block` otherwise. Decomposition SHALL NOT move the gate's decision boundary
in either mode: over the same scored evidence, the decomposed gate SHALL equal the gate
computed over the whole `text` as a single claim at the same threshold, because the
whole-answer risk equals the maximum over per-claim risks.

#### Scenario: Gate blocks on unsupported claim
- **WHEN** any claim in the response has verdict `unsupported`
- **THEN** `gate` is `block`

#### Scenario: Gate parity with the whole-text decision
- **WHEN** the same `text` and the same scored evidence are scored decomposed and as a single whole-text claim, at the same threshold, in either mode
- **THEN** the two `gate` decisions are identical

## REMOVED Requirements

### Requirement: Only fast mode is accepted
**Reason**: Thorough mode ships (ADR-0010); the mode vocabulary is now `fast` and
`thorough`, and rejection applies only to unknown modes.
**Migration**: Callers sending `mode: "fast"` are unaffected. Callers previously receiving
400 for `mode: "thorough"` now get a verification against the deployment's tenant store —
or a 400 stating the deployment is fast-only when no store is configured.

## ADDED Requirements

### Requirement: Thorough mode verifies against the tenant store
The system SHALL accept `mode: "thorough"` and, in that mode, retrieve evidence per claim
from the deployment's configured tenant store (per the evidence-retrieval capability),
pool it with any caller-provided `context` passages, and score the answer in the same
single joint pass as fast mode. Any mode other than `fast` or `thorough` SHALL be rejected
with 400, not silently degraded. When no tenant store is configured, a thorough request
SHALL be rejected with 400 stating the deployment is fast-only. When the store fails
mid-request, the API SHALL respond 502 with an error naming the store problem — never a
verdict computed on partial evidence. Each claim in a thorough response SHALL carry
`evidence`: an array of references to the chunks its query retrieved that made the scoring
window, each with source identity, chunk identity, retrieval rank, and — where the store
exposes one — a snapshot identity. The docs SHALL state plainly that `evidence` is
retrieval provenance ("retrieved for this claim"), not support attribution ("supports this
claim"). Fast-mode responses SHALL NOT carry retrieved evidence; fast mode's contract does
not move.

#### Scenario: Thorough mode without context
- **WHEN** a thorough request carries only `text` and the deployment has a store configured
- **THEN** the answer is verified against retrieved evidence and each claim carries its `evidence` provenance

#### Scenario: Caller context joins the pool
- **WHEN** a thorough request carries `context` passages
- **THEN** those passages are scored alongside retrieved evidence, and the response's per-claim `evidence` lists only store-retrieved chunks — caller passages are the caller's own

#### Scenario: Unknown mode is rejected
- **WHEN** the request carries a mode other than `fast` or `thorough`
- **THEN** the API responds 400 naming the supported modes

#### Scenario: Thorough on a fast-only deployment
- **WHEN** a thorough request reaches a deployment with no tenant store configured
- **THEN** the API responds 400 stating the deployment is fast-only

#### Scenario: Store outage fails loudly
- **WHEN** the tenant store errors during retrieval
- **THEN** the API responds 502 naming the store problem, and no verdict is returned

#### Scenario: Evidence carries snapshot identity when the store exposes one
- **WHEN** the configured store exposes a snapshot identity for its chunks
- **THEN** each evidence entry carries it, and when the store exposes none the field is absent — never invented
