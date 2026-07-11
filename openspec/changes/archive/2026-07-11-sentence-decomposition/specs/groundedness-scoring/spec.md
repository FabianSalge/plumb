# groundedness-scoring — delta

## REMOVED Requirements

### Requirement: Union support score behind the Scorer interface
**Reason**: Segment-after-score (ADR-0009) moves reduction off the scorer — the scorer now
returns the whole-answer per-token risk from one pass, and support is reduced per claim during
decomposition.
**Migration**: Replaced by "Whole-answer token scoring behind the Scorer interface" (the one-pass
output) and "Sentence decomposition reduces token risk per claim" (the per-claim support).

### Requirement: Spans mark unsupported claim regions
**Reason**: Spans are now derived per claim from the whole-answer pass — clipped to each claim's
boundaries and re-based claim-relative — rather than once over the whole text.
**Migration**: Replaced by "Per-claim spans mark unsupported regions".

## ADDED Requirements

### Requirement: Whole-answer token scoring behind the Scorer interface
The scorer SHALL expose `score(text, passages)` computing, in exactly one forward pass, the
answer's per-token hallucination risk against the union of all passages rendered jointly, and
SHALL return each token's risk with its offset into `text` as Unicode code points. A model
response with no token probabilities SHALL raise a scorer error; scoring against zero passages
SHALL raise a scorer error rather than checking the answer against an empty context. The scorer
SHALL NOT reduce to a per-claim score or derive spans — that is the decomposition step's work.

#### Scenario: One joint pass over the whole answer
- **WHEN** `text` is scored against N passages
- **THEN** exactly one model inference runs, over the whole answer paired with all N passages rendered jointly, and the result carries per-token risk with answer-relative code-point offsets

#### Scenario: Zero passages fail loudly
- **WHEN** `score(text, passages)` is called with an empty passages list
- **THEN** the scorer raises an error instead of scoring the answer against an empty context

#### Scenario: Empty model output fails loudly
- **WHEN** the model returns no token probabilities for the answer
- **THEN** the scorer raises an error rather than returning an emptily-scored answer

### Requirement: Sentence decomposition partitions the answer into verbatim claims
The engine SHALL decompose `text` into claims by deterministic rule-based sentence segmentation
with no model. Each claim SHALL carry answer-relative `start`/`end` (Unicode code-point offsets)
with the invariant `claim.text == text[start:end]` enforced fail-loud at construction. The claims
SHALL partition `text` with no gaps — their concatenation SHALL reconstruct `text` exactly. Text
with no detectable sentence boundary SHALL yield exactly one claim spanning the whole `text`. The
segmenter SHALL be pinned by golden tests covering abbreviations, lists, code blocks, and missing
terminal punctuation.

#### Scenario: Sentences become claims with exact offsets
- **WHEN** a multi-sentence `text` is segmented
- **THEN** each claim's `text` equals `text[start:end]`, and concatenating the claims in order reconstructs `text` with no gaps

#### Scenario: No boundary yields one whole-text claim
- **WHEN** `text` has no detectable sentence boundary
- **THEN** segmentation returns exactly one claim spanning the whole `text`

#### Scenario: Broken invariant fails loudly
- **WHEN** a claim's recorded `start`/`end` do not satisfy `claim.text == text[start:end]`
- **THEN** decomposition raises an error rather than emitting a claim with wrong offsets

#### Scenario: Golden segmentation regression
- **WHEN** the segmentation rules change the split for a pinned input (abbreviation, list, code block, or missing terminal punctuation)
- **THEN** a golden test fails

### Requirement: Sentence decomposition reduces token risk per claim
For each claim the engine SHALL compute support = 1 − the maximum token risk over the tokens of
the whole-answer pass that overlap the claim's `[start, end)` range, where a token `[ts, te)`
overlaps iff `ts < end and te > start`. A token straddling a claim boundary SHALL count toward
every claim it overlaps. Zero-width tokens (special tokens covering no answer characters) SHALL
be excluded from every claim's reduction. A computed support outside [0.0, 1.0] SHALL raise a
scorer error rather than being clamped. The maximum over all claims' risks SHALL equal the
maximum token risk over all answer-covering tokens, so the gate's decision boundary is unchanged
by decomposition.

#### Scenario: Support is one minus max overlapping token risk
- **WHEN** a claim's overlapping tokens carry known risks
- **THEN** the claim's support is 1 − the maximum of those risks

#### Scenario: Boundary-straddling token counts for both claims
- **WHEN** a token's character range straddles the boundary between two adjacent claims
- **THEN** its risk is counted in both claims' reductions, so no token risk is dropped

#### Scenario: Gate parity with the whole-answer score
- **WHEN** support is reduced per claim over a partition of the answer
- **THEN** the minimum per-claim support equals 1 − the maximum answer-covering token risk, the whole-answer support

#### Scenario: Out-of-range support fails loudly
- **WHEN** a computed per-claim support falls outside [0.0, 1.0]
- **THEN** the engine raises an error identifying the offending value

### Requirement: Per-claim spans mark unsupported regions
For each claim the engine SHALL derive character-level spans by merging contiguous overlapping
tokens whose hallucination risk is at or above the span-flagging threshold, with offsets clipped
to the claim's boundaries and re-based claim-relative so that `span.text == claim.text[span.start:span.end]`.
The span-flagging threshold SHALL come from the versioned verifier config, never a hardcoded
constant, and is a distinct knob from the verdict threshold. Structured logs SHALL carry the span
detail including each span's raw maximum token risk; raw confidences MUST NOT appear in the API
response until calibration produces one worth shipping.

#### Scenario: Flagged tokens produce claim-relative spans
- **WHEN** a claim's overlapping tokens are flagged at or above the span-flagging threshold
- **THEN** the engine returns the merged span with claim-relative offsets and the flagged substring, clipped to the claim's boundaries

#### Scenario: Span threshold comes from config
- **WHEN** the span-flagging threshold in the config file changes and the service reloads
- **THEN** the same request can yield different spans without any code change, and `config_version` in the response reflects the new config

#### Scenario: Confidences stay in the logs
- **WHEN** spans are derived for a scored claim
- **THEN** a structured log line carries the spans with their raw confidences, and the spans returned toward the API carry positions and text only
