# groundedness-scoring

## Purpose

The engine-side contract of the groundedness signal: how the scorer loads
(pinned revision, no remote code), the vendored multi-passage prompt format it
must reproduce, how token-level hallucination probabilities reduce to one
union-support score per claim behind the `Scorer` interface, span attribution
of unsupported claim regions, and fail-loud behavior on malformed model
output.
## Requirements
### Requirement: Scorer loads from a pinned revision with no remote code
The groundedness scorer SHALL load its model and tokenizer from the single Hub repository and revision hash named in the versioned verifier config, using standard `transformers` classes only. Loading MUST NOT enable `trust_remote_code` and MUST NOT fetch any repository other than the configured one. If the scoring dependencies are not installed, loading SHALL fail with an error that names the extra to install.

#### Scenario: Pinned load
- **WHEN** the scorer is loaded from config
- **THEN** the model weights come from exactly the configured repository at the configured revision, with remote code disabled

#### Scenario: Missing dependencies fail loudly
- **WHEN** the scoring stack (transformers/torch) is not installed
- **THEN** loading raises a scorer error telling the operator which extra to install, rather than failing later at request time

### Requirement: Vendored prompt format matches the model's training format
The engine SHALL construct the scoring input itself — with no dependency on the
`lettucedetect` package — reproducing the format the pinned model was trained on: all
evidence passages rendered into the vendored context template as enumerated
`passage <i>: <text>` lines joined by newlines, tokenized as a (context, claim) sentence
pair in which the claim occupies the answer slot. When the pair exceeds the model's
maximum sequence length, only the context side SHALL be truncated, never the claim, and
the truncation log SHALL carry the total passage count. The rendered format SHALL be
pinned by a regression test against a golden string, including a multi-passage rendering.

#### Scenario: Claim occupies the answer slot
- **WHEN** a claim is scored against passages
- **THEN** the passages are rendered into the context template and the claim is tokenized as the second segment of the pair — swapping the two is a format violation the test suite catches

#### Scenario: Multi-passage rendering
- **WHEN** a claim is scored against N passages
- **THEN** the context is the N passages as `passage 1: …` through `passage N: …` lines joined by newlines, rendered into the same template as the single-passage case

#### Scenario: Golden format regression
- **WHEN** the prompt-construction code changes the rendered output for a fixed input
- **THEN** a unit test comparing against the golden string fails, for both the single-passage and multi-passage renderings

#### Scenario: Oversized context
- **WHEN** the passages plus claim exceed the maximum sequence length
- **THEN** the context is truncated to fit, the claim is scored in full, and the truncation is logged with the total passage count

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
Each derived span SHALL carry its raw maximum token risk under a name that identifies it as raw,
distinct from the calibrated confidence the calibration capability attaches on the way to the
API. The span-flagging threshold SHALL come from the versioned verifier config, never a
hardcoded constant, and is a distinct knob from the verdict threshold. Structured logs SHALL
carry the span detail including each span's raw maximum token risk; the raw risk MUST NOT
appear in the API response, where every span confidence is the calibrated value.

#### Scenario: Flagged tokens produce claim-relative spans
- **WHEN** a claim's overlapping tokens are flagged at or above the span-flagging threshold
- **THEN** the engine returns the merged span with claim-relative offsets and the flagged substring, clipped to the claim's boundaries

#### Scenario: Span threshold comes from config
- **WHEN** the span-flagging threshold in the config file changes and the service reloads
- **THEN** the same request can yield different spans without any code change, and `config_version` in the response reflects the new config

#### Scenario: Confidences stay in the logs
- **WHEN** spans are derived for a scored claim
- **THEN** a structured log line carries the spans with their raw maximum token risks, and the spans returned toward the API carry positions, text, and the calibrated confidence only
