# groundedness-scoring — delta

## MODIFIED Requirements

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

## REMOVED Requirements

### Requirement: Per-passage support scores behind the Scorer interface
**Reason**: ADR-0007 retires per-passage scoring — it costs one forward pass per passage,
under-scores union-grounded claims, and deviates from the benchmarked configuration.
**Migration**: Replaced by "Union support score behind the Scorer interface"; callers
consume one score per claim instead of one per passage.

### Requirement: Span detail is emitted for observability
**Reason**: Spans graduate from log-only detail to the scorer's output — they are the
attribution mechanism that lets `evidence_index` retire (ADR-0007).
**Migration**: Replaced by "Spans mark unsupported claim regions"; structured logs keep
span detail including raw confidences.

## ADDED Requirements

### Requirement: Union support score behind the Scorer interface
The scorer SHALL expose `score(claim, passages)` computing the claim's support by the
union of all passages in one forward pass: the score SHALL equal one minus the maximum
per-token hallucination probability over the claim's tokens and SHALL lie in [0.0, 1.0].
A computed value outside [0.0, 1.0] SHALL raise a scorer error rather than being clamped
or passed through; a model response with no token probabilities SHALL raise a scorer
error; and scoring against zero passages SHALL raise a scorer error rather than checking
the claim against an empty context.

#### Scenario: One joint pass per claim
- **WHEN** a claim is scored against N passages
- **THEN** exactly one model inference runs, over the claim paired with all N passages rendered jointly, and the result carries a single support score

#### Scenario: Support is one minus max token risk
- **WHEN** the model returns per-token hallucination probabilities for the claim against the joint context
- **THEN** the claim's support score is 1 − max(probabilities)

#### Scenario: Out-of-range score fails loudly
- **WHEN** a computed support score falls outside [0.0, 1.0]
- **THEN** the scorer raises an error identifying the offending value

#### Scenario: Zero passages fail loudly
- **WHEN** `score(claim, passages)` is called with an empty passages list
- **THEN** the scorer raises an error instead of scoring the claim against an empty context

### Requirement: Spans mark unsupported claim regions
For each scoring call the scorer SHALL derive character-level spans of the claim by
merging contiguous tokens whose hallucination probability is at or above the
span-flagging threshold, and return them with the score. Span `start`/`end` SHALL be
Unicode code-point offsets into the claim text with `text` the corresponding substring.
The span-flagging threshold SHALL come from the versioned verifier config, never a
hardcoded constant, and is a distinct knob from the verdict threshold. Structured logs
SHALL carry the span detail including each span's raw maximum token probability; raw
confidences MUST NOT appear in the API response until calibration produces one worth
shipping.

#### Scenario: Flagged tokens produce spans
- **WHEN** the model flags contiguous claim tokens at or above the span-flagging threshold
- **THEN** the scorer returns the merged character span with offsets into the claim and the flagged substring

#### Scenario: Span threshold comes from config
- **WHEN** the span-flagging threshold in the config file changes and the service reloads
- **THEN** the same request can yield different spans without any code change, and `config_version` in the response reflects the new config

#### Scenario: Confidences stay in the logs
- **WHEN** spans are derived for a scored claim
- **THEN** a structured log line carries the spans with their raw confidences, and the spans returned toward the API carry positions and text only
