# groundedness-scoring

## ADDED Requirements

### Requirement: Scorer loads from a pinned revision with no remote code
The groundedness scorer SHALL load its model and tokenizer from the single Hub repository and revision hash named in the versioned verifier config, using standard `transformers` classes only. Loading MUST NOT enable `trust_remote_code` and MUST NOT fetch any repository other than the configured one. If the scoring dependencies are not installed, loading SHALL fail with an error that names the extra to install.

#### Scenario: Pinned load
- **WHEN** the scorer is loaded from config
- **THEN** the model weights come from exactly the configured repository at the configured revision, with remote code disabled

#### Scenario: Missing dependencies fail loudly
- **WHEN** the scoring stack (transformers/torch) is not installed
- **THEN** loading raises a scorer error telling the operator which extra to install, rather than failing later at request time

### Requirement: Vendored prompt format matches the model's training format
The engine SHALL construct the scoring input itself — with no dependency on the `lettucedetect` package — reproducing the format the pinned model was trained on: the passage rendered into the vendored context template, tokenized as a (context, claim) sentence pair in which the claim occupies the answer slot. When the pair exceeds the model's maximum sequence length, only the context side SHALL be truncated, never the claim. The rendered format SHALL be pinned by a regression test against a golden string.

#### Scenario: Claim occupies the answer slot
- **WHEN** a claim is scored against a passage
- **THEN** the passage is rendered into the context template and the claim is tokenized as the second segment of the pair — swapping the two is a format violation the test suite catches

#### Scenario: Golden format regression
- **WHEN** the prompt-construction code changes the rendered output for a fixed input
- **THEN** a unit test comparing against the golden string fails

#### Scenario: Oversized context
- **WHEN** passage plus claim exceed the maximum sequence length
- **THEN** the context is truncated to fit, the claim is scored in full, and the truncation is logged

### Requirement: Per-passage support scores behind the Scorer interface
The scorer SHALL expose `score(claim, passages)` returning one support score per passage, in passage order, computed by scoring the claim against each passage independently. Each score SHALL equal one minus the maximum per-token hallucination probability over the claim's tokens and SHALL lie in [0.0, 1.0]. A computed value outside [0.0, 1.0] SHALL raise a scorer error rather than being clamped or passed through.

#### Scenario: One score per passage in order
- **WHEN** a claim is scored against N passages
- **THEN** the result is a list of N floats whose i-th entry is the support of the claim by passage i

#### Scenario: Support is one minus max token risk
- **WHEN** the model returns per-token hallucination probabilities for the claim against one passage
- **THEN** the passage's support score is 1 − max(probabilities)

#### Scenario: Out-of-range score fails loudly
- **WHEN** a computed support score falls outside [0.0, 1.0]
- **THEN** the scorer raises an error identifying the offending value

### Requirement: Span detail is emitted for observability
For each scoring call the scorer SHALL derive character-level spans of the claim from contiguous tokens classified as hallucinated (start, end, text, confidence) and emit them as structured log detail. Span detail MUST NOT appear in the API response in this change.

#### Scenario: Unsupported content produces logged spans
- **WHEN** the model flags tokens of the claim as hallucinated against the best passage
- **THEN** a structured log line carries the corresponding character spans with their confidences

#### Scenario: Response contract untouched
- **WHEN** any verify request is served
- **THEN** the response contains exactly the fields promised by the verify-api spec, with no span fields added
