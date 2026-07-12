# groundedness-scoring delta — calibrated span confidences

## MODIFIED Requirements

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

#### Scenario: Raw risks stay in the logs
- **WHEN** spans are derived for a scored claim
- **THEN** a structured log line carries the spans with their raw maximum token risks, and the spans returned toward the API carry positions, text, and the calibrated confidence only
