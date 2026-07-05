# ADR-0004: Signal stack — orchestrate detectors, don't train them

2026-07-05 · Status: Accepted

## Context

Per-claim grounding checks can come from several families of detector:
groundedness cross-encoders (HHEM and its successors), NLI entailment
models, and self-consistency sampling from a served LLM judge. The
specialist-detector field moves fast and is partly commercial: the
[HHEM spike](../spikes/2026-07-04-fastapi-hhem.md) found the best-known
open detector frozen upstream mid-2024, with newer versions API-only, while
actively maintained alternatives (LettuceDetect, MiniCheck, Granite
Guardian) keep appearing. Competing on detector quality — training or
fine-tuning our own — is a race against funded specialist teams, and any
single detector we hard-wire today is likely to be the wrong one within a
year.

## Decision

Plumb orchestrates detectors rather than training them. The engine runs a
stack of pluggable signals per atomic claim — a groundedness cross-encoder,
an NLI entailment signal, and LLM self-consistency — each behind a common
signal interface, aggregated into a per-claim risk score that calibration
maps to a confidence. Which model fills the groundedness slot is a separate,
swappable decision (being made in #18); this ADR fixes the shape of the
stack, not its occupants.

## Consequences

- When a better detector ships, adopting it is a config-and-adapter change
  plus a benchmark run, not an architecture change.
- The product's accuracy claim rests on aggregation and calibration, which
  makes that math the trust-critical core of the codebase — it is a
  protected zone, and changes to it need evidence, not refactoring energy.
- Every signal added buys accuracy with latency, which is exactly the
  trade-off the tiered modes (ADR-0003) exist to price: fast mode runs one
  signal, thorough mode runs the stack.
- Signal disagreement is information, not noise — it is emitted in
  observability output rather than averaged away silently.
- No model training or fine-tuning enters the roadmap; benchmark effort
  goes into evaluating candidate detectors and our aggregate, not improving
  a detector.
