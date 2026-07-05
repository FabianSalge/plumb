# ADR-0007: Tier-2 response shape — span attribution and joint inference

2026-07-05 · Status: Accepted

## Context

Two decisions in the LettuceDetect migration (#27) were deliberate stepping
stones. The scoring wrapper checks the claim against each passage
independently so that `evidence_index` — the verify-api spec's promise to
name the best-supporting passage — stays honest. That costs one forward
pass per passage, under-scores claims grounded only in the *union* of
passages (multi-hop), and deviates from the configuration benchmarked in
`evals/RESULTS.md`, which prompted the whole context jointly. Once claim
decomposition lands, per-passage scoring becomes M claims × N passages —
untenable behind the fast-mode latency contract (ADR-0003). Meanwhile the
hallucination spans the model already produces go to structured logs only,
because their confidences are uncalibrated and the claim unit will change
when decomposition (#35) restructures the response.

These converge: spans in the response are the attribution mechanism that
lets `evidence_index` retire, and retiring it is what unlocks joint
inference. The question is the order of operations, since calibration
(#34, #32) must be fitted to whichever scoring mode wins — waiting leaves
the calibrator fitted to a mode being retired. Options considered:

- **Keep per-passage scoring** — preserves `evidence_index`, but stays
  permanently off the benchmarked configuration, stays blind to
  union-grounded claims, and multiplies by M when decomposition lands.
- **Joint verdict, per-passage re-score for attribution on demand** —
  recovers a best-passage index by re-running N passes when asked. Doubles
  gate-path cost exactly on failures, and "highest-scoring passage" is a
  weaker answer than "this part of the claim is unsupported" anyway.
- **Joint inference, spans with raw confidences labeled as raw** — ships a
  number people will read as a probability when it is not one.
- **Joint inference, no attribution until calibration** — a `block`
  verdict that points at nothing is hard to act on, for an interim of
  unknown length.
- **Joint inference, spans with positions only** — span *positions* come
  from the same argmax the verdict rests on and are as trustworthy as the
  verdict; only the confidence numbers need calibration. Chosen.

## Decision

The scorer moves to joint all-passages inference: one forward pass per
claim, all passages rendered into the vendored multi-passage context
format, matching the benchmarked configuration. `evidence_index` retires
in the same change; each claim instead carries `spans` — character offsets
(Unicode code points) into that claim's `text`, marking its unsupported
regions — with no confidence field until calibration produces one worth
shipping (#32). Span geometry is claim-relative and defined against the
claim unit, whatever #35 fixes that to be. The verify-api and
groundedness-scoring specs change accordingly in the implementation issue,
as one breaking change — pre-1.0, no deprecation window.

## Consequences

- The claim `score` changes meaning: support by the union of passages, not
  the max over per-passage supports. This is the scoring mode calibration
  gets fitted to — landing it before #34/#32 is the point of this ADR.
- Passage-level provenance leaves the response. Spans answer "which part
  of the claim is unsupported", not "which passage supports it"; the
  second question returns with retrieval, which knows exactly which chunks
  it fetched per claim (own ADR).
- An `unsupported` claim with zero spans is possible: the verdict
  threshold and the span-flagging threshold are different knobs, both in
  versioned config. Spans are localization, not the verdict's proof, and
  the docs say so.
- All passages now share one 4096-token model window, so truncation
  pressure grows with passage count. Truncation stays logged; budgeting
  the window properly becomes retrieval's problem.
- Per-claim cost drops from N forward passes to one, ahead of
  decomposition multiplying claim counts; #36 measures the result against
  the ADR-0003 contract.
- The API breaks once: `evidence_index` out, `spans` in, `config_version`
  and `engine_version` bumped, no interim shape.
- Span confidences enter the response only when calibrated — follow-up
  work blocked on #32, not before.
- #35 must define a claim unit consistent with claim-relative span
  geometry (each claim's `text` is the string its spans index into);
  consistency is checked both ways per that issue's acceptance criteria.
