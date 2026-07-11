# ADR-0009: Claim decomposition — sentences as the unit of verification

2026-07-06 · Status: Accepted

## Context

Tier-1 deliberately treats the whole `text` as one claim — the verify-api
spec says so explicitly — while the response is already shaped for more
than one. Everything week 2 decides leans on what a claim actually is:
ADR-0007 defines span geometry against the post-decomposition claim unit,
calibration (#34, #32) maps per-claim risk to confidence, and the gate
blocks on any unsupported claim. That definition should not be made
implicitly by whichever implementation lands first.

Two facts bound the choice. First, ADR-0007 makes spans character offsets
into each claim's `text`. If a claim's text is a model-rewritten
proposition, spans index into synthetic text and can never be mapped back
to the answer the caller sent; if it is a verbatim contiguous substring of
`text`, claim-relative spans compose to answer offsets by adding the
claim's own offset. Second, the groundedness model is token-level over the
whole answer — the benchmarked configuration (`evals/RESULTS.md`) scored
entire multi-sentence responses in the answer slot — so per-claim scores
can be derived from one whole-answer forward pass by segmenting the token
output, instead of one pass per claim. Options considered:

- **Atomic propositions via model rewrite** — the finest unit and the
  literal reading of ADR-0004's "atomic claim", but the rewritten text
  breaks span geometry, and it buys that with a new model dependency, its
  latency, and nondeterministic decomposition as a fresh failure surface.
- **Sub-sentence clauses, rule-based** — keeps the substring invariant,
  but clause-splitting rules are brittle across styles, and the extra
  granularity buys little when spans already localize within a sentence.
- **Decompose only in thorough mode** — fast mode keeps one claim per
  response, so the claim unit differs by mode: calibration faces two
  units, and ADR-0003's promise that mode changes the work per claim,
  never the scoring semantics, is broken at the root.
- **Per-sentence forward passes in fast mode** — scores each sentence in
  isolation, but latency scales linearly with sentence count and busts
  the sub-second contract on CPU by the fifth sentence.
- **Verbatim sentences, segment-after-score in fast mode** — one
  whole-answer pass, sentence geometry applied to the token output.
  Chosen.

## Decision

A claim is a sentence of the answer, verbatim. Each claim carries
answer-relative `start`/`end` with the invariant
`claim.text == text[start:end]`, enforced fail-loud; the granularity
promised is the sentence, and the substring invariant is the part any
finer future decomposition must preserve — breaking it takes a new ADR.
v0 decomposition is deterministic rule-based sentence segmentation — no
model, pinned by golden tests, the exact segmenter chosen in the
implementation issue — and every sentence is a claim: no checkworthiness
filter, because a filter that drops sentences loosens the gate silently.
Fast mode's "decomposition-light" (ADR-0003) means segment-after-score:
one whole-answer joint-inference pass (ADR-0007), then per-claim score =
1 − max token hallucination probability within the claim's range, spans
clipped to claim boundaries and kept claim-relative. The unit is
mode-invariant; thorough mode changes the work done per claim, never the
unit. Before this replaces whole-text-as-one-claim it must clear:
sentence-level discrimination measured on RAGTruth's span annotations and
published in `evals/RESULTS.md` with a stated floor, segmentation golden
tests, and a gate-parity property test.

## Consequences

- The gate's decision boundary provably does not move: the whole-answer
  score equals the min over per-claim scores, so some claim scoring below
  the shared threshold is exactly the whole text scoring below it.
  Decomposition refines attribution; it does not resettle verdicts.
- The claim unit is locked before calibration: #34/#32 fit per-sentence
  scores derived from the whole-answer pass, and a change of unit joins
  the list of events that force a refit.
- Consistency with ADR-0007 holds both ways: each claim's `text` is the
  string its spans index into, spans stay claim-relative, and answer
  offsets are recovered by adding the claim's `start`.
- The verify-api spec changes: "the whole `text` is treated as a single
  claim" retires, claims gain `start`/`end`, and the change rides the
  implementation issue as one breaking change, pre-1.0.
- Sequencing: implementation lands after #39, because per-sentence scores
  are cut from the token-level output joint inference exposes; #36
  measures the result, which stays one forward pass regardless of claim
  count.
- A multi-fact sentence is one claim; spans localize the unsupported part
  inside it. Finer granularity is future work bound by the substring
  invariant.
- Text with no detectable sentence boundary yields one claim spanning the
  whole `text` — tier-1 behavior as the defined floor, not a fallback.
  Segmentation partitions the full text, and a token straddling a
  boundary counts toward every claim it overlaps, so no token risk is
  dropped.
- The rule-based segmenter is only as good as its goldens (abbreviations,
  lists, code blocks, missing terminal punctuation); exotic text degrades
  toward coarser claims, never toward wrong offsets.
- Boilerplate sentences ("Sure, here's what I found") are claims too; the
  token-level model rarely flags them, and keeping them is what keeps the
  gate honest.
