# Design — sentence decomposition (segment-after-score)

## Context

ADR-0009 fixed the claim unit — a verbatim sentence of the answer — and the decomposition
strategy — segment-after-score in fast mode. This change implements it. Today `api/app.py`
wraps the whole `text` as one claim: it calls `scorer.score(text, context)`, which runs one
whole-answer joint pass (#39), reduces to `ClaimScore(support = 1 − max token risk, spans)`,
and returns it as the single claim. Both specs still say "the whole `text` is treated as a
single claim."

The forward pass already emits what decomposition needs. `_TransformersTokenClassifier.token_probs`
returns per-token risk with offsets relative to the claim, and the claim is the whole answer, so
those offsets are answer-relative today. Segment-after-score is therefore geometry over an output
the engine already produces: partition the answer into sentences, and reduce the same token risks
per sentence instead of once over the whole text.

## Goals / Non-Goals

**Goals:**

- Deterministic rule-based sentence segmentation, no model, pinned by golden tests
  (abbreviations, lists, code blocks, missing terminal punctuation).
- Each claim carries answer-relative `start`/`end` with `claim.text == text[start:end]` enforced
  fail-loud; no detectable boundary yields one whole-text claim.
- Per-claim score = 1 − max token risk within the claim's range, from the single whole-answer
  pass; spans clipped to claim boundaries and kept claim-relative; a straddling token counts for
  every claim it overlaps.
- Gate parity pinned by a property test: decomposed gate equals whole-text gate at the same
  threshold.
- Sentence-level discrimination on RAGTruth span annotations, published in `evals/RESULTS.md`
  with a stated floor.

**Non-Goals:**

- Checkworthiness filtering — every sentence is a claim (ADR-0009); a filter that drops sentences
  loosens the gate silently.
- Finer-than-sentence granularity — future ADR, bound by the substring invariant.
- Per-claim retrieval and thorough mode — own ADR; the unit is mode-invariant.
- Calibrated span confidences (#40); spans stay positions + text only.

## Decisions

- **Segmentation is a hand-rolled rule-based function, no new dependency.** `engine/decomposition.py`
  owns `segment(text) -> list[Claim]`. The engine is deliberately dependency-light (ADR-0006
  vendored the prompt format rather than depend on lettucedetect); a sentence segmenter is small
  enough to own, and the golden tests are the real contract. Off-the-shelf options were weighed
  and rejected: statistical/model segmenters (NLTK punkt, spaCy) violate ADR-0009's "no model,
  deterministic"; pysbd is unmaintained since 2021. The rules cover the ADR's named cases and
  degrade toward coarser claims — never toward wrong offsets — on exotic text.
- **Segmentation is a total partition of `text`.** Claims tile `[0, len(text))` with no gaps:
  `"".join(c.text for c in segment(text)) == text`. Trailing inter-sentence whitespace stays
  attached to the sentence it follows. This keeps the substring invariant trivially true and,
  more importantly, guarantees every token that feeds the whole-answer max lands in some claim —
  the precondition for provable gate parity. Prettier claim text (stripped whitespace) would
  create gaps that drop token risk; the partition wins.
- **`Claim` is a frozen offset-bearing record.** `Claim(text: str, start: int, end: int)` with
  answer-relative Unicode code-point offsets. `segment` asserts `text[start:end] == claim.text`
  and raises a decomposition error on any mismatch — the invariant is enforced at construction,
  not trusted.
- **The scorer stops pre-reducing.** `score(text, passages)` returns a frozen `AnswerScores`
  carrying the whole-answer per-token risk and answer-relative offsets (today's `TokenScores`,
  surfaced instead of collapsed to one support float). The zero-passage / empty-probs / one-joint-pass
  guarantees are unchanged. Reduction and span attribution — previously inside `LettuceDetectScorer.score`
  and `spans_from_token_scores` — move to `engine/decomposition.py`, because they are now per-claim.
- **Per-claim reduction is max-over-overlap.** For claim `[cs, ce)`, a token `[ts, te)` overlaps
  iff `ts < ce and te > cs`; the claim's risk is the max token risk over its overlapping tokens,
  and support = 1 − that. A token straddling a boundary overlaps — and counts toward — both
  adjacent claims. Zero-width tokens (special tokens at offset `(0,0)`) cover no text and overlap
  no claim; they are excluded from every claim's max and, symmetrically, from the whole-answer
  reference the property test compares against, so the two definitions agree.
- **Spans are clipped and re-based per claim.** The contiguous-token merge (today's
  `spans_from_token_scores`) runs over each claim's overlapping tokens with offsets shifted to
  claim-relative and clipped to `[0, ce − cs)`. `span.text == claim.text[span.start:span.end]`.
  The span-flagging threshold stays the config value, distinct from the verdict threshold; an
  unsupported claim with zero spans stays legal.
- **Gate parity holds by construction, and a property test pins it.** Because the partition is
  total and straddling tokens count for every overlapped claim, `max over claims (claim risk)
  == max over text-covering tokens (risk)`, so `min over claims (support) == whole-answer support`.
  The gate blocks iff some claim scores below the threshold iff the whole text does. The property
  test generates seeded random token-risk arrays and segmentations and asserts both the score
  identity and `gate(decomposed) == gate(whole-text)` at the same threshold — seeded cases, not a
  new `hypothesis` dependency.
- **Claims carry `start`/`end` toward the API; spans stay claim-relative.** `ClaimResult` gains
  `start`/`end` (answer-relative). Span offsets remain claim-relative per ADR-0007, and answer
  offsets are recovered by adding the claim's `start`. `api/app.py` orchestrates
  score → decompose → judge each → gate; `judge_claim`/`gate_decision` already handle N claims.

## Risks / Trade-offs

- [A hand-rolled segmenter is only as good as its goldens] → Accepted per ADR-0009: exotic text
  degrades toward coarser claims, never toward wrong offsets, and the substring invariant is
  enforced fail-loud regardless. The goldens pin the named hard cases; new failures become new
  goldens.
- [Total partition keeps trailing whitespace inside a claim's `text`] → Accepted: provable parity
  and a trivially-true substring invariant outweigh cosmetically cleaner claim text. Callers that
  want display-trimmed text can strip; the offsets stay authoritative.
- [Absolute scores are unchanged, but the gate could in principle move] → It cannot, by
  construction, and the property test fails loudly if a refactor breaks the identity. The verdict
  threshold (0.5) benchmarked in joint mode still applies to the same per-token risks.
- [The evals project must use the same segmenter to label sentences honestly] → The engine is
  added as a path dependency to `evals/`, so there is exactly one segmenter; a divergent copy
  would measure a different unit than the one that ships.

## Migration Plan

Single breaking API change, pre-1.0, no deprecation window (ADR-0009): whole-text-as-one-claim
retires, claims gain `start`/`end`, `config_version` bumps 0.3.0 → 0.4.0 and `engine_version`
0.1.0 → 0.2.0 in the same release. The e2e smoke test and chart golden request update in the same
PR. README response example gains the multi-claim shape with offsets.
