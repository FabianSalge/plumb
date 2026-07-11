# Sentence decomposition — segment-after-score

## Why

ADR-0009 is accepted: a claim is a verbatim sentence of the answer, fast mode decomposes
by segment-after-score — one whole-answer joint-inference pass (ADR-0007, #39) with
sentence geometry applied to the token-level output — and the gate's decision boundary
provably does not move. Tier-1 treats the whole `text` as one claim while the response is
already shaped for more than one. This locks the claim unit before the calibrator is
fitted (#32), so it lands first: #32 fits per-sentence scores.

## What Changes

- Deterministic rule-based sentence segmentation enters the engine (`engine/decomposition.py`),
  no model, pinned by golden tests covering abbreviations, lists, code blocks, and missing
  terminal punctuation. Segmentation is a total partition of `text`: claims tile
  `[0, len(text))` with no gaps, so every token's risk lands in some claim.
- **BREAKING**: the whole `text` is no longer one claim. The response `claims` array carries
  one entry per sentence, each with answer-relative `start`/`end` and the invariant
  `claim.text == text[start:end]` enforced fail-loud. Text with no detectable boundary yields
  one whole-text claim — the tier-1 floor, not a fallback.
- Per-claim score = 1 − max token hallucination probability within the claim's range, from
  the single whole-answer forward pass. Spans are clipped to claim boundaries and kept
  claim-relative; a token straddling a boundary counts toward every claim it overlaps, so no
  token risk is dropped.
- The scorer stops pre-reducing to one claim: `score(text, passages)` returns the
  whole-answer token output (per-token risk with answer-relative offsets); segmentation and
  per-claim reduction move to `engine/decomposition.py`.
- Gate parity is pinned by a property test: the decomposed gate equals the whole-text gate at
  the same threshold, because the whole-answer risk equals the max over per-claim risks.
- Sentence-level discrimination is measured on RAGTruth's span annotations and published in
  `evals/RESULTS.md` with a stated floor.
- `config_version` and `engine_version` bump together as one breaking change.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `verify-api`: the whole `text` is decomposed into verbatim sentence claims; each claim
  gains `start`/`end` (answer-relative code-point offsets) with the substring invariant;
  text with no boundary yields one whole-text claim; spans stay claim-relative. Verdict and
  gate semantics unchanged, and the gate's decision boundary is unchanged by construction.
- `groundedness-scoring`: `score(text, passages)` returns the whole-answer per-token risk
  from one forward pass instead of a single reduced claim score; per-claim reduction (1 − max
  overlapping token risk) and span attribution (clipped to the claim, claim-relative) move
  into the decomposition step; a boundary-straddling token counts toward every claim it
  overlaps.

## Impact

- New `engine/decomposition.py` (segmenter, per-claim reduction, span clipping); `engine/scoring.py`
  (scorer returns whole-answer token output, no pre-reduction); `engine/verdict.py` and
  `api/schemas.py` + `api/app.py` (claims carry `start`/`end`; orchestrate score → decompose →
  judge → gate); `config/verifier.yaml` (`version` bump 0.3.0 → 0.4.0); `pyproject.toml`
  (`version` bump 0.1.0 → 0.2.0).
- `evals/` gains a sentence-level discrimination run over RAGTruth span offsets, dogfooding the
  engine segmenter (engine added as a path dependency); `evals/RESULTS.md` publishes the number
  and its floor.
- API breaks once, pre-1.0, no deprecation window (ADR-0009). README/docs wherever the response
  shape is shown update in the same PR.
- Downstream: #32 fits the calibrator to per-sentence scores; #36 measures latency, still one
  forward pass regardless of claim count.
