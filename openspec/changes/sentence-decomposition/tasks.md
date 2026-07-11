# Tasks — sentence decomposition (segment-after-score)

## 1. Failing tests first

- [x] 1.1 Segmentation goldens: abbreviations, lists, code blocks, missing terminal punctuation, and multi-sentence prose; plus the invariant (`claim.text == text[start:end]` for every claim) and the total-partition property (concatenation reconstructs `text`)
- [x] 1.2 No-boundary case: text with no detectable boundary yields exactly one whole-text claim
- [x] 1.3 Invariant fail-loud: a constructed claim whose offsets disagree with its text raises a decomposition error
- [x] 1.4 Per-claim reduction: support = 1 − max risk over overlapping tokens; a boundary-straddling token counts toward both adjacent claims; zero-width special tokens excluded; out-of-range support raises
- [x] 1.5 Per-claim spans: contiguous flagged tokens merge into claim-relative spans clipped to claim boundaries, `span.text == claim.text[start:end]`; confidences stay in the log, not the return toward the API
- [x] 1.6 Gate-parity property test: over seeded random token-risk arrays and segmentations, min per-claim support equals whole-answer support, and `gate(decomposed) == gate(whole-text)` at the same threshold
- [x] 1.7 Scorer tests updated: `score(text, passages)` returns whole-answer per-token risk with answer-relative offsets from one pass; empty-probs and zero-passages still raise
- [x] 1.8 Contract tests: multi-sentence request returns multiple claims each with `start`/`end` and claim-relative spans; no-boundary request returns one claim; unsupported-with-zero-spans still holds

## 2. Engine

- [x] 2.1 `engine/decomposition.py`: `Claim(text, start, end)` frozen; `segment(text) -> list[Claim]` — deterministic rule-based, total partition, invariant enforced fail-loud, no-boundary → one claim
- [x] 2.2 `engine/decomposition.py`: per-claim reduction (max over overlapping tokens, support = 1 − max, out-of-range raises) and span clipping/re-basing (the merge logic moved from `spans_from_token_scores`, now claim-relative and clipped)
- [x] 2.3 `engine/scoring.py`: `score(text, passages)` returns the whole-answer token output (per-token risk + answer-relative offsets) instead of a reduced `ClaimScore`; keep empty-probs / zero-passages / truncation-log guarantees; drop the single-claim reduction and span derivation from the scorer
- [x] 2.4 `config/verifier.yaml`: bump `version` 0.3.0 → 0.4.0; `pyproject.toml`: bump `version` 0.1.0 → 0.2.0; chart `values.yaml` verifier config bumps in lockstep

## 3. API

- [x] 3.1 `api/schemas.py`: `ClaimResult` gains `start`/`end` (answer-relative code-point offsets); spans stay claim-relative
- [x] 3.2 `api/app.py`: orchestrate score → `segment` → per-claim reduce → `judge_claim` each → `gate_decision`; thread each claim's `start`/`end` and spans into the response

## 4. Evals — sentence-level discrimination

- [x] 4.1 Add the engine as a path dependency to `evals/` so the benchmark uses the same segmenter; parse RAGTruth `hallucination_labels` span offsets
- [x] 4.2 Sentence-level run: segment each response, label a sentence positive iff it overlaps an annotated span, score per-claim from the one whole-answer pass, compute AUROC over sentences
- [x] 4.3 `evals/RESULTS.md`: publish the sentence-level discrimination number (AUROC 0.926, 600 responses / 4,240 sentences) with a stated floor (≥ 0.70) and its protocol

## 5. Docs and verification

- [x] 5.1 README response example updated to the multi-claim shape with `start`/`end`; chart golden render test passes
- [x] 5.2 `make test`, `make lint`, `make typecheck` green; `make test-model` real-weights integration path green
- [x] 5.3 OpenSpec validate (strict) passes; sync delta specs + archive after merge
