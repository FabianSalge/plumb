# Tasks — joint inference and span attribution

## 1. Failing tests first

- [x] 1.1 Golden regression test for the multi-passage rendering (`passage 1: …\npassage 2: …` into the summary template), alongside the updated single-passage golden
- [x] 1.2 Scorer tests: one `token_probs` call for N passages; result carries union support (1 − max token prob) and spans; out-of-range and empty-probs still raise; truncation log carries total passage count
- [x] 1.3 Config tests: `span_threshold` required per-model, load fails loudly without it
- [x] 1.4 Verdict tests: `judge_claim(text, score, threshold)` maps score to verdict, no `evidence_index`; gate semantics unchanged
- [x] 1.5 Contract tests: response claims carry `spans` (start/end/text, no confidence) and no `evidence_index`; unsupported-with-zero-spans case; span threshold read from config

## 2. Engine

- [x] 2.1 `render_prompt(passages: list[str])` — enumerated passage lines joined by newline into the vendored template; docstring re-pinned to lettucedetect 0.2.1
- [x] 2.2 `LettuceDetectScorer.score` — one joint forward pass; returns frozen `ClaimScore(support, spans)`; spans thresholded by config value (`_FLAG_THRESHOLD` removed); structured span log keeps confidences; truncation warning notes passage count
- [x] 2.3 `SignalModelConfig.span_threshold`; `config/verifier.yaml` gains the value and bumps `version` to 0.3.0
- [x] 2.4 `judge_claim` narrowed to (text, score, threshold); `ClaimVerdict` loses `evidence_index`

## 3. API

- [x] 3.1 `ClaimResult`: drop `evidence_index`, add `spans: list[SpanResult]` (start, end, text)
- [x] 3.2 `api/app.py`: wire scorer result through verdict and spans into the response

## 4. Docs and verification

- [x] 4.1 README response example updated (spans in, evidence_index out); check chart/e2e golden request still passes
- [x] 4.2 `make test`, `make lint`, `make typecheck` green; run `make test-model` for the real-weights integration path
- [x] 4.3 OpenSpec validate + sync delta specs; archive after merge (archive pending merge)
