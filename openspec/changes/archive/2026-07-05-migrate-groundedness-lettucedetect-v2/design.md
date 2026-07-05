# Design: LettuceDetect v2 migration

## Context

The engine's groundedness signal is a `Scorer` protocol (`score(claim, passages) -> list[float]`) consumed by `judge_claim`, which takes the max over per-passage scores and reports the winning passage as `evidence_index` — a semantics the verify-api spec promises. The current implementation wraps HHEM's nonstandard remote-code `predict()` over (evidence, claim) pairs.

LettuceDetect is a different shape: a token classifier (`AutoModelForTokenClassification`) that scores an answer against a formatted context prompt and returns per-token hallucination probabilities over the answer. The upstream `lettucedetect` package carries OpenAI/sklearn baggage we won't ship in the engine, so the parts the model was trained against — prompt template, pair tokenization, probability readout — get vendored (verified against the package pinned in evals/, `lettucedetect` 0.1.11).

## Goals / Non-Goals

**Goals:**
- Swap the groundedness model per ADR-0006 with zero change to `engine/verdict.py`, gate semantics, or the API response contract (protected zones).
- Vendor exactly the protocol the model was trained on — prompt format drift is silent garbage, so it must be pinned by regression tests.
- Leave the engine on current transformers (≥5) with no remote code and one Hub repo.

**Non-Goals:**
- Calibration and threshold tuning beyond the config default (own issue).
- Spans in the API response — that is an API contract change; spans land in structured logs only (promotion tracked in #29).
- Joint all-passages inference — requires retiring `evidence_index`, an API contract change (#29).
- Multilingual scoring: v2 is multilingual, but its prompt templates are per-language; the engine vendors the English template. A `lang` config field is future work if a deployment needs it.
- NLI and self-consistency slots (ADR-0004).

## Decisions

### One inference per passage, not one multi-passage prompt

LettuceDetect natively takes all passages in one prompt and returns one set of token probabilities — no per-passage attribution. The verify-api spec requires `evidence_index`: the index of the passage that produced the best score. Scoring the claim against each passage independently (a one-passage prompt per passage) keeps the `Scorer` interface, `judge_claim`, and `evidence_index` honest and untouched.

*Alternative considered:* single all-passages inference (what the benchmark measured) — one forward pass instead of N, and it sees the union of passages, so claims grounded only across passages (multi-hop) score honestly. It yields no per-passage attribution, which would force an API contract change or a fabricated `evidence_index`, so it is out of reach for this change. It is, however, the long-run destination: per-passage compute becomes M claims × N passages once decomposition lands, and per-passage scoring deviates from the benchmarked configuration. Per-passage here is a deliberate stepping stone; #29 owns the successor decision (retire `evidence_index` for span-based attribution, move to joint inference) and must land before calibration locks in a scoring mode.

### Vendored prompt format: the summary template

The package formats context with `question=None` (Plumb's verify request has no question) as the summary prompt:

```
Summarize the following text:
passage 1: {passage}
output:
```

then tokenizes `(prompt, claim)` as a sentence pair — `truncation="only_first"`, `max_length=4096` — so the claim is never truncated. Answer-token start is located by counting the claim's tokens from the end (layout `[CLS] context [SEP] claim [SEP]`). Per-token hallucination probability is `softmax(logits)[:, 1]` over claim tokens; **support = 1 − max token probability**, which is the package's own example-level rule expressed as a continuous score and exactly what evals/RESULTS.md benchmarked.

*Alternative considered:* the QA template — needs a question the API doesn't have; inventing one moves the input off the training distribution.

### Format pinned by golden regression tests

Equivalent of the HHEM pair-order tests: unit tests assert the rendered prompt byte-for-byte against a golden string, assert claim-in-answer-slot / passage-in-context-slot (a flip scores the passage instead of the claim and returns plausible garbage), assert support = 1 − max prob, one score per passage in order, and fail-loud on probabilities outside [0, 1]. A fake tokenizer/model stands in; `make test-model` (`-m model`) exercises the real weights with the same direction check as today (supported ≥ threshold > contradicted), retargeted to v2.

### Span detail goes to structured logs

The wrapper computes contiguous hallucinated-token character spans over the claim (start, end, text, confidence) and emits them on the structured logger per scoring call. Nothing changes in the response schema; observability is the log line.

Long run, spans belong in the response — "which part of the answer is unsupported" is the actionable half of a `block`. They stay out of the contract here because span confidences are uncalibrated (users would build on false precision) and the claim unit changes when decomposition restructures the response. Logs let us watch span quality on real traffic first. Promotion to an additive response field is part of #29's tier-2 response shape, where spans become the attribution mechanism that lets `evidence_index` retire — the two stepping stones converge there.

### Config, packaging, sizing

- `config/verifier.yaml`: `KRLabsOrg/lettucedect-v2-mmbert-base` @ `0f85c7a15b17aee6e8f794dae7cb4e42e2b8fdac`, threshold stays `0.5` (the package's own decision rule; calibration is a separate issue), `version` bumps to `0.2.0`. The v1-large swap (`KRLabsOrg/lettucedect-large-modernbert-en-v1` @ `22296c700ef0ba4ab3e5c9afffa0185caaf61e52`, English-only, higher in-domain accuracy) documented as a comment.
- `pyproject.toml`: `hhem` extra becomes `model` (`transformers>=5.13`, `torch>=2.12`); the `<5` pin and its epitaph comment go. Makefile (`test-model`, `run`), Dockerfile (`--extra hhem` → `--extra model`, weight-size comments), and the pytest `model` marker description follow.
- Loading: `AutoModelForTokenClassification.from_pretrained(model, revision=...)` — no `trust_remote_code`, plus the matching `AutoTokenizer`. `.eval()` and `torch.no_grad()` at inference.
- Sizing notes: weights ~1.2 GB (vs ~420 MB); chart memory guidance in `charts/plumb/values.yaml` comments and README re-derived from a measured RSS with v2 loaded (measure during implementation, don't guess).

## Risks / Trade-offs

- [Per-passage inference multiplies latency by passage count] → Tier-1 requests carry few passages; v2 is 252 ms median per response on laptop CPU, and the wrapper batches nothing yet. Acceptable at tier-1 scale; the successor inference mode is #29's decision.
- [Vendored format can drift from upstream training format] → the golden-prompt tests pin ours; the pinned revision means upstream changes can't move under us silently. When the model revision is bumped, the format must be re-verified against the package version that trained it.
- [Summary template with a one-sentence "claim" is not literally RAGTruth's summarization distribution] → same trade-off the benchmark accepted (RAGTruth summary prompts are the bare passage); the direction test on real weights guards gross mismatch, calibration work owns the fine tuning.
- [~1.2 GB first-start download vs 420 MB] → same download-at-start posture as today (`/readyz` gates readiness, `HF_HOME` caches); only the numbers in docs change.

## Migration Plan

Single PR: wrapper + config + packaging + tests + docs move together (the old extra can't load the new model and vice versa, so there is no incremental path). Rollback is `git revert` — the HHEM config points at weights that remain on the Hub.

## Open Questions

None blocking. Threshold calibration is deferred to its own issue; joint inference and span promotion are deferred to #29, which must be decided before calibration starts.
