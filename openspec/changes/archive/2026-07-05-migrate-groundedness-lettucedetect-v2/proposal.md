# Migrate the groundedness signal to LettuceDetect v2

## Why

ADR-0006 retired HHEM-2.1-open as the groundedness signal: it is frozen upstream, loads only with `trust_remote_code`, pulls a second Hub repo at runtime, and pins the engine to `transformers<5` forever. The benchmark (evals/RESULTS.md) selected `lettucedect-v2-mmbert-base` as the replacement; this change performs the migration the ADR deferred to #27.

## What Changes

- The scoring wrapper speaks LettuceDetect's token-classification protocol: prompt format vendored into the engine (no dependency on the `lettucedetect` package), span probabilities reduced to a per-claim support score, span detail retained for observability via structured logs.
- The existing `Scorer` interface (`score(claim, passages) -> list[float]`) and `evidence_index` semantics are preserved by scoring the claim against each passage independently — `engine/verdict.py` and the API response contract do not change.
- `config/verifier.yaml` points at `KRLabsOrg/lettucedect-v2-mmbert-base` pinned to revision `0f85c7a15b17aee6e8f794dae7cb4e42e2b8fdac` with a per-model threshold; `config_version` bumped; the `lettucedect-large-modernbert-en-v1` swap documented.
- **BREAKING (dev env)**: the `hhem` extra and its `transformers<5` pin are removed; the engine targets current transformers (≥5, required by v2's tokenizer) under a new `model` extra. HHEM-specific machinery (`HHEMScorer`, `evidence_claim_pairs`, pair-order tests) is retired.
- Format/order regression tests equivalent to the HHEM pair-order tests; `make test-model` retargeted to the new weights.
- Image/chart sizing notes and README updated for the new weights (~1.2 GB vs ~420 MB, one Hub repo, no `trust_remote_code`).

## Capabilities

### New Capabilities

- `groundedness-scoring`: the engine-side contract of the groundedness signal — how the scorer loads (pinned revision, no remote code), the vendored prompt format it must reproduce, how token-level hallucination probabilities reduce to a per-claim support score in [0, 1], span detail for observability, and fail-loud behavior on malformed model output.

### Modified Capabilities

<!-- none — verify-api's request/response shapes, verdict vocabulary, threshold-from-config, and evidence_index semantics are unchanged; the model swap is a config-version bump by construction -->

## Impact

- `engine/scoring.py` — rewritten around `AutoModelForTokenClassification`; `pyproject.toml` — `hhem` extra replaced, transformers unpinned to ≥5; `config/verifier.yaml` — new model, revision, threshold, version bump.
- `tests/test_scoring.py`, `tests/test_hhem_model.py` — replaced with LettuceDetect equivalents; `Makefile` (`test-model`, `run`), `Dockerfile` — extra renamed, sizing comments updated.
- `charts/plumb/values.yaml` memory sizing and README model references re-derived for ~1.2 GB weights.
- evals/ is untouched (it keeps its own `lettucedetect` dependency for benchmarking).
