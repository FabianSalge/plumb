# Tasks: LettuceDetect v2 migration

## 1. Packaging and config

- [x] 1.1 Replace the `hhem` extra with `model` (`transformers>=5.13`, `torch>=2.12`) in pyproject.toml, drop the `<5` pin and its comment, update the pytest `model` marker description, and re-lock
- [x] 1.2 Point config/verifier.yaml at `KRLabsOrg/lettucedect-v2-mmbert-base` @ `0f85c7a15b17aee6e8f794dae7cb4e42e2b8fdac`, threshold 0.5, bump `version` to 0.2.0, and document the v1-large swap in a comment

## 2. Scoring wrapper (tests first)

- [x] 2.1 Write failing unit tests for the vendored protocol: golden prompt string, claim-in-answer-slot ordering, support = 1 − max token probability, one score per passage in order, out-of-range score raises `ScorerError`, missing-dependency load error names the `model` extra
- [x] 2.2 Write failing unit tests for span extraction: contiguous flagged tokens become character spans (start/end/text/confidence) emitted on the structured logger; no span fields in the verify response
- [x] 2.3 Rewrite engine/scoring.py: `LettuceDetectScorer` with vendored prompt template, pair tokenization (claim never truncated, truncation logged), per-passage inference, support reduction, span extraction to structured logs; delete `HHEMScorer` and `evidence_claim_pairs`; update api/app.py factory default
- [x] 2.4 Retarget tests/test_hhem_model.py → tests/test_model.py: same direction check (supported ≥ threshold > contradicted) through the real v2 weights via `make test-model`

## 3. Build and deploy surfaces

- [x] 3.1 Update Makefile (`test-model`, `run` use `--extra model`; weight-size comments) and Dockerfile (`--extra model`, download-at-start comment: ~1.2 GB, one repo, no remote code)
- [x] 3.2 Measure RSS with v2 loaded and re-derive the memory sizing comments in charts/plumb/values.yaml

## 4. Docs and verification

- [x] 4.1 Update README: model name, download size, `make test-model` description, network-policy egress note (still one Hub fetch)
- [x] 4.2 Run `make test`, `make lint`, `make typecheck`, `make test-model` (real weights, caffeinated) and record results for the PR description
- [x] 4.3 `make deploy` + `make e2e` against kind to confirm the chart serves the new model end to end
