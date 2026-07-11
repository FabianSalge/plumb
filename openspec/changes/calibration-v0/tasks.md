# Tasks — calibration v0

Checks come first throughout: each implementation task is preceded by its failing tests.
`engine/calibration/` and the threshold semantics are protected zones — this change is
the plan; nothing there moves outside it.

## 1. Reliability metrics (evals)

- [x] 1.1 Failing unit tests for ECE and reliability-bin data in `evals/tests`: 10
      equal-width bins; per-bin mean confidence, empirical support rate, count; ECE as
      the count-weighted mean absolute gap; refuses empty input per the `MetricError`
      convention
- [x] 1.2 Implement `ece` and `reliability_bins` in `evals/bench/metrics.py`

## 2. Platt fit (evals)

- [ ] 2.1 Failing unit tests for the two-parameter logistic MLE: recovers known
      coefficients on synthetic data, applies the ε-clamped logit, refuses single-class
      labels
- [ ] 2.2 Implement the fit (Newton–Raphson on log-loss) in `evals/bench/calibration.py`
- [ ] 2.3 Fit script `evals/bench/calibration_run.py`: load RAGTruth test, exclude the
      seed-18 slice by the same `stratified_slice` call the benchmark uses, score the
      remaining responses through the shipping scorer, label sentences by span overlap,
      fit, and emit the artifact YAML (schema, method, coefficients, bindings, fit-set
      hash, metrics) plus a results JSON

## 3. Out-of-domain evaluation (evals)

- [ ] 3.1 LLM-AggreFact loader in `evals/bench/`: RAGTruth subset excluded, remaining
      subsets checked against the pinned model card's documented training mix, the
      check's outcome recorded in the results JSON; loud failure on split-stat mismatch
      per the `data.py` convention
- [ ] 3.2 OOD run: score the LLM-AggreFact slice through the shipping scorer (each claim
      as a single unit), apply the fitted calibrator, emit ECE + reliability data

## 4. Fit run, artifact, RESULTS.md

- [ ] 4.1 Run the fit end to end (caffeinate; ~2,100 responses on the M4), commit the
      artifact under `config/calibration/` and the results JSONs under `evals/results/`
- [ ] 4.2 In-domain validation: ECE + reliability diagram of the fitted calibrator on
      the seed-18 slice's sentences
- [ ] 4.3 `evals/RESULTS.md` calibration section: protocol, in-domain and out-of-domain
      ECE, reliability-diagram tables, the granularity caveat for LLM-AggreFact, and the
      threshold picked against the reliability data

## 5. Engine calibration module

- [ ] 5.1 Failing engine tests: artifact loading (complete artifact loads; missing
      field / unknown schema fails naming the field), binding validation (mismatch fails
      naming every mismatched field with expected vs found; missing reference is a
      startup failure, no raw fallback), Platt application (strictly monotone, ε-clamp,
      output strictly inside (0, 1))
- [ ] 5.2 Implement `engine/calibration/`: artifact model + loader, binding validation,
      `PlattCalibrator.apply`
- [ ] 5.3 Declare protocol identifiers — inference mode in `engine/scoring.py`, claim
      unit in `engine/decomposition.py` — and validate artifact bindings against them
- [ ] 5.4 Config: `calibration` artifact path in `SignalModelConfig` (required —
      `ConfigError` when absent); `config/verifier.yaml` references the committed
      artifact, carries the re-picked confidence threshold, and bumps `version`

## 6. API surface

- [ ] 6.1 Failing API tests: claims carry `confidence` (calibrated, strictly inside
      (0, 1)) and no `score`; verdict thresholds the confidence; structured logs carry
      raw support alongside calibrated confidence; app startup fails on a mismatched or
      missing artifact
- [ ] 6.2 Wire it: load + validate the calibrator in `create_app` lifespan, apply per
      claim before `judge_claim`, rename the response field in `api/schemas.py`, log raw
      + calibrated
- [ ] 6.3 Ship the artifact with the config: chart configmap and image include
      `config/calibration/`; `make deploy` + `make e2e` green against the new response
      shape

## 7. Docs and verification

- [ ] 7.1 Document the confidence docs-plain (per ADR-0008) wherever the API response is
      described — README and any response-shape docs — in the same PR
- [ ] 7.2 `make test`, `make lint`, `make typecheck` clean; `make test-model` for the
      scorer-touching paths; OpenSpec change validated; PR describes how each acceptance
      criterion of #32 was verified
