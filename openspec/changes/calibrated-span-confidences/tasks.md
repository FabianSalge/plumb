# Tasks — calibrated span confidences (issue #40)

## 1. Span-level evidence (evals)

- [x] 1.1 Failing tests in `evals/tests`: span-label derivation — engine spans labeled unsupported iff overlapping ≥1 character of an annotated span; partial-overlap counting; seed-18 slice excluded from the fit population by the same `stratified_slice` construction
- [x] 1.2 Implement span extraction + labeling in `evals/bench`: score responses exactly as the serve path, derive spans at the configured flagging threshold, label against RAGTruth span annotations
- [x] 1.3 Failing tests: span calibration fit and transfer evaluation — Platt fit on (1−r, label) via the engine's `platt_confidence`, held-out span ECE for both transferred claim coefficients and the span fit, the 0.01 pre-registered decision rule, bootstrap CI
- [x] 1.4 Implement the span calibration run (`bench/`): fit on non-seed-18 spans, evaluate both candidates on seed-18 spans, emit reliability tables, ECE, CI, partial-overlap rate, and the selected coefficients
- [x] 1.5 Run the span calibration end to end (caffeinate, tf5 extra) and write the span section of `evals/RESULTS.md`: both candidates' held-out ECE, reliability tables with per-bin counts, which rule fired, and the explicit statement that span-level out-of-domain error is unmeasured and why
- [x] 1.6 Emit the schema-2 artifact file under `config/calibration/` carrying the claim calibration unchanged plus the span section (coefficients, threshold, provenance, metrics)

## 2. Engine: artifact schema 2 and the span map

- [x] 2.1 Failing tests: span confidence arithmetic — `1 − platt_confidence(1 − r)` with span coefficients, monotone increasing in r, strictly inside (0, 1) at r ∈ {0.0, 1.0}, error on non-finite coefficients or r outside [0, 1]
- [x] 2.2 Failing tests: artifact loading — schema-2 artifact with complete span section loads; schema-1 artifact refused naming found and served versions; missing span-section fields refused naming the field
- [x] 2.3 Failing tests: binding validation — span-threshold mismatch between artifact and running config fails startup naming the field with expected and found values, alongside the existing four bindings
- [x] 2.4 Implement in `engine/calibration/`: span section models, `KNOWN_SCHEMAS = {2}`, span confidence method on the artifact, span-threshold binding validation

## 3. Engine: reduction carries raw risk, not "confidence"

- [x] 3.1 Failing tests: `Span.raw_risk` rename — structured span logs carry the raw maximum token risk under the raw name
- [x] 3.2 Rename `reduction.Span.confidence` to `raw_risk` and update the span log line

## 4. API: span confidence in the response

- [x] 4.1 Failing tests: response spans carry calibrated `confidence` strictly inside (0, 1); raw risk absent from the response body; unsupported-claim-with-spans scenario includes per-span confidence
- [x] 4.2 Implement: `api/schemas.py` span shape, `api/app.py` applies the artifact's span map to each span's raw risk, structured log carries raw risk and calibrated confidence side by side
- [x] 4.3 Reference the new artifact from the verifier config with a config-version bump

## 5. Docs

- [x] 5.1 Update README/docs: span confidences are in the response and what the number means (docs-plain claim from the design), log-only statements removed
- [x] 5.2 Update the OpenAPI/API examples if they show span objects

## 6. Verify and close

- [x] 6.1 `make test`, `make lint`, `make typecheck` green locally; `make test-model` for the scorer-touching paths
- [x] 6.2 `make e2e` against the chart in kind with the new config + artifact
- [ ] 6.3 OpenSpec validate the change, sync delta specs, archive after merge per workflow
