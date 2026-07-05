# ADR-0005: One engine, three surfaces

2026-07-05 · Status: Accepted

## Context

The prevailing open-source pattern for groundedness checking is a stitch:
one library for dev-time evaluation, another as a CI gate, a third for
production monitoring — each with its own judge, its own configuration,
and thresholds that drift independently. The score that gates a PR and the
score that gates a production answer are then different numbers from
different judges, and nobody can say whether passing CI predicts passing in
production. For a tool whose entire job is trustworthy scores, that
inconsistency is disqualifying.

## Decision

Plumb has one verification engine, deployed once, and every surface
consults it. The runtime guardrail API, the CI gate (GitHub Action, pytest
plugin), and the observability output are views on the same deployed
service: same endpoint, same versioned config, same thresholds, same
calibration. No surface computes metrics locally — the CI harness sends the
golden dataset to the deployed engine rather than reimplementing scoring in
the client.

## Consequences

- CI-to-runtime consistency is provable by construction: a threshold exists
  in exactly one place, so it cannot drift between surfaces.
- Client integrations stay thin — they marshal requests and interpret
  verdicts — which keeps them cheap to maintain and language-agnostic.
- CI needs a reachable engine, so Plumb's own pipeline stands one up
  (kind cluster in CI) instead of taking the shortcut of in-process
  scoring. That is real infrastructure cost, accepted deliberately.
- Config and thresholds must be versioned and injected, because a verdict
  is only reproducible if the config that produced it is identifiable —
  this is the same requirement verdict pinning makes of the KB snapshot.
- The engine's API contract becomes load-bearing for every surface at once;
  it is a protected zone and changes go through spec review.
