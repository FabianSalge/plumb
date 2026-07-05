# ADR-0001: API language — Python, single process

2026-07-05 · Status: Accepted

## Context

Plumb's engine is a verification pipeline wrapped in an HTTP API. The first
irreversible-feeling choice was the language of that API, with two serious
candidates: Python end to end, or a Go API fronting a Python inference
sidecar. The question was settled by two timeboxed spikes —
[FastAPI + HHEM in-process](../spikes/2026-07-04-fastapi-hhem.md) (#5) and
[Go front + Python sidecar](../spikes/2026-07-05-go-api-python-sidecar.md)
(#6) — judged on four criteria: iteration speed, architectural honesty,
defensibility, and ecosystem pull.

The spikes converged:

- The split buys nothing at the bottleneck. The localhost hop costs ~0–1 ms
  against 20–140 ms of inference, and the ~500 MB model process exists in
  both architectures. Go's genuine virtues (0.75 s builds, 12 MB RSS,
  instant start) sit next to a model process that dwarfs them either way.
- The split costs a permanent seam. The request contract existed twice — Go
  structs and Pydantic models — with no compiler or test spanning it, and
  every future signal (NLI, self-consistency, judge calls) would add another
  duplicated internal schema. Two processes also meant two lifecycles:
  readiness gating, port contracts, and dev/CI babysitting that the
  single-process spike simply didn't have.
- The Go front held zero product logic — marshal, forward, unmarshal, map
  errors. Everything on the engine roadmap (decomposition, retrieval,
  signals, aggregation, calibration) is model-adjacent and lands on the
  Python side regardless.
- Ecosystem pull is one-directional: transformers, eval harnesses, and
  benchmark loaders are Python whichever language serves HTTP.

## Decision

The API and engine are Python — one process, one language. Target current
Python (3.13+) and FastAPI, with models loaded once at startup and blocking
inference kept off the event loop (sync endpoints running in the threadpool,
as validated in the spikes).

## Consequences

- One codebase, one lifecycle, one contract definition. Internal pipeline
  stages are function calls, not HTTP hops with duplicated schemas.
- The ML ecosystem is in-process: signal models, benchmark loaders, and the
  eval harness share the runtime.
- We inherit Python's serving footguns. The async trap is the sharpest —
  declaring an inference endpoint `async def` silently serializes the event
  loop — so blocking-inference handling is a convention the codebase must
  enforce with tests, not vigilance.
- We accept slower cold starts (seconds, dominated by imports and model
  load) and a resident-set floor set by the models; both are properties of
  serving the models at all, not of the language.
- If a hardened edge is ever warranted, a Go or Envoy front can be added in
  front of the stable HTTP contract later. That would be an additive
  deployment decision, not a reopening of this one.
