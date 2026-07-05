# Spike: Go API + Python HHEM sidecar (#6)

2026-07-05 · Spike B of the language decision (#7), companion to #5. Timeboxed;
code discarded per workflow — these notes are the artifact.

## What was built

The same external contract as #5 (`POST /check {"claim", "evidence"} →
{"score"}`), split into two processes: a ~100-line Go stdlib HTTP front that
validates the request, forwards it to a ~35-line FastAPI sidecar over
localhost, and maps sidecar failures to caller-facing errors (sidecar down →
502 with context, timeout → 504, bad JSON → 400). The sidecar is the #5 app
minus everything but the model: HHEM loaded in the lifespan hook, sync `def`
endpoint, same `transformers<5` pin. Score direction verified through the
full stack: supported claim 0.811, contradicted 0.008.

Stack: Go 1.26.4 (stdlib only, no modules), sidecar identical to #5
(Python 3.12.1, FastAPI 0.139.0, uvicorn 0.50.0, transformers 4.57.6,
torch 2.12.1). Same hardware: MacBook, Apple M4, 10 cores, 16 GB, fp32 CPU.

## Numbers (single check, laptop CPU, median of 30)

| Measurement | Sidecar direct | Through Go front |
| --- | --- | --- |
| Short pair (~10-word evidence) | 20.7 ms (p95 21.4) | 20.5 ms (p95 20.9) |
| 500-word evidence | 141.7 ms (p95 143.2) | 142.8 ms (p95 146.5) |

The localhost hop costs ~0–1 ms — noise against 20–140 ms of inference.
(Absolute latencies are lower than #5's table because the test pairs differ;
only the direct-vs-through-Go comparison on identical payloads is meaningful.)

Other measurements: Go build 0.75 s cold into a 9.1 MB static binary with zero
dependencies; Go front ready in well under a second vs 5.2 s for the sidecar
(imports + model load + first inference); RSS 12 MB for the Go front vs 484 MB
for the sidecar. Concurrency was not tested in either spike.

## Friction

- The contract now exists twice: Go structs in the front, Pydantic models in
  the sidecar. Every field change touches two codebases in two languages with
  no compiler or test spanning the seam. This is the finding that generalizes
  worst: each future signal (NLI, self-consistency, judge calls) adds another
  internal endpoint whose schema is maintained twice, and decomposition →
  retrieval → signals → aggregation would either all live in the sidecar
  (making the Go front pure plumbing) or turn every internal call into a
  duplicated-schema HTTP hop.
- Two processes means two lifecycles. The front is "up" and returning 502s
  for ~5 s while the sidecar loads the model; a real deployment needs
  readiness gating across the pair (doable as two containers in one pod, but
  it's plumbing #5 simply doesn't have). Local dev and CI likewise babysit
  two processes plus a port contract — the spike immediately hit a port
  collision with an unrelated local service, exactly the class of ops noise
  the split buys.
- The Go front contains zero product logic — marshal, forward, unmarshal,
  map errors. Everything on the engine roadmap is model-adjacent and would
  live on the Python side regardless, so Go's costs are permanent while its
  contribution stays a pass-through.
- To Go's credit: the front was pleasant to build and honest about failure.
  Explicit error mapping, a client timeout, and method routing are all
  stdlib one-liners; the binary builds in under a second, starts instantly,
  and idles at 12 MB. None of that was hard to get. It just doesn't touch
  the actual bottleneck — inference latency and the 484 MB model process —
  which is identical in both architectures.
- Plumbing did not eat the timebox: the whole stack stood in well under an
  hour. The overhead is not build cost or latency; it is owning two
  languages, duplicated contracts, and a distributed-system seam through the
  middle of a single-purpose service, forever.

## Recommendation

Python, single process. The spike removed the best argument for the split —
the hop is free, performance-wise — and left only its costs: HHEM (and every
candidate successor, NLI model, and judge) cannot run natively in Go, so a Go
API is condemned to be a thin proxy in front of the Python process that does
all the work, with every contract written twice. Go's genuine virtues here
(instant start, 12 MB RSS, static binary) are marginal next to a 484 MB model
process that exists either way, and the ecosystem the roadmap leans on —
transformers, eval harnesses, benchmark loaders — is Python. If a hardened
edge is ever needed, a Go or Envoy front can be added in front of a stable
HTTP contract later without reopening this decision. Iteration speed in #5,
architectural honesty here, and ecosystem pull all point the same way;
ADR-0001 should record Python and close the question.
