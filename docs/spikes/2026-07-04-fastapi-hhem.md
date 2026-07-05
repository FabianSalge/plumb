# Spike: FastAPI + HHEM single-check endpoint (#5)

2026-07-04 · Spike A of the language decision (#7), companion to #6. Timeboxed;
code discarded per workflow — these notes are the artifact.

## What was built

A ~40-line FastAPI app serving `POST /check {"claim", "evidence"} → {"score"}`
with HHEM-2.1-open (`vectara/hallucination_evaluation_model`) loaded in-process
via transformers. Model loads once in the lifespan hook; the endpoint is a sync
`def` so the blocking inference runs in FastAPI's threadpool. Sanity check on
score direction: a supported claim scored 0.898, a contradicted one 0.013.

Stack: Python 3.12.1, FastAPI 0.139.0, uvicorn 0.50.0, transformers 4.57.6,
torch 2.12.1. Hardware: MacBook, Apple M4, 10 cores, 16 GB — fp32, CPU only,
default torch threads.

## Latency (single check, laptop CPU)

| Measurement | Result |
| --- | --- |
| One check, short pair (~15-word evidence), median of 30 | 31 ms (p95 32–37 ms) |
| One check, 500-word evidence, median of 30 | 285 ms (p95 ~320 ms) |
| Cold first request after startup | 41–67 ms |
| Process start → ready (warm model cache) | ~4 s (2.6 s imports + 1.3 s model load) |
| First-ever start | + ~420 MB model download from HF |
| Server RSS after load | ~610 MB |
| Weights on disk / venv on disk | 418 MB / 615 MB |

## Friction

- transformers 5.x breaks HHEM outright: the remote-code class trips
  `AttributeError: 'HHEMv2ForSequenceClassification' object has no attribute
  'all_tied_weights_keys'` at load. Pinning `transformers<5` (4.57.6) fixed it.
  The model's custom code tracks HF internals, so every transformers upgrade is
  a potential break.
- The model requires `trust_remote_code=True` — arbitrary Python pulled from
  the Hub at load time. For a self-hostable product we'd have to pin a revision
  hash or vendor the modeling code; unpinned remote code in a trust product is
  a non-starter.
- The remote code also fetches its tokenizer from a *second* Hub repo
  (`google/flan-t5-base`) at runtime, so a deployment silently depends on two
  upstream repos staying available and unchanged.
- Scoring goes through a nonstandard `model.predict([(premise, hypothesis)])`
  API bolted on by the remote code. Pair order is positional and unchecked —
  flipping (evidence, claim) still returns plausible-looking scores, which is
  exactly the kind of silent wrongness this product exists to prevent. Needs a
  test the moment real code exists.
- Even on 4.x, loading emits a config-mismatch warning ("model of type
  HHEMv2Config to instantiate a model of type HHEMv2") plus a load report
  flagging a MISSING (tied) weight — apparently benign, but it's noise we'd
  have to explain or suppress in structured logs.
- Async footgun: the endpoint must be sync `def` so FastAPI threadpools the
  blocking inference. Writing the natural `async def` instead would serialize
  the event loop with zero warnings — easy mistake for anyone touching this
  code later.
- To Python's credit: the whole thing was standing in well under an hour
  including debugging the version pin, and single-check CPU latency (31 ms
  short, ~285 ms at 500 words of evidence) is comfortably fine for both CI
  and interactive use.
- Deployment weight to keep in mind for the in-cluster story: ~610 MB resident
  per replica, ~1 GB of image (venv + weights), and the model either gets baked
  into the image or pulled from HF on first boot.
