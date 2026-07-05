# ADR-0006: Groundedness slot — LettuceDetect replaces HHEM-2.1-open

2026-07-05 · Status: Accepted

## Context

ADR-0004 fixed the shape of the signal stack and left the groundedness
slot's occupant to this decision. The HHEM spike
(../spikes/2026-07-04-fastapi-hhem.md) showed the incumbent,
HHEM-2.1-open, to be a dead end as a long-term dependency: frozen
upstream since mid-2024 with successors behind Vectara's API, loadable
only with `trust_remote_code`, dependent on a second Hub repo fetched at
runtime, and permanently pinned to `transformers<5`.

Issue #18 set the bar: candidates compared on a common RAGTruth slice
plus the spike's laptop-CPU protocol, and a winner that loads from one
pinned revision with no remote code. Evidence in `evals/RESULTS.md`
(harness under `evals/bench/`). Options considered:

- **HHEM-2.1-open, pinned and vendored** — the do-nothing baseline.
  AUROC 0.844 on the slice; fails the operational bar structurally, and
  vendoring buys a frozen model on a frozen 4.x transformers stack.
- **LettuceDetect v2 mmBERT-base (Apache-2.0) and v1 large (MIT)** —
  token-classification encoders (0.3B / 0.4B) from the one actively
  maintained open detector line (KR Labs; v2 released June 2026). Plain
  `AutoModelForTokenClassification`, single repo, pinned revision, both
  verified on transformers 5.13 (v2 requires ≥5; v1 also runs on 4.57).
  Top accuracy on the slice — v1 large AUROC 0.909, v2 mmBERT-base
  0.891 — with span-level output. RAGTruth is in-domain for both (v1 was
  trained on exactly its training split, v2 on a broader mix including
  it), so the gap over out-of-domain candidates overstates.
- **MiniCheck-Flan-T5-Large (MIT)** — respectable out-of-domain detector
  (AUROC 0.839 here), but ~6 s median per response on laptop CPU, ~3 GB
  fp32 weights, and frozen upstream since Dec 2024. The stronger
  Bespoke-MiniCheck-7B is CC-BY-NC — disqualified on license for a
  self-hostable product before benchmarking.
- **Granite Guardian 3.2 3B-A800M (Apache-2.0)** — the smallest
  groundedness-capable variant of an actively maintained family of
  generative judges. On a 16 GB CPU box it swaps and needs ~31 s median
  per response — two orders of magnitude off the bar — and its Yes/No
  reading barely flags on this data (recall 0.07). The LLM-judge shape
  belongs in ADR-0004's self-consistency slot, not this one.

## Decision

LettuceDetect fills the groundedness slot: `lettucedect-v2-mmbert-base`
becomes the default model, pinned by revision in versioned config, with
`lettucedect-large-modernbert-en-v1` as a config-swap alternative for
English-only deployments that want its higher in-domain accuracy. v2 is
half the latency (252 ms vs 549 ms median per response), smaller,
multilingual, Apache-2.0, and the line upstream actively develops.

## Consequences

- The operational objections that forced this decision disappear: no
  remote code, one Hub repo per model, pinned revisions, and the engine
  moves to current transformers — the `<5` pin leaves with HHEM.
- The engine's scoring wrapper changes shape: HHEM's nonstandard
  `predict()` gives way to token classification with LettuceDetect's
  prompt format, and the signal gains span-level detail that claim
  decomposition can surface in observability output.
- The tracer bullet (#8) ships on HHEM as-is; the wrapper migration,
  config change, and retirement of the `hhem` extra are follow-up work
  tracked in #27.
- Accuracy on unseen tenant domains is a bet informed by an in-domain
  benchmark; the evals harness exists so the slot gets re-benchmarked
  when the field moves, and the product's accuracy claim continues to
  rest on aggregation and calibration, not on any one detector.
- We depend on KR Labs' maintenance cadence. If it stops, the pinned
  model remains serveable indefinitely with no remote code — strictly
  better than the incumbent's failure mode.
