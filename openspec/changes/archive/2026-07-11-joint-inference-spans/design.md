# Design — joint inference and span attribution

## Context

ADR-0007 settled the tier-2 response shape; this change implements it. Today
`engine/scoring.py` renders one passage per prompt (`passage 1: <text>` into the vendored
summary template), runs one forward pass per passage, returns `list[float]`, and logs
spans as structured detail only. `engine/verdict.py` picks the best-scoring passage to
produce `evidence_index`, which `api/schemas.py` promises in the response.

The benchmarked configuration (`evals/RESULTS.md`) ran through lettucedetect 0.2.1's own
`HallucinationDetector`, whose question-less rendering is: all passages as
`passage {i+1}: {text}` lines joined by `\n`, substituted into the same summary template
the engine already vendors (`Summarize the following text:\n{context}\noutput:`). The
tokenization layout (pair encoding, `only_first` truncation, claim in the answer slot) is
unchanged between the vendored 0.1.11 protocol and 0.2.1 — only the context rendering
generalizes to multiple passages.

## Goals / Non-Goals

**Goals:**

- One forward pass per claim; score = support by the union of passages.
- Byte-for-byte match with lettucedetect 0.2.1's multi-passage rendering, pinned by a
  golden regression test.
- Spans (positions + text, no confidence) replace `evidence_index` in the response.
- Span-flagging threshold in versioned config; `config_version` bump.

**Non-Goals:**

- Span confidences in the response (blocked on #32).
- Retrieval provenance (returns with the retrieval ADR).
- Passage-level chunking when the window overflows (lettucedetect 0.2.1 chunks and
  max-aggregates; ADR-0007 keeps truncate-and-log — window budgeting becomes retrieval's
  problem).
- Claim decomposition (#35); the whole `text` stays one claim.

## Decisions

- **Scorer returns a result object, not a float list.** `score(claim, passages)` returns a
  frozen `ClaimScore(support: float, spans: list[Span])`. Spans are now part of the
  scorer's contract (the API renders them), so returning them beats re-deriving them at
  the API layer or smuggling them through logs. The `Scorer` protocol changes with it.
- **Internal spans keep `confidence`; the API shape drops it.** Structured logs keep the
  raw confidences for observability (today's behavior), while the response schema carries
  only `start`, `end`, `text` per ADR-0007. The boundary is `api/schemas.py`, not the
  engine.
- **`render_prompt` takes `list[str]`.** Enumerated `passage {i+1}: {text}` lines joined
  by `\n` into the existing template — exactly lettucedetect 0.2.1's
  `PromptUtils.format_context(context, question=None, lang="en")`. The golden test pins a
  two-passage rendering alongside the existing shape.
- **`judge_claim` narrows to verdict mapping.** It takes the single union-support score
  and the verdict threshold, returns `ClaimVerdict(text, verdict, score)` — no best-index
  selection left to do. Spans attach to the response at the API layer from the scorer
  result. Gate semantics (`pass` iff every claim `supported`) untouched.
- **Span threshold joins `SignalModelConfig`.** `span_threshold` sits next to `threshold`
  in `config/verifier.yaml` — per-model, like the verdict threshold, because it is
  expressed on the same model's token probabilities. Config `version` bumps 0.2.0 → 0.3.0.
  The module constant `_FLAG_THRESHOLD` dies. The two knobs are independent: an
  `unsupported` claim with zero spans is legal, and the docs say so.
- **Offsets are Unicode code points.** Tokenizer offsets are Python `str` indices, which
  are code points already; the schema documents the unit so non-Python callers slice
  correctly.

## Risks / Trade-offs

- [Joint rendering shifts absolute scores; the pinned verdict threshold (0.5) was
  benchmarked in this joint mode, but per-passage serving scores drift from it]
  → Acceptable by construction: the benchmark that chose the model/threshold ran joint
  inference, so this change moves serving *onto* the measured configuration; #34/#32
  calibrate on top of it.
- [Truncation now affects all passages at once, not one at a time] → Kept loud: the
  truncation log gains the total passage count; budgeting is deferred to retrieval per
  ADR-0007.
- [Rendered-format drift against future lettucedetect versions] → Golden regression test
  fails on any drift; re-verify against the package version that trained the pinned model
  when the revision bumps (existing rule, docstring updated to 0.2.1).

## Migration Plan

Single breaking API change, pre-1.0, no deprecation window (ADR-0007): `evidence_index`
out, `spans` in, `config_version` bumped in the same release. The e2e smoke test and chart
golden request update in the same PR.
