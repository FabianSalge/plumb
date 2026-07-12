# Benchmark results

Raw per-run output lives in `results/*.json`; the harness is `bench/` (see
`bench/run.py` for the exact protocol and invocation). Numbers below are
rendered from those JSONs.

## Fast-mode p95 re-measured: the serve path exonerated (#59)

2026-07-12. #36 measured fast mode at 340 / 1,198 ms (p50 / p95) and read
the gap over scorer-only time (214 / 564 ms) as serve-path cost: ~125 ms at
the median, double at the tail. #59 re-measured under the identical
protocol — same seed-18 slice, same sequential client, same hardware, same
`bench/latency_run.py` — and the gap is gone:

| Metric | #36 (2026-07-11) | #59 (2026-07-12) | Contract |
| --- | --- | --- | --- |
| p50 | 340 ms | 216 ms | — |
| p95 | 1,198 ms | **568 ms** | sub-second p95 — **met** |

**Nothing in the code path changed; the measurement itself was the
defect.** Two lines of evidence:

- Per-stage instrumentation over the same 600-request slice puts the entire
  serve path at ~2 ms: HTTP plus client 0.6 ms, ASGI middleware 0.6 ms, and
  segmentation, calibration, gate, structured logging and response
  serialization each under 1 ms at p99. The model forward pass is 99% of
  wall time among the slowest 5% of requests, and in-server forward times
  (218 / 600 ms median / p95) match the offline scorer benchmark
  (214 / 564 ms) — the server does not slow the model down.
- Between the two measurements the serve path only gained work (#54 added
  span-confidence calibration to every response), so a code change cannot
  explain a tail that halved.

What can: machine state. The #36 run was taken with the kind cluster up
(PR #53 records port 8000 held by a stale port-forward) and no record of
what else the machine was doing. The protocol now stamps ambient load
averages, sampled before the first request, into every latency results
JSON — a contaminated run reads as one. This run's stamp: 3.3 / 4.0 / 4.0
(1 m / 5 m / 15 m; a Docker VM was busy on the host throughout) — adverse
conditions, and the contract is still met with margin.

### Forward-pass levers, measured and declined

The tail is now entirely the forward pass on long Data2txt and Summary
contexts. The cheap levers were measured on the eight longest-context slice
examples: attention already runs sdpa, and raising torch intra-op threads
from 4 to 10 buys ~6% on the longest sequences (Accelerate/AMX does the
matmul work regardless of the thread pool). Anything that would actually
move the tail — int8 dynamic quantization, `torch.compile` — changes
inference numerics and therefore demands an `INFERENCE_MODE` bump and a
calibration refit (ADR-0008): out of scope for #59, on the table if the
contract tightens or the hardware class drops.

No engine or API code changed, so the offline sentence benchmark numbers
(#45) are unchanged by construction.

### Re-measurement protocol

Identical to the #36 protocol below — server `make run`, sequential
`bench/latency_run.py` client on the same machine, 10 warmup requests
excluded, run aborts on identity drift or an empty claims list — plus the
ambient-load stamp above. Server engine 0.2.0, config 0.6.0 (#36 measured
config 0.5.0; the delta is #54's span-confidence work). MacBook, Apple M4,
16 GB, CPU only. Python 3.13.7.

## Span calibration: does the claim map transfer? (#40)

2026-07-12. ADR-0007 shipped spans with positions only; #40 asks whether the
claim-level Platt map (#32) is valid on span-level scores, or a span-level fit
is needed. The rule was fixed before fitting (change design §3): evaluate both
candidates on the held-out seed-18 slice's spans, ship the transferred claim
coefficients iff their span ECE is within 0.01 absolute of the span fit's.

**It does not transfer: the span-level fit ships.** Transfer ECE 0.0527 vs
fitted ECE 0.0370 — the 0.0157 gap exceeds the margin, so the artifact
(`config/calibration/lettucedect-v2-mmbert-base-sentence-v2.yaml`, schema 2)
carries span coefficients a = 0.736, b = 0.416 fitted at span level. One
honest caveat: the paired-bootstrap 95% CI on the ECE difference is
[−0.016, 0.046] — it straddles zero, so with only 422 held-out spans the
preference for the fit is the pre-registered point rule firing, not a
statistically decisive separation. Both candidates are recorded in the
artifact and below.

### Span protocol

- **Fit:** spans produced by the serve path itself (whole-answer joint pass,
  the engine's segmenter and reduction, flagging threshold 0.5) over the
  2,100 RAGTruth test responses outside the seed-18 slice → 1,282 spans,
  62.2% genuinely unsupported. A span is labeled unsupported iff it overlaps
  ≥1 character of a human-annotated hallucination span (any-overlap, the
  mirror of the #45 sentence convention); 351 of the 798 unsupported fit
  spans overlap only partially — the convention's fuzzy edge, published, not
  hidden. Fit-set SHA-256 is in the artifact.
- **Held-out evaluation:** the seed-18 slice's spans (600 responses →
  422 spans, 60.7% unsupported, 105 partial). Span confidence direction:
  P(the flagged region is genuinely unsupported), monotone increasing in the
  span's raw max token risk — raw span AUROC 0.731 (unchanged by the
  monotone map; the run aborts if it moves). Spans discriminate much more
  weakly than sentences (0.926) — a flagged span is a localization hint, not
  a second verdict.
- **Out-of-domain:** unmeasured and recorded as such in the artifact —
  LLM-AggreFact carries claim-level labels only; no span-annotated OOD
  dataset exists.
- **Hardware:** MacBook, Apple M4, CPU only. Python 3.13.7, torch 2.12.1,
  transformers 5.13.0 (`tf5`). `bench/span_calibration_run.py`.

### Span reliability

| Candidate | Held-out span ECE |
| --- | --- |
| Raw max token risk (uncalibrated) | 0.1236 |
| Transferred claim map (a = 0.829, b = 0.506) | 0.0527 |
| **Span-level fit (a = 0.736, b = 0.416) — ships** | **0.0370** |

Held-out reliability of the shipped span map (seed-18 spans, confidence =
P(unsupported), 10 equal-width bins; empty bins omitted):

| Bin | n | Mean confidence | Unsupported rate |
| --- | --- | --- | --- |
| [0.3, 0.4) | 6 | 0.399 | 0.500 |
| [0.4, 0.5) | 166 | 0.439 | 0.392 |
| [0.5, 0.6) | 50 | 0.548 | 0.600 |
| [0.6, 0.7) | 57 | 0.640 | 0.649 |
| [0.7, 0.8) | 56 | 0.749 | 0.750 |
| [0.8, 0.9) | 65 | 0.852 | 0.908 |
| [0.9, 1.0] | 22 | 0.924 | 0.909 |

The transferred claim map on the same spans, for comparison:

| Bin | n | Mean confidence | Unsupported rate |
| --- | --- | --- | --- |
| [0.3, 0.4) | 64 | 0.386 | 0.438 |
| [0.4, 0.5) | 112 | 0.443 | 0.357 |
| [0.5, 0.6) | 45 | 0.547 | 0.644 |
| [0.6, 0.7) | 52 | 0.641 | 0.615 |
| [0.7, 0.8) | 49 | 0.747 | 0.735 |
| [0.8, 0.9) | 65 | 0.853 | 0.892 |
| [0.9, 1.0] | 35 | 0.929 | 0.943 |

The bins are thin — the population is every span the engine flags across 600
responses, and that is 422. Per-bin counts stay published so nobody reads
three-decimal stability into a 22-span bin. Because the artifact binds the
flagging threshold (spans exist only where risk ≥ threshold), a threshold
change forces a span refit by construction.

## Fast-mode latency end to end through `/v1/verify` (#36, ADR-0003)

*Superseded by the #59 re-measurement above: the tail here was measurement
contamination, not serve-path cost.*

2026-07-11. ADR-0003 gives fast mode a sub-second p95 contract and calls its
numbers targets until measured. This measures it — wall-clock per request
against a running server, so HTTP, validation, the joint forward pass,
segmentation, calibration, gate, and structured logging are all inside the
clock, not scorer-only time.

### End-to-end protocol

- **Server:** `make run` (uvicorn, one worker, root project's `model` extra) —
  engine 0.2.0, config 0.5.0, the pinned LettuceDetect v2 weights.
- **Client:** `bench/latency_run.py`, a pure HTTP client on the same machine.
  Requests are sequential — this measures latency, not throughput — with 10
  warmup requests excluded. The run aborts if the server's identity drifts
  mid-run or a response carries no claims.
- **Input distribution:** the seed-18 RAGTruth test slice (200 per task type →
  600 responses), whole response as `text`, its source as the single context
  passage, `mode: "fast"`. Claims per response mean 7.1, max 22 — matching the
  offline sentence benchmark on the same slice, so the server segments exactly
  as the bench does.
- **Hardware:** MacBook, Apple M4, 16 GB, CPU only. Python 3.13.7.

### Measured vs contract

| Metric | Value | Contract |
| --- | --- | --- |
| p50 | 340 ms | — |
| p95 | **1,198 ms** | sub-second p95 — **missed** |

The contract is missed at the tail, by ~20%. Scorer-only per-response latency
on the identical slice is 214 / 564 ms (median / p95, below): the serve path
adds ~125 ms at the median and roughly doubles the tail, where the long
Data2txt and Summary contexts already make the forward pass expensive.
ADR-0003 says published numbers become a regression surface; this one starts
as a gap to close, not a pass to defend.

## Calibration v0, Platt scaling on held-out RAGTruth (#32, ADR-0008)

2026-07-11. The raw support score becomes a calibrated confidence: among
claims the engine scores c, about a fraction c should be fully supported.
This section measures how close the shipping calibrator gets — in-domain and,
honestly, out of it. The fitted artifact is
`config/calibration/lettucedect-v2-mmbert-base-sentence-v1.yaml`.

### Fit

- **Data:** every RAGTruth test response *outside* the seed-18 benchmark
  slice — 2,100 responses, excluded by the same `stratified_slice` call the
  benchmark runs, so nothing the calibrator was fitted on evaluates it.
  Scored exactly as `/v1/verify` scores (one whole-answer joint pass,
  the engine's segmenter and reduction) → 14,589 sentences, 91.8% supported
  (a sentence is supported iff it overlaps no annotated span, the #45
  convention). Fit-set SHA-256 is recorded in the artifact.
- **Method:** two-parameter Platt scaling on the ε-clamped logit of the raw
  support, plain MLE by Newton–Raphson (`bench/calibration.py`); the map
  applied is the engine's own `engine.calibration.platt_confidence`, so
  fit-time and serve-time arithmetic are the same code. Fitted
  a = 0.829, b = 0.506. Monotone, so AUROC is untouched — the run aborts
  if it moves.
- **Hardware:** MacBook, Apple M4, CPU only. Python 3.13.7, torch 2.12.1,
  transformers 5.13.0 (`tf5`). `bench/calibration_run.py`.

### Reliability

| Slice | Unit | n | ECE raw | ECE calibrated | AUROC |
| --- | --- | --- | --- | --- | --- |
| RAGTruth seed-18 (in-domain) | sentence | 4,240 | 0.0179 | **0.0074** | 0.926 |
| LLM-AggreFact ex-RAGTruth (out-of-domain) | claim | 1,000 | 0.1068 | **0.1531** | 0.847 |

In-domain reliability (seed-18 sentences, calibrated confidence, 10
equal-width bins; full diagram data in the results JSONs):

| Bin | n | Mean confidence | Supported rate |
| --- | --- | --- | --- |
| [0.0, 0.1) | 35 | 0.071 | 0.057 |
| [0.1, 0.2) | 62 | 0.148 | 0.113 |
| [0.2, 0.3) | 48 | 0.252 | 0.229 |
| [0.3, 0.4) | 50 | 0.360 | 0.380 |
| [0.4, 0.5) | 41 | 0.453 | 0.366 |
| [0.5, 0.6) | 55 | 0.550 | 0.745 |
| [0.6, 0.7) | 83 | 0.653 | 0.675 |
| [0.7, 0.8) | 128 | 0.750 | 0.750 |
| [0.8, 0.9) | 305 | 0.857 | 0.859 |
| [0.9, 1.0] | 3,433 | 0.978 | 0.981 |

Out-of-domain reliability (LLM-AggreFact claims, calibrated confidence):

| Bin | n | Mean confidence | Supported rate |
| --- | --- | --- | --- |
| [0.0, 0.1) | 13 | 0.070 | 0.077 |
| [0.1, 0.2) | 34 | 0.149 | 0.059 |
| [0.2, 0.3) | 69 | 0.257 | 0.087 |
| [0.3, 0.4) | 70 | 0.349 | 0.214 |
| [0.4, 0.5) | 58 | 0.453 | 0.310 |
| [0.5, 0.6) | 45 | 0.555 | 0.289 |
| [0.6, 0.7) | 48 | 0.651 | 0.312 |
| [0.7, 0.8) | 64 | 0.749 | 0.578 |
| [0.8, 0.9) | 107 | 0.856 | 0.645 |
| [0.9, 1.0] | 492 | 0.971 | 0.852 |

### Out-of-domain protocol

`bench/aggrefact_run.py` on `lytang/LLM-AggreFact` test (gated; evaluation
use only, per its terms): 100 claims per subset, seed 18 → 1,000 claims,
59.5% supported. The RAGTruth subset is excluded because the calibrator is
fitted on RAGTruth; every remaining subset is checked programmatically
against the pinned model's actual training mix (the `dataset` column of the
KR Labs unified training sets) at load time — no further overlap was found,
and the exclusion record ships inside the artifact. Claims are scored at
LLM-AggreFact's own claim granularity (one claim, one joint pass against its
document), not our sentence unit — a stated mismatch, not a hidden one. Long
documents (TofuEval meeting transcripts) truncate at the model window and
are logged as such.

### Reading the numbers

- **In-domain the map earns its place:** ECE more than halves, to 0.7% —
  the raw score was systematically pessimistic mid-range (raw 0.4–0.5
  scores were supported 72% of the time), which is exactly the sigmoid-shaped
  miscalibration Platt corrects. A verdict threshold of 0.5 on the calibrated
  confidence now means "more likely supported than not"; gate policy can be
  set against a probability instead of a model artifact.
- **Out-of-domain the calibrated number is *overconfident*, and worse than
  raw** (ECE 0.153 vs 0.107). The driver is base-rate shift: the fit
  distribution is 91.8% supported RAG traffic, LLM-AggreFact is 59.5%
  supported fact-checking-style claims, so the upward shift the map learned
  in-domain overshoots. The claim-unit mismatch contributes. This is the
  published answer to "what happens on my domain": on RAGTruth-style RAG
  traffic the confidence means what it says; the further your traffic is
  from that, the more it overstates support — the standing case for
  per-tenant refits (ADR-0008). Discrimination stays solid OOD (AUROC 0.847),
  so operators on shifted domains should raise the threshold, not trust the
  absolute number.
- The reliability diagrams show no systematic sigmoid misfit in-domain, so
  no evidence yet for a superseding ADR (isotonic/beta).

## Sentence-level discrimination, segment-after-score (#45, ADR-0009)

2026-07-11. The gate before whole-text-as-one-claim retires: sentence
decomposition must localize hallucinations, not just detect them at the
response level. Measures the shipping configuration — the engine's own
segmenter and question-less scorer, one whole-answer pass per response.

### Measurement

- **Data:** RAGTruth test split (`wandb/RAGTruth-processed`), same stratified
  200-per-task slice as the model decision above (seed 18 → 600 responses).
  Each response is segmented by `engine.decomposition.segment`; a sentence is
  labelled hallucinated iff its character range overlaps an annotated span
  (`hallucination_labels` offsets) → 4,240 sentences, 363 (8.6%) hallucinated.
- **Scoring:** one whole-answer forward pass per response through the shipping
  scorer (`engine.signals.groundedness`, passages only — no question, exactly as `/v1/verify`
  runs), reduced per sentence by `engine.decomposition.decompose`. Sentence risk
  = 1 − support. AUROC over sentence risk vs the overlap label
  (`bench/sentence_run.py`).
- **Hardware:** MacBook, Apple M4, CPU only. Python 3.13.7, torch 2.12.1,
  transformers 5.13.0 (`tf5`).

### Result

| Metric | Value | Floor |
| --- | --- | --- |
| Sentence-level AUROC | **0.926** | ≥ 0.70 |
| Sentences / response (mean, max) | 7.1, 22 | — |
| Per-response latency (median / p95) | 214 / 564 ms | sub-second |

**Floor:** 0.70 AUROC. Below it, per-sentence attribution would be too noisy to
localize which sentence broke, and decomposition would not earn its place over
the whole-text claim. The measured 0.926 clears it comfortably — and matches the
same model's response-level AUROC (0.891, above), so refining to sentences does
not cost discrimination. Latency stays one forward pass regardless of sentence
count (mean 7.1, max 22 per response), median well under the sub-second contract.

## Signal model decision, groundedness slot (#18, ADR-0006)

2026-07-05. Decides which model fills the groundedness slot fixed by
ADR-0004, replacing frozen-upstream HHEM-2.1-open
(`docs/spikes/2026-07-04-fastapi-hhem.md`).

### Protocol

- **Data:** RAGTruth test split via `wandb/RAGTruth-processed` (MIT,
  revision `eb4f4b9d`): 2,700 responses, 943 (34.9%) hallucinated.
  Response-level binary label: hallucinated iff the response carries at
  least one annotated span (evident conflict or baseless info).
- **Slice:** stratified 200 per task type (QA, Summary, Data2txt), seed 18
  → 600 examples, 215 (35.8%) hallucinated. CPU-heavy candidates ran on
  prefix subsets of the same shuffled frame (same seed, strict subset —
  see *n* per row).
- **Scoring:** each candidate produces a response-level support score in
  [0, 1] via its own documented protocol (`bench/adapters/`); hallucination
  is predicted iff support < 0.5, identically for all. AUROC is computed
  over risk = 1 − support. Before scoring, every adapter must rank a
  supported claim above a contradicted one or the run aborts (`sanity` in
  each JSON).
- **Latency:** median/p95 per response over the slice, plus the HHEM
  spike's fixed-pair protocol (~15-word and 500-word evidence, median of
  30 warm calls) for comparability with the spike.
- **Hardware:** MacBook, Apple M4, 10 cores, 16 GB, CPU only (fp32;
  Granite in bf16). Python 3.13.7, torch 2.12.1, transformers 4.57.6
  (`tf4` extra) / 5.13.0 (`tf5` extra — the two candidates' requirements
  conflict; see `pyproject.toml`).

### Accuracy

| Candidate | n | AUROC | Balanced acc | F1 (halluc.) | Precision | Recall |
| --- | --- | --- | --- | --- | --- | --- |
| hhem-2.1-open | 600 | 0.844 | 0.730 | 0.646 | 0.727 | 0.581 |
| lettucedetect-large-v1 | 600 | 0.909 | 0.827 | 0.776 | 0.762 | 0.791 |
| lettucedetect-v2-mmbert-base | 600 | 0.891 | 0.782 | 0.721 | 0.830 | 0.637 |
| minicheck-flan-t5-large | 150 | 0.839 | 0.671 | 0.625 | 0.472 | 0.926 |
| granite-guardian-3.2-3b-a800m | 75 | 0.724 | 0.506 | 0.125 | 0.400 | 0.074 |

### Latency and footprint

| Candidate | Short pair | 500-word pair | Slice median / p95 | Load | RSS after load | Weights on disk |
| --- | --- | --- | --- | --- | --- | --- |
| hhem-2.1-open | 22 ms | 209 ms | 227 / 798 ms | 2.2 s | 409 MB | 421 MB |
| lettucedetect-large-v1 | 92 ms | 588 ms | 549 / 1431 ms | 2.1 s | 542 MB | 1,513 MB |
| lettucedetect-v2-mmbert-base | 40 ms | 425 ms | 252 / 656 ms | 4.2 s | 897 MB | 1,207 MB |
| minicheck-flan-t5-large | 93 ms | 1326 ms | 6110 / 34707 ms | 2.8 s | 489 MB | 2,992 MB |
| granite-guardian-3.2-3b-a800m | 16752 ms | 38533 ms | 30644 / 52303 ms | 1.9 s | 484 MB | 6,297 MB |

HHEM's numbers reproduce the spike (22 vs 31 ms short pair, 209 vs 285 ms
at 500 words — spike numbers included cold-ish process state).

### Operational bar (issue #18 acceptance criteria)

| Candidate | Pinned revision | No `trust_remote_code` | Single Hub repo | transformers | Upstream status (July 2026) |
| --- | --- | --- | --- | --- | --- |
| hhem-2.1-open (Apache-2.0) | yes | **no** | **no — fetches `google/flan-t5-base` at runtime** | **frozen `<5`** | frozen mid-2024, successors API-only |
| lettucedetect-large-v1 (MIT) | yes | yes | yes | 4.x and 5.x, both verified | model Apr 2025; package actively maintained |
| lettucedetect-v2-mmbert-base (Apache-2.0) | yes | yes | yes | 5.x only (tokenizer needs `TokenizersBackend`) | released June 2026, active line |
| minicheck-flan-t5-large (MIT) | yes | yes | yes | 4.x and 5.x | frozen Dec 2024 |
| granite-guardian-3.2-3b-a800m (Apache-2.0) | yes | yes | yes | 4.x and 5.x | family active (4.1-8B is current) |

### Not run locally

- **Bespoke-MiniCheck-7B** — CC-BY-NC 4.0 with commercial licensing by
  negotiation; a non-starter for a self-hostable product regardless of
  accuracy (its published LLM-AggreFact results lead the open-detector
  field). Also an 8B model served via vLLM on GPU.
- **Granite Guardian 4.1-8B** — the family's current release is an 8B
  generative judge (~16 GB bf16); 3.2-3B-A800M above is the family's best
  CPU case and already misses the latency bar by two orders of magnitude.

### Caveats

- RAGTruth is in-domain for LettuceDetect: v1 was fine-tuned on exactly
  its training split, v2 on a broader mix that includes it. HHEM's
  training data is not fully documented; MiniCheck and Granite are
  out-of-domain here (MiniCheck's published out-of-domain results are
  near-GPT-4 for its size). The accuracy table therefore overstates the
  LettuceDetect gap on unseen domains; the operational columns carry the
  decision weight.
- Granite's slice latencies include heavy swapping — a 3B bf16 model plus
  long summarisation prompts exceed a 16 GB machine. That is the realistic
  behaviour at this memory size, not a measurement artefact. Its near-zero
  recall is the model's own Yes/No reading at face value; the AUROC shows
  the underlying score carries some signal at other thresholds.
- Single machine, single run; medians are stable, p95s indicative only.
- MiniCheck and Granite subsets (n = 150 / 75) widen their confidence
  intervals; both are far enough from the frontier that this does not
  affect the decision.
