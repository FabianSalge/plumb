# Benchmark results

Raw per-run output lives in `results/*.json`; the harness is `bench/` (see
`bench/run.py` for the exact protocol and invocation). Numbers below are
rendered from those JSONs.

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
  scorer (`engine.scoring`, passages only — no question, exactly as `/v1/verify`
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
