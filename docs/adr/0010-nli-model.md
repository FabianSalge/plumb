# ADR-0010: NLI slot — deberta-base-long-nli brings contradicted within reach

2026-07-12 · Status: Accepted

## Context

ADR-0004 fixed the signal stack's shape and left each slot's occupant to a
benchmarked decision. ADR-0006 filled the groundedness slot and set the
operational bar: maintained upstream, permissive license, no remote code,
single Hub repo, pinned revision, laptop-CPU latency. The NLI slot is next
because `contradicted` verdicts — promised in the README, deliberately
absent from tiers 1 and 2 — need a signal that can tell *refuted by the
evidence* apart from *merely unsupported*, a distinction the groundedness
cross-encoder does not make.

Issue #60 set the evidence: candidates scored sentence-by-sentence on the
seed-18 RAGTruth slice, labelled conflict/baseless/supported from the
corpus's own annotation types, under the established laptop-CPU protocol.
Full numbers in `evals/RESULTS.md`. Options considered:

- **tasksource/deberta-base-long-nli (Apache-2.0, 184M, 1,680-token
  window)** — best contradiction AUROC (0.693), highest contradicted-verdict
  precision (0.679) at the lowest false-contradiction rate on supported
  sentences (6.7%), second-fastest per sentence (241 ms median), 1.6%
  premise truncation. Trained on tasksource's multi-task NLI mix including
  doc-level premises — which is what this slot feeds it.
- **tasksource/ModernBERT-base-nli and -large-nli (Apache-2.0, 150M/395M,
  8k window)** — never truncate and detect hallucination well (0.825/0.832),
  but both trail on the headline contradiction discrimination (0.609/0.651);
  large costs 2.6× the DeBERTa's latency to lose to it. Scale bought
  nothing; training mix decided this round.
- **MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli and
  mDeBERTa-v3-base-mnli-xnli (MIT, 512-token window)** — the field-standard
  baselines. Both truncate the majority of RAG-length premises (61%/73%)
  and their hallucination detection collapses (0.665/0.614); both are also
  the slowest here, and the author's NLI line has been dormant since early
  2025. The window, not the head, disqualifies them for RAG evidence.
- **dleemiller/finecat-nli-l** — the newest checkpoint in the field (July
  2026) ships without a license; disqualified before benchmarking, as
  Bespoke-MiniCheck was in ADR-0006.

## Decision

tasksource/deberta-base-long-nli fills the NLI entailment slot, pinned at
revision `04dcf11f844b07bc57015169fca2b7d6df8299d5` in versioned config.
Integration into the engine — signal interface, config, and how the
`contradicted` vocabulary enters verdicts — is #61's work; this ADR fixes
the occupant.

## Consequences

- The `contradicted` verdict becomes buildable: at argmax the chosen model
  accuses conservatively (precision 0.679, recall 0.283, 6.7% false alarms
  on supported sentences), which is the right failure direction for a
  verdict that tells a tenant their model contradicted their documents.
- The best available separation of refuted from unsupported is modest
  (AUROC 0.693 against ~0.9 on curated NLI benchmarks). #61 must treat
  P(contradiction) as evidence for a conservative verdict, not a calibrated
  probability; the accuracy claim continues to rest on aggregation and
  calibration (ADR-0004), and RESULTS.md publishes the class counts that
  bound confidence in these numbers.
- The slot pays one forward pass per sentence, unlike the groundedness
  signal's single whole-answer pass: ~1.4 s median per response on laptop
  CPU. That prices the NLI signal into thorough mode and keeps it out of
  fast mode's sub-second contract, exactly as ADR-0003 anticipated.
- Contradicted verdicts start English-first: the only multilingual
  candidate failed on its 512-token window, while the default groundedness
  model is multilingual. A multilingual NLI model clearing the bar is a
  re-benchmark trigger, as is finecat-nli-l gaining a license.
- We add a second tasksource dependency (groundedness is KR Labs). Same
  mitigation as ADR-0006: the pinned revision serves indefinitely with no
  remote code if upstream goes quiet.
