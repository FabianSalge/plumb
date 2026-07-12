"""Run one NLI candidate over the RAGTruth slice sentences and write a results JSON.

Usage (from evals/):
    uv run --extra tf5 python -m bench.nli_run --candidate modernbert-base-nli \
        --per-task 200 --out results/nli-modernbert-base.json

The unit is the engine-segmented sentence (issue #60): each sentence scores as
one (premise=context, hypothesis=sentence) pass, labelled conflict / baseless /
supported from the annotated span kinds. The headline metric is whether
P(contradiction) separates refuted from merely unsupported; hallucination AUROC
(risk = 1 − P(entailment)) keeps the number comparable with the groundedness
sentence benchmark. Truncation share reports how much evidence each candidate's
window actually sees.
"""

import argparse
import json
import statistics
import time
from pathlib import Path

import psutil

from bench.adapters.nli import CANDIDATES, NliCandidate, load
from bench.data import load_ragtruth_test, stratified_slice
from bench.harness import (
    LATENCY_REPEATS,
    LONG_CLAIM,
    LONG_EVIDENCE,
    SHORT_CLAIM,
    SHORT_EVIDENCE,
    environment,
    percentile,
    progress,
    weights_on_disk_bytes,
    write_results,
)
from bench.metrics import auroc, precision_recall
from bench.nli import (
    SUPPORTED,
    contradiction_pairs,
    hallucination_pairs,
    nli_sentence_classes,
    predicted_contradicted,
)

SANITY_PREMISE = (
    "Paris is the capital of France. The city is known for the Eiffel Tower "
    "and hosted the 2024 Summer Olympics."
)
SANITY_ENTAILED = "The capital of France is Paris."
SANITY_NEUTRAL = "France produces excellent wine."
SANITY_CONTRADICTED = "The capital of France is Berlin."


def sanity_check(candidate: NliCandidate) -> dict:
    """The three-class direction check: an entailed, a neutral, and a contradicted
    hypothesis against one premise must each win their own class, or the run stops."""
    entailed = candidate.probs(SANITY_PREMISE, SANITY_ENTAILED)
    neutral = candidate.probs(SANITY_PREMISE, SANITY_NEUTRAL)
    contradicted = candidate.probs(SANITY_PREMISE, SANITY_CONTRADICTED)
    failures = []
    if entailed.entailment <= max(entailed.neutral, entailed.contradiction):
        failures.append(f"entailed pair not argmax-entailment: {entailed}")
    if neutral.neutral <= max(neutral.entailment, neutral.contradiction):
        failures.append(f"neutral pair not argmax-neutral: {neutral}")
    if contradicted.contradiction <= max(contradicted.entailment, contradicted.neutral):
        failures.append(f"contradicted pair not argmax-contradiction: {contradicted}")
    if failures:
        raise RuntimeError(f"{candidate.name} failed direction sanity check: {failures}")
    return {
        "entailed": entailed.__dict__,
        "neutral": neutral.__dict__,
        "contradicted": contradicted.__dict__,
    }


def fixed_pair_latency(candidate: NliCandidate, premise: str, hypothesis: str) -> dict:
    candidate.probs(premise, hypothesis)  # warm-up, uncounted
    millis = []
    for _ in range(LATENCY_REPEATS):
        start = time.perf_counter()
        candidate.probs(premise, hypothesis)
        millis.append((time.perf_counter() - start) * 1000)
    return {"median_ms": statistics.median(millis), "p95_ms": percentile(millis, 95)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True, choices=sorted(CANDIDATES))
    parser.add_argument("--per-task", type=int, default=200)
    parser.add_argument("--seed", type=int, default=18)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    examples = stratified_slice(load_ragtruth_test(), per_task=args.per_task, seed=args.seed)
    print(f"slice: {len(examples)} responses", flush=True)

    process = psutil.Process()
    load_start = time.perf_counter()
    candidate = load(args.candidate)
    load_seconds = time.perf_counter() - load_start
    rss_after_load = process.memory_info().rss

    sanity = sanity_check(candidate)
    print(f"sanity: {sanity}", flush=True)

    rows = []  # (class, probs) per sentence, harness-wide metric views read this
    per_example = []
    sentence_millis: list[float] = []
    response_millis: list[float] = []
    for i, example in enumerate(examples):
        response_ms = 0.0
        for claim, cls in nli_sentence_classes(example):
            start = time.perf_counter()
            probs = candidate.probs(example.context, claim.text)
            ms = (time.perf_counter() - start) * 1000
            sentence_millis.append(ms)
            response_ms += ms
            rows.append((cls, probs))
            per_example.append(
                {
                    "id": example.id,
                    "task_type": example.task_type,
                    "start": claim.start,
                    "end": claim.end,
                    "class": cls,
                    "entailment": probs.entailment,
                    "neutral": probs.neutral,
                    "contradiction": probs.contradiction,
                    "truncated": probs.truncated,
                    "ms": ms,
                }
            )
        response_millis.append(response_ms)
        progress(i + 1, len(examples), every=50)

    classes = [cls for cls, _ in rows]
    hallucinated_rows = [(cls, probs) for cls, probs in rows if cls != SUPPORTED]
    contradiction_labels, contradiction_scores = contradiction_pairs(rows)
    hallucination_labels, hallucination_risks = hallucination_pairs(rows)
    # Precision/recall of the argmax `contradicted` verdict among hallucinated
    # sentences: of the sentences it calls refuted, how many are; of the refuted,
    # how many it finds.
    verdict_precision, verdict_recall = precision_recall(
        contradiction_labels, [predicted_contradicted(probs) for _, probs in hallucinated_rows]
    )
    supported_rows = [probs for cls, probs in rows if cls == SUPPORTED]
    false_contradicted_supported = sum(
        predicted_contradicted(probs) for probs in supported_rows
    ) / len(supported_rows)

    result = {
        "candidate": candidate.name,
        "repo": candidate.repo,
        "revision": candidate.revision,
        "max_length": candidate.max_length,
        "slice": {
            "per_task": args.per_task,
            "seed": args.seed,
            "responses": len(examples),
            "sentences": len(rows),
            "conflict_sentences": classes.count("conflict"),
            "baseless_sentences": classes.count("baseless"),
            "supported_sentences": classes.count("supported"),
        },
        "metrics": {
            "contradiction_auroc": auroc(contradiction_labels, contradiction_scores),
            "hallucination_auroc": auroc(hallucination_labels, hallucination_risks),
            "contradicted_verdict_precision": verdict_precision,
            "contradicted_verdict_recall": verdict_recall,
            "false_contradicted_rate_supported": false_contradicted_supported,
            "truncated_share": sum(p.truncated for _, p in rows) / len(rows),
        },
        "latency": {
            "per_sentence_median_ms": statistics.median(sentence_millis),
            "per_sentence_p95_ms": percentile(sentence_millis, 95),
            "per_response_median_ms": statistics.median(response_millis),
            "per_response_p95_ms": percentile(response_millis, 95),
            "short_pair": fixed_pair_latency(candidate, SHORT_EVIDENCE, SHORT_CLAIM),
            "long_pair_500w": fixed_pair_latency(candidate, LONG_EVIDENCE, LONG_CLAIM),
        },
        "footprint": {
            "load_seconds": load_seconds,
            "rss_after_load_bytes": rss_after_load,
            "rss_after_run_bytes": process.memory_info().rss,
            "weights_on_disk_bytes": weights_on_disk_bytes([(candidate.repo, candidate.revision)]),
        },
        "sanity": sanity,
        "environment": environment(hardware=True),
        "per_example": per_example,
    }

    write_results(args.out, result)
    print(
        json.dumps({k: result[k] for k in ("candidate", "slice", "metrics", "latency")}, indent=1)
    )


if __name__ == "__main__":
    main()
