"""Run one candidate over the RAGTruth slice and write a results JSON.

Usage:
    uv run python -m bench.run --candidate hhem --per-task 200 --out results/hhem.json

Every run records the slice parameters, hardware, library versions, pinned
revision, accuracy metrics, latency (per-example plus the spike's fixed
short/500-word protocol), memory, and weights-on-disk size.
"""

import argparse
import json
import statistics
import time
from pathlib import Path

import psutil

from bench.data import Example, load_ragtruth_test, stratified_slice
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
from bench.metrics import auroc, balanced_accuracy, f1_score, precision_recall

THRESHOLD = 0.5  # pred_hallucinated = support < THRESHOLD, every candidate alike

CANDIDATES = {
    "hhem": ("bench.adapters.hhem", "load"),
    "lettucedetect-v1-large": ("bench.adapters.lettucedetect_adapter", "load_v1_large"),
    "lettucedetect-v2-mmbert-base": ("bench.adapters.lettucedetect_adapter", "load_v2_mmbert_base"),
    "minicheck-flan-t5-large": ("bench.adapters.minicheck_flan_t5", "load"),
    "granite-guardian-3.2-3b-a800m": ("bench.adapters.granite_guardian", "load"),
}

SANITY_EVIDENCE = (
    "Paris is the capital of France. The city is known for the Eiffel Tower "
    "and hosted the 2024 Summer Olympics."
)
SANITY_QUESTION = "What is the capital of France?"
SANITY_SUPPORTED = "The capital of France is Paris."
SANITY_CONTRADICTED = "The capital of France is Berlin."


def synthetic_example(evidence: str, claim: str, question: str = "") -> Example:
    return Example(
        id="synthetic",
        task_type="QA",
        query=question,
        context=evidence,
        response=claim,
        hallucinated=False,
    )


def sanity_check(candidate) -> dict:
    supported = candidate.support_score(
        synthetic_example(SANITY_EVIDENCE, SANITY_SUPPORTED, SANITY_QUESTION)
    )
    contradicted = candidate.support_score(
        synthetic_example(SANITY_EVIDENCE, SANITY_CONTRADICTED, SANITY_QUESTION)
    )
    if supported <= contradicted:
        raise RuntimeError(
            f"{candidate.name} failed direction sanity check: "
            f"supported={supported:.3f} <= contradicted={contradicted:.3f}"
        )
    return {"supported": supported, "contradicted": contradicted}


def timed_scores(candidate, examples: list[Example]) -> tuple[list[float], list[float]]:
    supports, millis = [], []
    for i, example in enumerate(examples):
        start = time.perf_counter()
        supports.append(candidate.support_score(example))
        millis.append((time.perf_counter() - start) * 1000)
        progress(i + 1, len(examples), noun="examples", every=50)
    return supports, millis


def fixed_pair_latency(candidate, evidence: str, claim: str) -> dict:
    example = synthetic_example(evidence, claim)
    candidate.support_score(example)  # warm-up, uncounted
    millis = []
    for _ in range(LATENCY_REPEATS):
        start = time.perf_counter()
        candidate.support_score(example)
        millis.append((time.perf_counter() - start) * 1000)
    return {"median_ms": statistics.median(millis), "p95_ms": percentile(millis, 95)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True, choices=sorted(CANDIDATES))
    parser.add_argument("--per-task", type=int, default=200)
    parser.add_argument("--seed", type=int, default=18)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    module_name, loader_name = CANDIDATES[args.candidate]
    module = __import__(module_name, fromlist=[loader_name])

    examples = stratified_slice(load_ragtruth_test(), per_task=args.per_task, seed=args.seed)
    labels = [int(e.hallucinated) for e in examples]
    print(f"slice: {len(examples)} examples, {sum(labels)} hallucinated", flush=True)

    process = psutil.Process()
    load_start = time.perf_counter()
    candidate = getattr(module, loader_name)()
    load_seconds = time.perf_counter() - load_start
    rss_after_load = process.memory_info().rss

    sanity = sanity_check(candidate)
    print(f"sanity: {sanity}", flush=True)

    supports, millis = timed_scores(candidate, examples)
    risks = [1 - s for s in supports]
    predicted = [s < THRESHOLD for s in supports]

    precision, recall = precision_recall(labels, predicted)
    repo = getattr(candidate, "repo", None) or getattr(module, "REPO", None)
    revision = getattr(candidate, "revision", None) or getattr(module, "REVISION", None)
    result = {
        "candidate": candidate.name,
        "repo": repo,
        "revision": revision,
        "slice": {"per_task": args.per_task, "seed": args.seed, "n": len(examples)},
        "threshold": THRESHOLD,
        "metrics": {
            "auroc": auroc(labels, risks),
            "balanced_accuracy": balanced_accuracy(labels, predicted),
            "f1_hallucinated": f1_score(labels, predicted),
            "precision_hallucinated": precision,
            "recall_hallucinated": recall,
        },
        "latency": {
            "slice_median_ms": statistics.median(millis),
            "slice_p95_ms": percentile(millis, 95),
            "short_pair": fixed_pair_latency(candidate, SHORT_EVIDENCE, SHORT_CLAIM),
            "long_pair_500w": fixed_pair_latency(candidate, LONG_EVIDENCE, LONG_CLAIM),
        },
        "footprint": {
            "load_seconds": load_seconds,
            "rss_after_load_bytes": rss_after_load,
            "rss_after_run_bytes": process.memory_info().rss,
            "weights_on_disk_bytes": weights_on_disk_bytes(
                [(repo, revision)] + [(extra, None) for extra in getattr(module, "EXTRA_REPOS", [])]
            ),
        },
        "sanity": sanity,
        "environment": environment(hardware=True),
        "per_example": [
            {
                "id": e.id,
                "task_type": e.task_type,
                "hallucinated": e.hallucinated,
                "support": s,
                "ms": ms,
            }
            for e, s, ms in zip(examples, supports, millis, strict=True)
        ],
    }

    write_results(args.out, result)
    print(json.dumps({k: result[k] for k in ("candidate", "metrics", "latency")}, indent=1))


if __name__ == "__main__":
    main()
