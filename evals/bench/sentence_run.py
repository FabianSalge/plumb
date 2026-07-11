"""Measure sentence-level discrimination of the shipping scorer on RAGTruth (#45).

One whole-answer forward pass per response, segmented by the engine's own
segmenter (`engine.decomposition`), reduced to per-sentence support. A sentence
is hallucinated iff it overlaps an annotated span; the metric is AUROC over
sentence risk = 1 − support. This scores the deployed, question-less
configuration — passages only, exactly as `/v1/verify` runs.

Usage (from evals/):
    uv run --extra tf5 python -m bench.sentence_run --per-task 200 \
        --out results/sentence-lettucedetect-v2.json
"""

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

from bench.data import load_ragtruth_test, stratified_slice
from bench.metrics import auroc
from bench.sentence import sentence_scores
from engine.config import load_config
from engine.scoring import LettuceDetectScorer

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "config" / "verifier.yaml"


def percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, round(pct / 100 * len(ordered) + 0.5) - 1)
    return ordered[index]


def main() -> None:
    import torch
    import transformers

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-task", type=int, default=200)
    parser.add_argument("--seed", type=int, default=18)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    examples = stratified_slice(load_ragtruth_test(), per_task=args.per_task, seed=args.seed)
    print(f"slice: {len(examples)} responses", flush=True)

    scorer = LettuceDetectScorer.load(cfg.groundedness)

    labels: list[int] = []
    risks: list[float] = []
    millis: list[float] = []
    per_response_sentences: list[int] = []
    for i, example in enumerate(examples):
        start = time.perf_counter()
        pairs = list(sentence_scores(example, scorer, cfg.groundedness.span_threshold))
        millis.append((time.perf_counter() - start) * 1000)
        per_response_sentences.append(len(pairs))
        for label, risk in pairs:
            labels.append(label)
            risks.append(risk)
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(examples)} responses scored", flush=True)

    result = {
        "model": cfg.groundedness.model,
        "revision": cfg.groundedness.revision,
        "config_version": cfg.version,
        "slice": {
            "per_task": args.per_task,
            "seed": args.seed,
            "responses": len(examples),
            "sentences": len(labels),
            "hallucinated_sentences": sum(labels),
        },
        "metrics": {"sentence_auroc": auroc(labels, risks)},
        "latency": {
            "per_response_median_ms": statistics.median(millis),
            "per_response_p95_ms": percentile(millis, 95),
        },
        "sentences_per_response": {
            "mean": sum(per_response_sentences) / len(per_response_sentences),
            "max": max(per_response_sentences),
        },
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({k: result[k] for k in ("slice", "metrics", "latency")}, indent=1))


if __name__ == "__main__":
    main()
