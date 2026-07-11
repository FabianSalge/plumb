"""Fit the Platt calibrator on held-out RAGTruth and validate in-domain (#32, ADR-0008).

Fits on the sentences of every RAGTruth test response outside the seed-18
benchmark slice — excluded by the same `stratified_slice` call the benchmark
runs, so nothing the calibrator is fitted on ever evaluates it. Each response is
scored exactly as `/v1/verify` scores it (one whole-answer joint pass, the
engine's own segmenter and reduction); a sentence's outcome is supported iff it
overlaps no annotated span. In-domain validation applies the fitted map to the
seed-18 slice's sentences and reports ECE with reliability-diagram data, plus a
sanity check that the monotone map left AUROC untouched.

The output JSON carries everything the calibration artifact needs from the fit;
`bench.artifact_run` merges it with the out-of-domain results into the artifact.

Usage (from evals/):
    caffeinate uv run --extra tf5 python -m bench.calibration_run \
        --out results/calibration-fit-lettucedetect-v2.json
"""

import argparse
import hashlib
import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from bench.calibration import apply_fit, fit_platt
from bench.data import Example, load_ragtruth_test, stratified_slice
from bench.harness import DEFAULT_CONFIG, environment, progress, write_results
from bench.metrics import auroc, ece, reliability_bins
from bench.sentence import scored_sentences
from engine.decomposition import CLAIM_UNIT
from engine.signals import Scorer
from engine.signals.groundedness import INFERENCE_MODE, LettuceDetectScorer


def sentence_outcomes(
    examples: list[Example], scorer: Scorer, span_threshold: float, label: str
) -> tuple[list[str], list[int], list[float]]:
    """Score every sentence of `examples` the way `/v1/verify` does: returns per-sentence
    fingerprints (`id\\tstart\\tend\\toutcome`), outcomes (1 = supported), raw supports."""
    fingerprints: list[str] = []
    outcomes: list[int] = []
    supports: list[float] = []
    for i, example in enumerate(examples):
        for claim, hallucinated in scored_sentences(example, scorer, span_threshold):
            outcome = 1 - hallucinated
            fingerprints.append(f"{example.id}\t{claim.start}\t{claim.end}\t{outcome}")
            outcomes.append(outcome)
            supports.append(claim.support)
        progress(i + 1, len(examples), label=label, every=100)
    return fingerprints, outcomes, supports


def main() -> None:
    from engine.config import load_config

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-task", type=int, default=200)
    parser.add_argument("--seed", type=int, default=18)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    examples = load_ragtruth_test()
    benchmark_slice = stratified_slice(examples, per_task=args.per_task, seed=args.seed)
    slice_ids = {example.id for example in benchmark_slice}
    fit_examples = [example for example in examples if example.id not in slice_ids]
    print(
        f"fit set: {len(fit_examples)} responses "
        f"(test split {len(examples)} minus seed-{args.seed} slice {len(benchmark_slice)})",
        flush=True,
    )

    scorer = LettuceDetectScorer.load(cfg.groundedness)
    span_threshold = cfg.groundedness.span_threshold

    fingerprints, outcomes, supports = sentence_outcomes(
        fit_examples, scorer, span_threshold, label="fit"
    )
    fit = fit_platt(outcomes, supports)
    fit_hash = hashlib.sha256("\n".join(fingerprints).encode()).hexdigest()
    print(f"fitted a={fit.a:.6f} b={fit.b:.6f} on {len(outcomes)} sentences", flush=True)

    _, eval_outcomes, eval_supports = sentence_outcomes(
        benchmark_slice, scorer, span_threshold, label="in-domain"
    )
    confidences = apply_fit(fit, eval_supports)
    raw_auroc = auroc([1 - o for o in eval_outcomes], [1.0 - s for s in eval_supports])
    calibrated_auroc = auroc([1 - o for o in eval_outcomes], [1.0 - c for c in confidences])
    if abs(raw_auroc - calibrated_auroc) > 1e-9:
        raise SystemExit(
            f"sanity failed: the monotone map moved AUROC {raw_auroc} -> {calibrated_auroc}"
        )

    result = {
        "config_version": cfg.version,
        "coefficients": {"a": fit.a, "b": fit.b},
        "bindings": {
            "model": cfg.groundedness.model,
            "revision": cfg.groundedness.revision,
            "inference_mode": INFERENCE_MODE,
            "claim_unit": CLAIM_UNIT,
        },
        "fit": {
            "dataset": "wandb/RAGTruth-processed test",
            "exclusion": (
                f"stratified {args.per_task}-per-task seed-{args.seed} benchmark "
                "slice removed before fitting"
            ),
            "responses": len(fit_examples),
            "sentences": len(outcomes),
            "supported_sentences": sum(outcomes),
            "sha256": fit_hash,
            "fitted_at": datetime.now(UTC).date().isoformat(),
        },
        "in_domain": {
            "slice": f"stratified {args.per_task}-per-task seed-{args.seed}",
            "responses": len(benchmark_slice),
            "sentences": len(eval_outcomes),
            "supported_sentences": sum(eval_outcomes),
            "ece_raw": ece(eval_outcomes, eval_supports),
            "ece_calibrated": ece(eval_outcomes, confidences),
            "sentence_auroc": calibrated_auroc,
            "reliability": [asdict(bin) for bin in reliability_bins(eval_outcomes, confidences)],
            "reliability_raw": [
                asdict(bin) for bin in reliability_bins(eval_outcomes, eval_supports)
            ],
        },
        "environment": environment(),
    }

    write_results(args.out, result)
    summary = {k: result[k] for k in ("coefficients", "fit")}
    summary["in_domain"] = {
        k: v for k, v in result["in_domain"].items() if not k.startswith("reliability")
    }
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
