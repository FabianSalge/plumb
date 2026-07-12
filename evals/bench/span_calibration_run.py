"""Fit and validate the span calibrator on held-out RAGTruth (#40, ADR-0008).

Collects every engine-flagged span from the responses outside the seed-18
benchmark slice — the same fit population the claim calibrator used, scored
exactly as `/v1/verify` scores — labels each span unsupported iff it overlaps a
human-annotated hallucination span, and fits the span-level Platt map. Both
candidates are then evaluated on the seed-18 slice's spans: the transferred
claim coefficients (read from the claim fit results, so the two runs cannot
drift apart silently) and the span fit. The pre-registered rule
(`span_calibration.TRANSFER_MARGIN`) picks the served coefficients; both
candidates' reliability lands in the output either way.

The output JSON carries everything the artifact's span section needs;
`bench.artifact_run` merges it with the claim fit and OOD results.

Usage (from evals/):
    caffeinate uv run --extra tf5 python -m bench.span_calibration_run \
        --claim-fit results/calibration-fit-lettucedetect-v2.json \
        --out results/span-calibration-fit-lettucedetect-v2.json
"""

import argparse
import hashlib
import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from bench.calibration import PlattFit
from bench.data import Example, load_ragtruth_test, stratified_slice
from bench.harness import DEFAULT_CONFIG, environment, progress, write_results
from bench.metrics import auroc, ece, reliability_bins
from bench.span_calibration import (
    TRANSFER_MARGIN,
    bootstrap_ece_difference,
    decide_source,
    fit_span_platt,
    span_confidences,
)
from bench.spans import LabeledSpan, labeled_spans
from engine.decomposition import CLAIM_UNIT
from engine.signals import Scorer
from engine.signals.groundedness import INFERENCE_MODE, LettuceDetectScorer

LABEL_CONVENTION = "any-overlap"


def collect_spans(
    examples: list[Example], scorer: Scorer, span_threshold: float, label: str
) -> list[LabeledSpan]:
    """Score every response the way `/v1/verify` does and collect its flagged spans."""
    spans: list[LabeledSpan] = []
    for i, example in enumerate(examples):
        spans.extend(labeled_spans(example, scorer, span_threshold))
        progress(i + 1, len(examples), label=label, every=100)
    return spans


def population(spans: list[LabeledSpan]) -> dict:
    return {
        "spans": len(spans),
        "unsupported_spans": sum(span.unsupported for span in spans),
        "partial_overlap_spans": sum(span.partial for span in spans),
    }


def main() -> None:
    from engine.config import load_config

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-task", type=int, default=200)
    parser.add_argument("--seed", type=int, default=18)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--claim-fit", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    claim_fit_result = json.loads(args.claim_fit.read_text())
    for name in ("model", "revision"):
        if claim_fit_result["bindings"][name] != getattr(cfg.groundedness, name):
            raise SystemExit(
                f"claim fit was measured against {name} "
                f"{claim_fit_result['bindings'][name]!r}, the running config pins "
                f"{getattr(cfg.groundedness, name)!r} — refit the claim calibrator first"
            )
    transfer = PlattFit(
        a=claim_fit_result["coefficients"]["a"], b=claim_fit_result["coefficients"]["b"]
    )

    examples = load_ragtruth_test()
    benchmark_slice = stratified_slice(examples, per_task=args.per_task, seed=args.seed)
    slice_ids = {example.id for example in benchmark_slice}
    fit_examples = [example for example in examples if example.id not in slice_ids]
    span_threshold = cfg.groundedness.span_threshold
    print(
        f"fit set: {len(fit_examples)} responses "
        f"(test split {len(examples)} minus seed-{args.seed} slice {len(benchmark_slice)}), "
        f"spans flagged at threshold {span_threshold}",
        flush=True,
    )

    scorer = LettuceDetectScorer.load(cfg.groundedness)

    fit_spans = collect_spans(fit_examples, scorer, span_threshold, label="fit")
    fit = fit_span_platt(
        [span.unsupported for span in fit_spans], [span.raw_risk for span in fit_spans]
    )
    fingerprints = [
        f"{span.example_id}\t{span.start}\t{span.end}\t{span.unsupported}" for span in fit_spans
    ]
    fit_hash = hashlib.sha256("\n".join(fingerprints).encode()).hexdigest()
    print(f"fitted a={fit.a:.6f} b={fit.b:.6f} on {len(fit_spans)} spans", flush=True)

    eval_spans = collect_spans(benchmark_slice, scorer, span_threshold, label="in-domain")
    outcomes = [span.unsupported for span in eval_spans]
    risks = [span.raw_risk for span in eval_spans]
    transfer_confidences = span_confidences(transfer, risks)
    fitted_confidences = span_confidences(fit, risks)
    ece_transferred = ece(outcomes, transfer_confidences)
    ece_fitted = ece(outcomes, fitted_confidences)
    ci_low, ci_high = bootstrap_ece_difference(outcomes, transfer_confidences, fitted_confidences)
    source = decide_source(ece_transferred, ece_fitted)
    chosen, chosen_confidences = (
        (transfer, transfer_confidences) if source == "transferred" else (fit, fitted_confidences)
    )
    print(
        f"transfer ECE {ece_transferred:.4f} vs fitted ECE {ece_fitted:.4f} "
        f"(diff CI95 [{ci_low:.4f}, {ci_high:.4f}]) -> {source}",
        flush=True,
    )

    raw_auroc = auroc(outcomes, risks)
    calibrated_auroc = auroc(outcomes, chosen_confidences)
    if abs(raw_auroc - calibrated_auroc) > 1e-9:
        raise SystemExit(
            f"sanity failed: the monotone map moved AUROC {raw_auroc} -> {calibrated_auroc}"
        )

    result = {
        "config_version": cfg.version,
        "span_threshold": span_threshold,
        "source": source,
        "coefficients": {"a": chosen.a, "b": chosen.b},
        "candidates": {
            "transferred": {"a": transfer.a, "b": transfer.b, "ece": ece_transferred},
            "fitted": {"a": fit.a, "b": fit.b, "ece": ece_fitted},
            "ece_difference_ci95": [ci_low, ci_high],
            "transfer_margin": TRANSFER_MARGIN,
        },
        "bindings": {
            "model": cfg.groundedness.model,
            "revision": cfg.groundedness.revision,
            "inference_mode": INFERENCE_MODE,
            "claim_unit": CLAIM_UNIT,
        },
        "fit": {
            "dataset": "wandb/RAGTruth-processed test",
            "label_convention": LABEL_CONVENTION,
            "exclusion": (
                f"stratified {args.per_task}-per-task seed-{args.seed} benchmark "
                "slice removed before fitting"
            ),
            "responses": len(fit_examples),
            **population(fit_spans),
            "sha256": fit_hash,
            "fitted_at": datetime.now(UTC).date().isoformat(),
        },
        "in_domain": {
            "slice": f"stratified {args.per_task}-per-task seed-{args.seed}",
            "responses": len(benchmark_slice),
            **population(eval_spans),
            "ece_raw": ece(outcomes, risks),
            "ece_calibrated": ece(outcomes, chosen_confidences),
            "span_auroc": calibrated_auroc,
            "reliability": [asdict(bin) for bin in reliability_bins(outcomes, chosen_confidences)],
            "reliability_transferred": [
                asdict(bin) for bin in reliability_bins(outcomes, transfer_confidences)
            ],
            "reliability_fitted": [
                asdict(bin) for bin in reliability_bins(outcomes, fitted_confidences)
            ],
        },
        "environment": environment(),
    }

    write_results(args.out, result)
    summary = {
        k: result[k] for k in ("span_threshold", "source", "coefficients", "candidates", "fit")
    }
    summary["in_domain"] = {
        k: v for k, v in result["in_domain"].items() if not k.startswith("reliability")
    }
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
