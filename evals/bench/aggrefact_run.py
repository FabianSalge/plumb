"""Out-of-domain calibration check on LLM-AggreFact (#32, ADR-0008).

Applies the fitted Platt map (from `bench.calibration_run`'s output) to an
LLM-AggreFact slice the calibrator never saw — RAGTruth and every subset found
in the pinned model's training mix excluded, with the exclusions recorded in
the output. Claims are scored at the benchmark's own granularity: one claim,
one joint pass against its document, support reduced over the whole claim.

Needs an authenticated HF session — lytang/LLM-AggreFact is gated.

Usage (from evals/):
    caffeinate uv run --extra tf5 python -m bench.aggrefact_run \
        --fit results/calibration-fit-lettucedetect-v2.json \
        --out results/calibration-ood-lettucedetect-v2.json
"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from bench.aggrefact import load_aggrefact_test, stratified_claims
from bench.harness import DEFAULT_CONFIG, environment, progress, write_results
from bench.metrics import auroc, ece, reliability_bins
from engine.calibration import platt_confidence
from engine.decomposition import Claim, reduce_claim
from engine.signals.groundedness import LettuceDetectScorer


def main() -> None:
    from engine.config import load_config

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit", type=Path, required=True)
    parser.add_argument("--per-subset", type=int, default=100)
    parser.add_argument("--seed", type=int, default=18)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    fit = json.loads(args.fit.read_text())
    bindings = fit["bindings"]
    if (bindings["model"], bindings["revision"]) != (
        cfg.groundedness.model,
        cfg.groundedness.revision,
    ):
        raise SystemExit(
            f"fit bindings {bindings['model']}@{bindings['revision']} do not match "
            f"the running config {cfg.groundedness.model}@{cfg.groundedness.revision}"
        )
    a, b = fit["coefficients"]["a"], fit["coefficients"]["b"]

    examples, exclusions = load_aggrefact_test()
    sliced = stratified_claims(examples, per_subset=args.per_subset, seed=args.seed)
    print(f"slice: {len(sliced)} claims from {len({e.subset for e in sliced})} subsets", flush=True)
    print(f"excluded subsets: {exclusions}", flush=True)

    scorer = LettuceDetectScorer.load(cfg.groundedness)
    outcomes: list[int] = []
    supports: list[float] = []
    for i, example in enumerate(sliced):
        scores = scorer.score(example.claim, [example.doc])
        whole = Claim(text=example.claim, start=0, end=len(example.claim))
        reduced = reduce_claim(whole, scores, cfg.groundedness.span_threshold)
        outcomes.append(int(example.supported))
        supports.append(reduced.support)
        progress(i + 1, len(sliced), noun="claims", every=100)

    confidences = [platt_confidence(s, a=a, b=b) for s in supports]
    result = {
        "config_version": cfg.version,
        "fit_source": str(args.fit),
        "coefficients": {"a": a, "b": b},
        "slice": {
            "dataset": "lytang/LLM-AggreFact test",
            "per_subset": args.per_subset,
            "seed": args.seed,
            "claims": len(sliced),
            "supported_claims": sum(outcomes),
            "subsets": sorted({e.subset for e in sliced}),
            "excluded_subsets": exclusions,
            "granularity": "LLM-AggreFact's own claim unit, scored as a single claim",
        },
        "out_of_domain": {
            "ece_raw": ece(outcomes, supports),
            "ece_calibrated": ece(outcomes, confidences),
            "claim_auroc": auroc([1 - o for o in outcomes], [1.0 - s for s in supports]),
            "reliability": [asdict(bin) for bin in reliability_bins(outcomes, confidences)],
            "reliability_raw": [asdict(bin) for bin in reliability_bins(outcomes, supports)],
        },
        "environment": environment(),
    }

    write_results(args.out, result)
    summary = dict(result["out_of_domain"])
    summary.pop("reliability")
    summary.pop("reliability_raw")
    print(json.dumps({"slice": result["slice"], "out_of_domain": summary}, indent=1))


if __name__ == "__main__":
    main()
