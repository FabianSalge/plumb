"""Assemble the calibration artifact from the fit and OOD results (#32, ADR-0008).

Merges `bench.calibration_run` and `bench.aggrefact_run` outputs into the
versioned artifact file the engine loads: coefficients plus the bindings the
calibrator was fitted against, the fit-set identity, and both measured ECEs.
Kept as a separate step so the artifact can only exist once both measurements
do — an artifact with no out-of-domain number cannot be assembled.

Usage (from evals/):
    uv run python -m bench.artifact_run \
        --fit results/calibration-fit-lettucedetect-v2.json \
        --ood results/calibration-ood-lettucedetect-v2.json \
        --out ../config/calibration/lettucedect-v2-mmbert-base-sentence-v1.yaml
"""

import argparse
import json
from pathlib import Path

import yaml

ARTIFACT_SCHEMA = 1


class _IndentedDumper(yaml.SafeDumper):
    """Indents block sequences under their key, matching the repo yamllint rules."""

    def increase_indent(self, flow: bool = False, indentless: bool = False) -> None:
        return super().increase_indent(flow, False)


def build_artifact(fit: dict, ood: dict) -> dict:
    if fit["coefficients"] != ood["coefficients"]:
        raise SystemExit(
            f"fit and OOD results carry different coefficients: "
            f"{fit['coefficients']} vs {ood['coefficients']} — refit or re-measure"
        )
    return {
        "schema": ARTIFACT_SCHEMA,
        "method": "platt",
        "coefficients": {
            "a": fit["coefficients"]["a"],
            "b": fit["coefficients"]["b"],
        },
        "bindings": {
            "model": fit["bindings"]["model"],
            "revision": fit["bindings"]["revision"],
            "inference_mode": fit["bindings"]["inference_mode"],
            "claim_unit": fit["bindings"]["claim_unit"],
        },
        "fit": {
            "dataset": fit["fit"]["dataset"],
            "exclusion": fit["fit"]["exclusion"],
            "responses": fit["fit"]["responses"],
            "sentences": fit["fit"]["sentences"],
            "sha256": fit["fit"]["sha256"],
            "fitted_at": fit["fit"]["fitted_at"],
        },
        "metrics": {
            "in_domain": {
                "dataset": fit["fit"]["dataset"],
                "slice": fit["in_domain"]["slice"],
                "sentences": fit["in_domain"]["sentences"],
                "ece": fit["in_domain"]["ece_calibrated"],
            },
            "out_of_domain": {
                "dataset": ood["slice"]["dataset"],
                "subsets": ood["slice"]["subsets"],
                "excluded_subsets": ood["slice"]["excluded_subsets"],
                "claims": ood["slice"]["claims"],
                "ece": ood["out_of_domain"]["ece_calibrated"],
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit", type=Path, required=True)
    parser.add_argument("--ood", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    artifact = build_artifact(json.loads(args.fit.read_text()), json.loads(args.ood.read_text()))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(yaml.dump(artifact, Dumper=_IndentedDumper, sort_keys=False))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
