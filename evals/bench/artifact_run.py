"""Assemble the calibration artifact from the fit, OOD, and span results
(#32, #40, ADR-0008).

Merges `bench.calibration_run`, `bench.aggrefact_run`, and
`bench.span_calibration_run` outputs into the versioned artifact file the engine
loads: claim and span coefficients plus the bindings they were fitted against,
the fit-set identities, and the measured ECEs. Kept as a separate step so the
artifact can only exist once every measurement does — an artifact with no
out-of-domain number or no span reliability cannot be assembled. Span-level
out-of-domain error is recorded as unmeasured with its reason: no span-annotated
out-of-domain dataset exists.

Usage (from evals/):
    uv run python -m bench.artifact_run \
        --fit results/calibration-fit-lettucedetect-v2.json \
        --ood results/calibration-ood-lettucedetect-v2.json \
        --span results/span-calibration-fit-lettucedetect-v2.json \
        --out ../config/calibration/lettucedect-v2-mmbert-base-sentence-v2.yaml
"""

import argparse
import json
from pathlib import Path

import yaml

ARTIFACT_SCHEMA = 2

SPAN_OOD_REASON = (
    "no span-annotated out-of-domain dataset exists; LLM-AggreFact carries claim-level labels only"
)


class _IndentedDumper(yaml.SafeDumper):
    """Indents block sequences under their key, matching the repo yamllint rules."""

    def increase_indent(self, flow: bool = False, indentless: bool = False) -> None:
        return super().increase_indent(flow, False)


def build_artifact(fit: dict, ood: dict, span: dict) -> dict:
    if fit["coefficients"] != ood["coefficients"]:
        raise SystemExit(
            f"fit and OOD results carry different coefficients: "
            f"{fit['coefficients']} vs {ood['coefficients']} — refit or re-measure"
        )
    if span["bindings"] != fit["bindings"]:
        raise SystemExit(
            f"span and claim fits were measured against different bindings: "
            f"{span['bindings']} vs {fit['bindings']} — refit both"
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
        "span": {
            "coefficients": {
                "a": span["coefficients"]["a"],
                "b": span["coefficients"]["b"],
            },
            "span_threshold": span["span_threshold"],
            "fit": {
                "source": span["source"],
                "dataset": span["fit"]["dataset"],
                "label_convention": span["fit"]["label_convention"],
                "exclusion": span["fit"]["exclusion"],
                "spans": span["fit"]["spans"],
                "sha256": span["fit"]["sha256"],
                "fitted_at": span["fit"]["fitted_at"],
            },
            "metrics": {
                "in_domain": {
                    "dataset": span["fit"]["dataset"],
                    "slice": span["in_domain"]["slice"],
                    "spans": span["in_domain"]["spans"],
                    "ece": span["in_domain"]["ece_calibrated"],
                },
                "out_of_domain": {
                    "measured": False,
                    "reason": SPAN_OOD_REASON,
                },
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit", type=Path, required=True)
    parser.add_argument("--ood", type=Path, required=True)
    parser.add_argument("--span", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    artifact = build_artifact(
        json.loads(args.fit.read_text()),
        json.loads(args.ood.read_text()),
        json.loads(args.span.read_text()),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(yaml.dump(artifact, Dumper=_IndentedDumper, sort_keys=False))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
