"""Builders for the on-disk config + calibration artifact pair the engine loads."""

from pathlib import Path

import yaml

from engine.decomposition import CLAIM_UNIT
from engine.signals.groundedness import INFERENCE_MODE


def make_config(*, threshold: float = 0.5, span_threshold: float = 0.5) -> dict:
    return {
        "version": "test-1",
        "signals": {
            "groundedness": {
                "model": "fake/model",
                "revision": "deadbeef",
                "threshold": threshold,
                "span_threshold": span_threshold,
                "calibration": "calibration.yaml",
            }
        },
    }


def make_artifact(
    *,
    a: float = 1.0,
    b: float = 0.0,
    span_a: float = 1.0,
    span_b: float = 0.0,
    span_threshold: float = 0.5,
    **binding_overrides,
) -> dict:
    """A valid calibration artifact matching `make_config`'s fake model. The identity
    coefficients (a=1, b=0) make confidence equal raw support — and span confidence
    equal raw span risk — up to the ε clamp, so threshold-behaviour tests read
    naturally. `span_threshold` must match the config's, or startup refuses."""
    bindings = {
        "model": "fake/model",
        "revision": "deadbeef",
        "inference_mode": INFERENCE_MODE,
        "claim_unit": CLAIM_UNIT,
    }
    bindings.update(binding_overrides)
    return {
        "schema": 2,
        "method": "platt",
        "coefficients": {"a": a, "b": b},
        "bindings": bindings,
        "fit": {
            "dataset": "test-fixture",
            "exclusion": "none",
            "responses": 3,
            "sentences": 12,
            "sha256": "0" * 64,
            "fitted_at": "2026-07-11",
        },
        "metrics": {
            "in_domain": {"dataset": "test-fixture", "slice": "s", "sentences": 12, "ece": 0.01},
            "out_of_domain": {
                "dataset": "ood-fixture",
                "subsets": ["a"],
                "excluded_subsets": {"RAGTruth": "fitted on RAGTruth"},
                "claims": 5,
                "ece": 0.06,
            },
        },
        "span": {
            "coefficients": {"a": span_a, "b": span_b},
            "span_threshold": span_threshold,
            "fit": {
                "source": "fitted",
                "dataset": "test-fixture",
                "label_convention": "any-overlap",
                "exclusion": "none",
                "spans": 40,
                "sha256": "1" * 64,
                "fitted_at": "2026-07-11",
            },
            "metrics": {
                "in_domain": {"dataset": "test-fixture", "slice": "s", "spans": 20, "ece": 0.02},
                "out_of_domain": {
                    "measured": False,
                    "reason": "no span-annotated out-of-domain dataset exists",
                },
            },
        },
    }


def write_config(
    directory: Path, *, config: dict | None = None, artifact: dict | None = None
) -> Path:
    """Write a verifier config and its calibration artifact side by side."""
    path = directory / "verifier.yaml"
    path.write_text(yaml.safe_dump(config or make_config()))
    (directory / "calibration.yaml").write_text(yaml.safe_dump(artifact or make_artifact()))
    return path
