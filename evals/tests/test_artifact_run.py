"""The assembled artifact must load through the engine's own loader — the writer
and the serve-time schema cannot drift apart."""

import pytest
import yaml
from bench.artifact_run import build_artifact

from engine.calibration import load_artifact


def claim_fit_result() -> dict:
    return {
        "config_version": "0.5.0",
        "coefficients": {"a": 0.83, "b": 0.51},
        "bindings": {
            "model": "fake/model",
            "revision": "deadbeef",
            "inference_mode": "joint-questionless-v1",
            "claim_unit": "sentence-maxrisk-v1",
        },
        "fit": {
            "dataset": "wandb/RAGTruth-processed test",
            "exclusion": "stratified 200-per-task seed-18 benchmark slice removed before fitting",
            "responses": 2100,
            "sentences": 14589,
            "sha256": "0" * 64,
            "fitted_at": "2026-07-12",
        },
        "in_domain": {
            "slice": "stratified 200-per-task seed-18",
            "sentences": 4240,
            "ece_calibrated": 0.0074,
        },
    }


def ood_result() -> dict:
    return {
        "coefficients": {"a": 0.83, "b": 0.51},
        "slice": {
            "dataset": "lytang/LLM-AggreFact test",
            "subsets": ["a", "b"],
            "excluded_subsets": {"RAGTruth": "fitted on RAGTruth"},
            "claims": 1000,
        },
        "out_of_domain": {"ece_calibrated": 0.153},
    }


def span_result() -> dict:
    return {
        "config_version": "0.5.0",
        "span_threshold": 0.5,
        "source": "transferred",
        "coefficients": {"a": 0.83, "b": 0.51},
        "bindings": claim_fit_result()["bindings"],
        "fit": {
            "dataset": "wandb/RAGTruth-processed test",
            "label_convention": "any-overlap",
            "exclusion": "stratified 200-per-task seed-18 benchmark slice removed before fitting",
            "responses": 2100,
            "spans": 3200,
            "unsupported_spans": 2400,
            "partial_overlap_spans": 300,
            "sha256": "1" * 64,
            "fitted_at": "2026-07-12",
        },
        "in_domain": {
            "slice": "stratified 200-per-task seed-18",
            "spans": 900,
            "ece_calibrated": 0.03,
        },
    }


def test_assembled_artifact_loads_through_the_engine(tmp_path):
    artifact = build_artifact(claim_fit_result(), ood_result(), span_result())
    path = tmp_path / "artifact.yaml"
    path.write_text(yaml.safe_dump(artifact))
    loaded = load_artifact(path)
    assert loaded.artifact_schema == 2
    assert loaded.coefficients.a == 0.83
    assert loaded.span.coefficients.a == 0.83
    assert loaded.span.span_threshold == 0.5
    assert loaded.span.fit.source == "transferred"
    assert loaded.span.fit.label_convention == "any-overlap"
    assert loaded.span.metrics.in_domain.ece == 0.03
    assert loaded.span.metrics.out_of_domain.measured is False
    assert loaded.span.metrics.out_of_domain.reason


def test_span_binding_drift_refuses_assembly():
    span = span_result()
    span["bindings"] = {**span["bindings"], "revision": "cafebabe"}
    with pytest.raises(SystemExit, match="bindings"):
        build_artifact(claim_fit_result(), ood_result(), span)


def test_claim_coefficient_drift_refuses_assembly():
    ood = ood_result()
    ood["coefficients"] = {"a": 0.9, "b": 0.5}
    with pytest.raises(SystemExit, match="coefficients"):
        build_artifact(claim_fit_result(), ood, span_result())
