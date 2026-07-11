"""Unit tests for the calibration engine: the Platt map, the versioned artifact,
and the binding validation that refuses a mismatched calibrator."""

from pathlib import Path

import pytest
import yaml

from engine.calibration import (
    EPSILON,
    CalibrationError,
    load_artifact,
    platt_confidence,
    validate_bindings,
)
from engine.config import SignalModelConfig
from engine.decomposition import CLAIM_UNIT
from engine.signals.groundedness import INFERENCE_MODE


def test_identity_coefficients_return_the_raw_support():
    # a=1, b=0 is the identity on the logit scale
    assert platt_confidence(0.3, a=1.0, b=0.0) == pytest.approx(0.3)
    assert platt_confidence(0.75, a=1.0, b=0.0) == pytest.approx(0.75)


def test_map_is_strictly_monotone():
    supports = [0.0, 0.1, 0.4, 0.5, 0.9, 1.0]
    confidences = [platt_confidence(s, a=1.7, b=-0.4) for s in supports]
    assert all(c1 < c2 for c1, c2 in zip(confidences, confidences[1:], strict=False))


def test_saturated_supports_stay_strictly_inside_the_open_interval():
    low = platt_confidence(0.0, a=1.0, b=0.0)
    high = platt_confidence(1.0, a=1.0, b=0.0)
    assert 0.0 < low < high < 1.0


def test_clamp_is_epsilon():
    # support 0.0 clamps to EPSILON before the logit; identity coefficients expose it
    assert platt_confidence(0.0, a=1.0, b=0.0) == pytest.approx(EPSILON)
    assert platt_confidence(1.0, a=1.0, b=0.0) == pytest.approx(1.0 - EPSILON)


def test_known_value():
    # sigmoid(2 * logit(0.5) + 1) = sigmoid(1)
    assert platt_confidence(0.5, a=2.0, b=1.0) == pytest.approx(0.7310585786300049)


def test_out_of_range_support_fails_loudly():
    with pytest.raises(CalibrationError):
        platt_confidence(1.2, a=1.0, b=0.0)
    with pytest.raises(CalibrationError):
        platt_confidence(-0.1, a=1.0, b=0.0)


def test_non_finite_coefficients_fail_loudly():
    with pytest.raises(CalibrationError):
        platt_confidence(0.5, a=float("nan"), b=0.0)
    with pytest.raises(CalibrationError):
        platt_confidence(0.5, a=1.0, b=float("inf"))


# --- Artifact loading and binding validation ---


def artifact_dict() -> dict:
    return {
        "schema": 1,
        "method": "platt",
        "coefficients": {"a": 1.3, "b": -0.2},
        "bindings": {
            "model": "fake/model",
            "revision": "deadbeef",
            "inference_mode": INFERENCE_MODE,
            "claim_unit": CLAIM_UNIT,
        },
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
                "subsets": ["a", "b"],
                "excluded_subsets": {"RAGTruth": "fitted on RAGTruth"},
                "claims": 5,
                "ece": 0.06,
            },
        },
    }


def write_artifact(tmp_path, data: dict) -> str:
    path = tmp_path / "calibration.yaml"
    path.write_text(yaml.safe_dump(data))
    return str(path)


def signal_config(**overrides) -> SignalModelConfig:
    values = {
        "model": "fake/model",
        "revision": "deadbeef",
        "threshold": 0.5,
        "span_threshold": 0.5,
        "calibration": "calibration.yaml",
    }
    values.update(overrides)
    return SignalModelConfig.model_validate(values)


def test_complete_artifact_loads(tmp_path):
    artifact = load_artifact(write_artifact(tmp_path, artifact_dict()))
    assert artifact.coefficients.a == 1.3
    assert artifact.coefficients.b == -0.2
    assert artifact.bindings.model == "fake/model"
    assert artifact.confidence(0.5) == pytest.approx(platt_confidence(0.5, a=1.3, b=-0.2))


@pytest.mark.parametrize(
    "section,field",
    [
        ("coefficients", "a"),
        ("bindings", "model"),
        ("bindings", "revision"),
        ("bindings", "inference_mode"),
        ("bindings", "claim_unit"),
        ("fit", "sha256"),
        ("metrics", "in_domain"),
        ("metrics", "out_of_domain"),
    ],
)
def test_missing_field_fails_naming_it(tmp_path, section, field):
    data = artifact_dict()
    del data[section][field]
    with pytest.raises(CalibrationError, match=field):
        load_artifact(write_artifact(tmp_path, data))


def test_unknown_schema_version_fails(tmp_path):
    data = artifact_dict()
    data["schema"] = 99
    with pytest.raises(CalibrationError, match="schema"):
        load_artifact(write_artifact(tmp_path, data))


def test_unknown_method_fails(tmp_path):
    data = artifact_dict()
    data["method"] = "isotonic"
    with pytest.raises(CalibrationError, match="method"):
        load_artifact(write_artifact(tmp_path, data))


def test_missing_file_fails_with_path(tmp_path):
    with pytest.raises(CalibrationError, match="nope.yaml"):
        load_artifact(str(tmp_path / "nope.yaml"))


def test_invalid_yaml_fails(tmp_path):
    path = tmp_path / "calibration.yaml"
    path.write_text("schema: [unclosed")
    with pytest.raises(CalibrationError):
        load_artifact(str(path))


def test_matching_bindings_validate(tmp_path):
    artifact = load_artifact(write_artifact(tmp_path, artifact_dict()))
    validate_bindings(artifact, signal_config())  # must not raise


def test_revision_mismatch_names_field_and_both_values(tmp_path):
    artifact = load_artifact(write_artifact(tmp_path, artifact_dict()))
    with pytest.raises(CalibrationError) as excinfo:
        validate_bindings(artifact, signal_config(revision="cafebabe"))
    message = str(excinfo.value)
    assert "revision" in message
    assert "deadbeef" in message
    assert "cafebabe" in message


def test_repo_default_artifact_matches_the_repo_config():
    """The checked-in artifact must load and bind to the checked-in config — the
    exact validation the engine runs at startup."""
    from engine.config import load_config

    cfg = load_config("config/verifier.yaml")
    artifact = load_artifact(Path("config") / cfg.groundedness.calibration)
    validate_bindings(artifact, cfg.groundedness)
    assert artifact.coefficients.a > 0, "the fitted map must be increasing"
    assert 0 < artifact.metrics.in_domain.ece < artifact.metrics.out_of_domain.ece


def test_every_mismatch_is_named(tmp_path):
    data = artifact_dict()
    data["bindings"]["inference_mode"] = "stale-mode"
    data["bindings"]["claim_unit"] = "stale-unit"
    artifact = load_artifact(write_artifact(tmp_path, data))
    with pytest.raises(CalibrationError) as excinfo:
        validate_bindings(artifact, signal_config(model="other/model"))
    message = str(excinfo.value)
    assert "model" in message
    assert "inference_mode" in message
    assert "claim_unit" in message
    assert INFERENCE_MODE in message and "stale-mode" in message
    assert CLAIM_UNIT in message and "stale-unit" in message
