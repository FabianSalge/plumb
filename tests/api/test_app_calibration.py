"""Startup refusal: the app must not come up through a missing or mismatched
calibrator — no raw-score fallback, no degraded mode (calibration spec)."""

import pytest
import yaml

from api.app import create_app
from engine.calibration import CalibrationError
from engine.signals import TokenScores
from tests.engine.factories import make_artifact, make_config, write_config
from tests.engine.signals.fakes import FakeScorer


def scorer_factory(cfg):
    return FakeScorer(TokenScores(probs=[0.1], offsets=[(0, 1)]))


def test_mismatched_revision_refuses_startup(tmp_path):
    config = make_config()
    config["signals"]["groundedness"]["revision"] = "cafebabe"
    path = write_config(tmp_path, config=config, artifact=make_artifact())
    with pytest.raises(CalibrationError) as excinfo:
        create_app(config_path=path, scorer_factory=scorer_factory)
    message = str(excinfo.value)
    assert "revision" in message
    assert "cafebabe" in message and "deadbeef" in message


def test_mismatched_claim_unit_refuses_startup(tmp_path):
    path = write_config(tmp_path, artifact=make_artifact(claim_unit="stale-unit"))
    with pytest.raises(CalibrationError, match="claim_unit"):
        create_app(config_path=path, scorer_factory=scorer_factory)


def test_mismatched_span_threshold_refuses_startup(tmp_path):
    """The span map was fitted on spans flagged at one threshold; a config flagging
    at another produces a span population the calibrator never saw."""
    path = write_config(tmp_path, artifact=make_artifact(span_threshold=0.9))
    with pytest.raises(CalibrationError) as excinfo:
        create_app(config_path=path, scorer_factory=scorer_factory)
    message = str(excinfo.value)
    assert "span_threshold" in message
    assert "0.9" in message and "0.5" in message


def test_missing_artifact_file_refuses_startup(tmp_path):
    path = tmp_path / "verifier.yaml"
    path.write_text(yaml.safe_dump(make_config()))  # config only — no artifact beside it
    with pytest.raises(CalibrationError, match="calibration.yaml"):
        create_app(config_path=path, scorer_factory=scorer_factory)


def test_artifact_path_resolves_relative_to_the_config_file(tmp_path):
    """The artifact path in config is relative to the config file's directory,
    not the process working directory."""
    nested = tmp_path / "etc" / "plumb"
    nested.mkdir(parents=True)
    path = write_config(nested)
    app = create_app(config_path=path, scorer_factory=scorer_factory)
    assert app is not None
