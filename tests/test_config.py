"""Unit tests for the versioned verifier config loader."""

import pytest
import yaml

from engine.config import ConfigError, load_config

VALID = {
    "version": "test-1",
    "signals": {
        "groundedness": {
            "model": "fake/model",
            "revision": "deadbeef",
            "threshold": 0.5,
        }
    },
}


def write(tmp_path, data) -> str:
    path = tmp_path / "verifier.yaml"
    path.write_text(yaml.safe_dump(data))
    return str(path)


def without(data: dict, dotted: str) -> dict:
    copy = yaml.safe_load(yaml.safe_dump(data))
    *parents, leaf = dotted.split(".")
    node = copy
    for key in parents:
        node = node[key]
    del node[leaf]
    return copy


def test_valid_config_loads(tmp_path):
    cfg = load_config(write(tmp_path, VALID))
    assert cfg.version == "test-1"
    assert cfg.groundedness.model == "fake/model"
    assert cfg.groundedness.revision == "deadbeef"
    assert cfg.groundedness.threshold == 0.5


def test_missing_file_fails_loudly(tmp_path):
    missing = tmp_path / "nope.yaml"
    with pytest.raises(ConfigError, match="nope.yaml"):
        load_config(str(missing))


@pytest.mark.parametrize(
    "field",
    [
        "version",
        "signals.groundedness.model",
        "signals.groundedness.revision",
        "signals.groundedness.threshold",
    ],
)
def test_missing_field_fails_loudly_naming_the_field(tmp_path, field):
    with pytest.raises(ConfigError, match=field.rsplit(".", 1)[-1]):
        load_config(write(tmp_path, without(VALID, field)))


def test_non_numeric_threshold_fails(tmp_path):
    broken = yaml.safe_load(yaml.safe_dump(VALID))
    broken["signals"]["groundedness"]["threshold"] = "high"
    with pytest.raises(ConfigError):
        load_config(write(tmp_path, broken))


def test_invalid_yaml_fails(tmp_path):
    path = tmp_path / "verifier.yaml"
    path.write_text("version: [unclosed")
    with pytest.raises(ConfigError):
        load_config(str(path))


def test_repo_default_config_is_valid_and_pins_a_revision():
    """The checked-in config must always load and carry a real revision pin."""
    cfg = load_config("config/verifier.yaml")
    assert cfg.version
    assert cfg.groundedness.model == "KRLabsOrg/lettucedect-v2-mmbert-base"
    assert len(cfg.groundedness.revision) == 40, "expected a full git revision hash"
