"""Versioned verifier config: which signal model runs, at which revision, with which threshold."""

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, ValidationError


class ConfigError(Exception):
    """The verifier config is missing or invalid — the service must not start without one."""


class SignalModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    model: str
    revision: str
    threshold: float


class SignalsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    groundedness: SignalModelConfig


class VerifierConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    version: str
    signals: SignalsConfig

    @property
    def groundedness(self) -> SignalModelConfig:
        return self.signals.groundedness


def load_config(path: str | Path) -> VerifierConfig:
    resolved = Path(path)
    try:
        raw = resolved.read_text()
    except OSError as exc:
        raise ConfigError(f"cannot read verifier config {resolved}: {exc}") from exc
    try:
        data = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in verifier config {resolved}: {exc}") from exc
    try:
        return VerifierConfig.model_validate(data)
    except ValidationError as exc:
        raise ConfigError(f"invalid verifier config {resolved}: {exc}") from exc
