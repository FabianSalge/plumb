"""Versioned verifier config: which signal model runs, at which revision, with which thresholds."""

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, ValidationError


class ConfigError(Exception):
    """The verifier config is missing or invalid — the service must not start without one."""


class SignalModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    model: str
    revision: str
    # Thresholds the calibrated confidence, not the raw score (ADR-0008).
    threshold: float
    # Flags claim tokens into unsupported spans; independent from the verdict
    # threshold — an unsupported claim with zero spans is legal (ADR-0007).
    # Applies to raw token risks: span confidences are uncalibrated until #40.
    span_threshold: float
    # Path to the calibration artifact, relative to this config file's directory.
    # Required — an engine without a matching calibrator refuses to start (ADR-0008).
    calibration: str


class SignalsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    groundedness: SignalModelConfig


class RerankerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    model: str
    revision: str


class RetrievalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    # Sentences of preceding answer context joined into each claim's query
    # (plus the leading sentence) — deterministic expansion, no model (ADR-0010).
    expansion_window: int
    # Chunks recalled from the store per query before reranking (ADR-0002).
    recall_depth: int
    # Guaranteed scoring-window slots for each claim's top-reranked chunks.
    per_claim_quota: int
    # Pool budget in scoring-tokenizer tokens; pooling, not the scorer's
    # window, decides what is scored — truncation is logged, never silent.
    pool_budget_tokens: int
    reranker: RerankerConfig


class VerifierConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    version: str
    signals: SignalsConfig
    # Absent means a fast-only deployment: thorough requests are rejected.
    retrieval: RetrievalConfig | None = None

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
