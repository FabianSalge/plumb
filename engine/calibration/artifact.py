"""The versioned calibration artifact: what it records, and the binding validation
that refuses to serve a calibrator fitted against a different model, revision,
inference mode, or claim unit (ADR-0008). Fail loudly, never fall back to raw."""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from engine.calibration.platt import CalibrationError, platt_confidence
from engine.config import SignalModelConfig
from engine.decomposition import CLAIM_UNIT
from engine.signals.groundedness import INFERENCE_MODE

# Artifact schema versions this engine can serve. The ε clamp and the meaning of
# the coefficients are pinned to the schema version, so an unknown version is a
# refusal, not a best-effort parse.
KNOWN_SCHEMAS = frozenset({1})


class Coefficients(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    a: float
    b: float


class Bindings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    model: str
    revision: str
    inference_mode: str
    claim_unit: str


class FitIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset: str
    exclusion: str
    responses: int
    sentences: int
    sha256: str
    fitted_at: str


class InDomainMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset: str
    slice: str
    sentences: int
    ece: float


class OutOfDomainMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset: str
    subsets: list[str]
    excluded_subsets: dict[str, str]
    claims: int
    ece: float


class Metrics(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    in_domain: InDomainMetrics
    out_of_domain: OutOfDomainMetrics


class CalibrationArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    artifact_schema: int = Field(alias="schema")
    method: Literal["platt"]
    coefficients: Coefficients
    bindings: Bindings
    fit: FitIdentity
    metrics: Metrics

    def confidence(self, support: float) -> float:
        """Calibrated probability of support for one claim's raw support score."""
        return platt_confidence(support, a=self.coefficients.a, b=self.coefficients.b)


def load_artifact(path: str | Path) -> CalibrationArtifact:
    resolved = Path(path)
    try:
        raw = resolved.read_text()
    except OSError as exc:
        raise CalibrationError(f"cannot read calibration artifact {resolved}: {exc}") from exc
    try:
        data = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        raise CalibrationError(f"invalid YAML in calibration artifact {resolved}: {exc}") from exc
    try:
        artifact = CalibrationArtifact.model_validate(data)
    except ValidationError as exc:
        raise CalibrationError(f"invalid calibration artifact {resolved}: {exc}") from exc
    if artifact.artifact_schema not in KNOWN_SCHEMAS:
        raise CalibrationError(
            f"calibration artifact {resolved} carries unknown schema version "
            f"{artifact.artifact_schema}; this engine serves {sorted(KNOWN_SCHEMAS)}"
        )
    return artifact


def validate_bindings(artifact: CalibrationArtifact, cfg: SignalModelConfig) -> None:
    """Refuse a calibrator fitted against anything but the running configuration.
    Every mismatched binding is named with its expected and found values."""
    expected = {
        "model": cfg.model,
        "revision": cfg.revision,
        "inference_mode": INFERENCE_MODE,
        "claim_unit": CLAIM_UNIT,
    }
    found = {
        "model": artifact.bindings.model,
        "revision": artifact.bindings.revision,
        "inference_mode": artifact.bindings.inference_mode,
        "claim_unit": artifact.bindings.claim_unit,
    }
    mismatches = [
        f"{name}: running config expects {expected[name]!r}, artifact was fitted "
        f"against {found[name]!r}"
        for name in expected
        if expected[name] != found[name]
    ]
    if mismatches:
        raise CalibrationError(
            "calibration artifact bindings do not match the running engine — refusing "
            "to serve scores through a mismatched calibrator; refit per ADR-0008. "
            + "; ".join(mismatches)
        )
