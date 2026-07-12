"""The versioned calibration artifact: what it records, and the binding validation
that refuses to serve a calibrator fitted against a different model, revision,
inference mode, claim unit, or span-flagging threshold (ADR-0008, #40). Fail
loudly, never fall back to raw."""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from engine.calibration.platt import CalibrationError, platt_confidence, span_confidence
from engine.config import SignalModelConfig
from engine.decomposition import CLAIM_UNIT
from engine.signals.groundedness import INFERENCE_MODE

# Artifact schema versions this engine can serve. The ε clamp and the meaning of
# the coefficients are pinned to the schema version, so an unknown version is a
# refusal, not a best-effort parse. Schema 1 (no span calibration) is refused:
# an engine that ships span confidences has no span-uncalibrated mode.
KNOWN_SCHEMAS = frozenset({2})


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


class SpanFitIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    # Whether the served span coefficients were transferred from the claim map or
    # fitted at span level — decided by the pre-registered rule, never assumed.
    source: Literal["transferred", "fitted"]
    dataset: str
    label_convention: str
    exclusion: str
    spans: int
    sha256: str
    fitted_at: str


class SpanInDomainMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset: str
    slice: str
    spans: int
    ece: float


class SpanOutOfDomainMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    # No span-annotated out-of-domain dataset exists; the absence is recorded with
    # its reason rather than proxied by a claim-level number. A schema bump adds
    # real fields when such data does.
    measured: Literal[False]
    reason: str


class SpanMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    in_domain: SpanInDomainMetrics
    out_of_domain: SpanOutOfDomainMetrics


class SpanCalibration(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    coefficients: Coefficients
    # The flagging threshold the span population was derived at — a binding: spans
    # exist only where risk ≥ threshold, so a different threshold produces a
    # population this map never saw.
    span_threshold: float
    fit: SpanFitIdentity
    metrics: SpanMetrics


class CalibrationArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    artifact_schema: int = Field(alias="schema")
    method: Literal["platt"]
    coefficients: Coefficients
    bindings: Bindings
    fit: FitIdentity
    metrics: Metrics
    span: SpanCalibration

    def confidence(self, support: float) -> float:
        """Calibrated probability of support for one claim's raw support score."""
        return platt_confidence(support, a=self.coefficients.a, b=self.coefficients.b)

    def span_confidence(self, raw_risk: float) -> float:
        """Calibrated probability that a span flagged at `raw_risk` is unsupported."""
        return span_confidence(raw_risk, a=self.span.coefficients.a, b=self.span.coefficients.b)


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
    # The schema version gates field validation: a schema-1 artifact is refused by
    # version, not by complaints about span fields it was never meant to carry.
    schema = data.get("schema") if isinstance(data, dict) else None
    if schema not in KNOWN_SCHEMAS:
        raise CalibrationError(
            f"calibration artifact {resolved} carries schema version "
            f"{schema!r}; this engine serves {sorted(KNOWN_SCHEMAS)}"
        )
    try:
        return CalibrationArtifact.model_validate(data)
    except ValidationError as exc:
        raise CalibrationError(f"invalid calibration artifact {resolved}: {exc}") from exc


def validate_bindings(artifact: CalibrationArtifact, cfg: SignalModelConfig) -> None:
    """Refuse a calibrator fitted against anything but the running configuration.
    Every mismatched binding is named with its expected and found values."""
    expected: dict[str, str | float] = {
        "model": cfg.model,
        "revision": cfg.revision,
        "inference_mode": INFERENCE_MODE,
        "claim_unit": CLAIM_UNIT,
        "span_threshold": cfg.span_threshold,
    }
    found: dict[str, str | float] = {
        "model": artifact.bindings.model,
        "revision": artifact.bindings.revision,
        "inference_mode": artifact.bindings.inference_mode,
        "claim_unit": artifact.bindings.claim_unit,
        "span_threshold": artifact.span.span_threshold,
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
