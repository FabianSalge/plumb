"""Score calibration (ADR-0008): the Platt map from raw support to calibrated
confidence, and the versioned artifact that binds it to the model, inference
mode, and claim unit it was fitted against."""

from engine.calibration.artifact import (
    CalibrationArtifact,
    load_artifact,
    validate_bindings,
)
from engine.calibration.platt import EPSILON, CalibrationError, platt_confidence

__all__ = [
    "EPSILON",
    "CalibrationArtifact",
    "CalibrationError",
    "load_artifact",
    "platt_confidence",
    "validate_bindings",
]
