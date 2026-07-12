"""Unit tests for the span-map fit and the pre-registered transfer decision — pure
math, no models."""

import pytest
from bench.calibration import PlattFit
from bench.span_calibration import (
    TRANSFER_MARGIN,
    bootstrap_ece_difference,
    decide_source,
    fit_span_platt,
    span_confidences,
    span_ece,
)

from engine.calibration import span_confidence


def synthetic(a: float, b: float, per_risk: int = 1000) -> tuple[list[int], list[float]]:
    """Exact-proportion span data drawn from a known span map: for each raw risk,
    the unsupported counts match the mapped probability to rounding, so the MLE
    must recover (a, b) almost exactly. Deterministic — no RNG."""
    unsupported: list[int] = []
    risks: list[float] = []
    for risk in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
        probability = span_confidence(risk, a=a, b=b)
        positives = round(per_risk * probability)
        unsupported.extend([1] * positives + [0] * (per_risk - positives))
        risks.extend([risk] * per_risk)
    return unsupported, risks


def test_fit_recovers_identity():
    unsupported, risks = synthetic(a=1.0, b=0.0)
    fit = fit_span_platt(unsupported, risks)
    assert fit.a == pytest.approx(1.0, abs=0.05)
    assert fit.b == pytest.approx(0.0, abs=0.05)


def test_fit_recovers_shifted_steeper_map():
    unsupported, risks = synthetic(a=1.6, b=-0.7)
    fit = fit_span_platt(unsupported, risks)
    assert fit.a == pytest.approx(1.6, abs=0.1)
    assert fit.b == pytest.approx(-0.7, abs=0.1)


def test_span_confidences_apply_the_engine_map():
    fit = PlattFit(a=1.3, b=0.4)
    risks = [0.5, 0.8, 1.0]
    assert span_confidences(fit, risks) == [
        pytest.approx(span_confidence(r, a=1.3, b=0.4)) for r in risks
    ]


def test_span_ece_is_measured_in_the_unsupported_direction():
    # Perfectly calibrated in the unsupported direction: confidence 0.75 on a
    # population that is 75% unsupported.
    unsupported = [1, 1, 1, 0]
    confidences = [0.75] * 4
    assert span_ece(unsupported, confidences) == pytest.approx(0.0)
    assert span_ece([0, 0, 0, 1], confidences) == pytest.approx(0.5)


def test_transfer_wins_within_the_margin_including_ties():
    assert decide_source(transfer_ece=0.020, fitted_ece=0.020) == "transferred"
    assert decide_source(transfer_ece=0.030, fitted_ece=0.020 + 1e-12) == "transferred"
    assert decide_source(transfer_ece=0.015, fitted_ece=0.020) == "transferred"


def test_fit_wins_beyond_the_margin():
    assert decide_source(transfer_ece=0.020 + TRANSFER_MARGIN + 1e-6, fitted_ece=0.020) == "fitted"


def test_bootstrap_ci_is_deterministic_and_ordered():
    unsupported, risks = synthetic(a=1.0, b=0.0, per_risk=50)
    transfer = span_confidences(PlattFit(a=1.0, b=0.0), risks)
    fitted = span_confidences(PlattFit(a=1.2, b=0.1), risks)
    lo, hi = bootstrap_ece_difference(unsupported, transfer, fitted)
    again = bootstrap_ece_difference(unsupported, transfer, fitted)
    assert (lo, hi) == again
    assert lo <= hi


def test_bootstrap_ci_brackets_the_point_difference():
    unsupported, risks = synthetic(a=1.0, b=0.0, per_risk=50)
    transfer = span_confidences(PlattFit(a=1.0, b=0.0), risks)
    fitted = span_confidences(PlattFit(a=1.1, b=0.05), risks)
    point = span_ece(unsupported, transfer) - span_ece(unsupported, fitted)
    lo, hi = bootstrap_ece_difference(unsupported, transfer, fitted)
    assert lo <= point <= hi
