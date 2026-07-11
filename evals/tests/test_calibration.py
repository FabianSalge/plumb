"""Unit tests for the Platt fit — pure math, no models."""

import pytest
from bench.calibration import FitError, fit_platt

from engine.calibration import platt_confidence


def synthetic(a: float, b: float, per_score: int = 1000) -> tuple[list[int], list[float]]:
    """Exact-proportion data drawn from a known Platt map: for each score, the
    outcome counts match the mapped probability to rounding, so the MLE must
    recover (a, b) almost exactly. Deterministic — no RNG."""
    outcomes: list[int] = []
    supports: list[float] = []
    for score in (0.05, 0.2, 0.35, 0.5, 0.65, 0.8, 0.95):
        probability = platt_confidence(score, a=a, b=b)
        positives = round(per_score * probability)
        outcomes.extend([1] * positives + [0] * (per_score - positives))
        supports.extend([score] * per_score)
    return outcomes, supports


def test_fit_recovers_identity():
    outcomes, supports = synthetic(a=1.0, b=0.0)
    fit = fit_platt(outcomes, supports)
    assert fit.a == pytest.approx(1.0, abs=0.02)
    assert fit.b == pytest.approx(0.0, abs=0.02)


def test_fit_recovers_shifted_steeper_map():
    outcomes, supports = synthetic(a=1.6, b=-0.7)
    fit = fit_platt(outcomes, supports)
    assert fit.a == pytest.approx(1.6, abs=0.05)
    assert fit.b == pytest.approx(-0.7, abs=0.05)


def test_fit_handles_saturated_supports():
    outcomes, supports = synthetic(a=1.0, b=0.0)
    outcomes += [1, 0]
    supports += [1.0, 0.0]  # must clamp, not blow up
    fit = fit_platt(outcomes, supports)
    assert fit.a == pytest.approx(1.0, abs=0.05)


def test_fit_refuses_single_class():
    with pytest.raises(FitError):
        fit_platt([1, 1, 1], [0.2, 0.5, 0.8])
    with pytest.raises(FitError):
        fit_platt([0, 0, 0], [0.2, 0.5, 0.8])


def test_fit_refuses_empty_and_mismatched_input():
    with pytest.raises(FitError):
        fit_platt([], [])
    with pytest.raises(FitError):
        fit_platt([1, 0], [0.5])


def test_fit_refuses_out_of_range_support():
    with pytest.raises(FitError):
        fit_platt([1, 0], [0.5, 1.4])
