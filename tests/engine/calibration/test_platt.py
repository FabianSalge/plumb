"""Unit tests for the Platt confidence map."""

import pytest

from engine.calibration import EPSILON, CalibrationError, platt_confidence


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
