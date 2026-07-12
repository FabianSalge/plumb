"""Unit tests for the Platt confidence map and its span-risk counterpart."""

import pytest

from engine.calibration import EPSILON, CalibrationError, platt_confidence, span_confidence


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


# --- Span map: confidence that a flagged region is unsupported -------------------


def test_span_map_identity_coefficients_return_the_raw_risk():
    # 1 − platt(1 − r) with a=1, b=0 is the identity on the risk
    assert span_confidence(0.3, a=1.0, b=0.0) == pytest.approx(0.3)
    assert span_confidence(0.75, a=1.0, b=0.0) == pytest.approx(0.75)


def test_span_map_is_strictly_monotone_increasing_in_risk():
    risks = [0.0, 0.1, 0.4, 0.5, 0.9, 1.0]
    confidences = [span_confidence(r, a=1.7, b=-0.4) for r in risks]
    assert all(c1 < c2 for c1, c2 in zip(confidences, confidences[1:], strict=False))


def test_span_map_saturated_risks_stay_strictly_inside_the_open_interval():
    low = span_confidence(0.0, a=1.0, b=0.0)
    high = span_confidence(1.0, a=1.0, b=0.0)
    assert 0.0 < low < high < 1.0


def test_span_map_is_one_minus_the_claim_map_on_the_support_side():
    # The one arithmetic path (design §2): the span map is the claim map applied
    # to the span's support-analog 1 − r, complemented back to the risk direction.
    for risk in (0.0, 0.2, 0.5, 0.8, 1.0):
        expected = 1.0 - platt_confidence(1.0 - risk, a=2.0, b=1.0)
        assert span_confidence(risk, a=2.0, b=1.0) == pytest.approx(expected)


def test_span_map_out_of_range_risk_fails_loudly():
    with pytest.raises(CalibrationError):
        span_confidence(1.2, a=1.0, b=0.0)
    with pytest.raises(CalibrationError):
        span_confidence(-0.1, a=1.0, b=0.0)


def test_span_map_non_finite_coefficients_fail_loudly():
    with pytest.raises(CalibrationError):
        span_confidence(0.5, a=float("nan"), b=0.0)
    with pytest.raises(CalibrationError):
        span_confidence(0.5, a=1.0, b=float("-inf"))
