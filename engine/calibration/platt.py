"""The Platt map: confidence = sigmoid(a · logit(s) + b) with an ε-clamped logit.

The clamp is part of the artifact schema semantics, not a tunable: raw supports
can round to exactly 0.0 or 1.0 in float, the logit must not blow up, and the
output must never claim exact certainty. The evals fit (`evals/bench/`) imports
this exact map, so fit-time and serve-time arithmetic cannot diverge.
"""

import math

EPSILON = 1e-6


class CalibrationError(Exception):
    """The calibrator cannot honestly map this input — refuse rather than guess."""


def platt_confidence(support: float, a: float, b: float) -> float:
    """Calibrated probability that a claim with raw support `support` is supported."""
    if not math.isfinite(a) or not math.isfinite(b):
        raise CalibrationError(f"non-finite Platt coefficients: a={a}, b={b}")
    if not 0.0 <= support <= 1.0:
        raise CalibrationError(f"raw support {support} outside [0, 1]")
    clamped = min(max(support, EPSILON), 1.0 - EPSILON)
    logit = math.log(clamped / (1.0 - clamped))
    return 1.0 / (1.0 + math.exp(-(a * logit + b)))


def span_confidence(raw_risk: float, a: float, b: float) -> float:
    """Calibrated probability that a span flagged at max token risk `raw_risk` marks
    a genuinely unsupported region: the claim map applied to the span's support-analog
    1 − r, complemented back to the risk direction — one arithmetic path for both."""
    return 1.0 - platt_confidence(1.0 - raw_risk, a=a, b=b)
