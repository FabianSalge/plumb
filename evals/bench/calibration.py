"""Platt-scaling fit (ADR-0008): two-parameter logistic MLE on the ε-clamped
logit of the raw support score, implemented directly so it is unit-testable and
free of silent library defaults. The map being fitted is the engine's own
`platt_confidence` — fit-time and serve-time arithmetic are the same code.

Plain MLE by Newton–Raphson on the log-loss; at ~15k sentences and two
parameters, Platt's target-smoothing refinement buys nothing (design §3).
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass

from engine.calibration import EPSILON, platt_confidence

_MAX_ITERATIONS = 100
_GRADIENT_TOLERANCE = 1e-9


class FitError(Exception):
    """The fit is undefined or did not converge — refuse rather than ship a bad map."""


@dataclass(frozen=True)
class PlattFit:
    a: float
    b: float


def _clamped_logit(support: float) -> float:
    if not 0.0 <= support <= 1.0:
        raise FitError(f"raw support {support} outside [0, 1]")
    clamped = min(max(support, EPSILON), 1.0 - EPSILON)
    return math.log(clamped / (1.0 - clamped))


def fit_platt(outcomes: Sequence[int], supports: Sequence[float]) -> PlattFit:
    """Fit `confidence = sigmoid(a · logit(s) + b)` by maximum likelihood.

    `outcomes` are 1 iff the claim is supported — the event the confidence asserts.
    """
    if len(outcomes) != len(supports):
        raise FitError(f"length mismatch: {len(outcomes)} outcomes, {len(supports)} supports")
    if not outcomes:
        raise FitError("fit undefined: no examples")
    if not any(outcomes) or all(outcomes):
        raise FitError("fit undefined: outcomes contain a single class")

    logits = [_clamped_logit(support) for support in supports]
    a, b = 1.0, 0.0
    for _ in range(_MAX_ITERATIONS):
        # Gradient and Hessian of the negative log-likelihood in (a, b).
        grad_a = grad_b = 0.0
        h_aa = h_ab = h_bb = 0.0
        for outcome, logit in zip(outcomes, logits, strict=True):
            predicted = 1.0 / (1.0 + math.exp(-(a * logit + b)))
            residual = predicted - outcome
            weight = predicted * (1.0 - predicted)
            grad_a += residual * logit
            grad_b += residual
            h_aa += weight * logit * logit
            h_ab += weight * logit
            h_bb += weight

        if max(abs(grad_a), abs(grad_b)) < _GRADIENT_TOLERANCE * len(outcomes):
            return PlattFit(a=a, b=b)

        determinant = h_aa * h_bb - h_ab * h_ab
        if determinant <= 0.0 or not math.isfinite(determinant):
            raise FitError(f"singular Hessian at a={a}, b={b} — the fit cannot proceed")
        a -= (h_bb * grad_a - h_ab * grad_b) / determinant
        b -= (h_aa * grad_b - h_ab * grad_a) / determinant

    raise FitError(f"Newton–Raphson did not converge in {_MAX_ITERATIONS} iterations")


def apply_fit(fit: PlattFit, supports: Sequence[float]) -> list[float]:
    """Map raw supports through the fitted calibrator — the engine's own map."""
    return [platt_confidence(support, a=fit.a, b=fit.b) for support in supports]
