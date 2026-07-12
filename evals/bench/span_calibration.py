"""Span-map fitting and the pre-registered transfer decision (issue #40, design §3).

Both candidate maps run through the engine's own `span_confidence`, so fit-time
and serve-time arithmetic are the same code. The decision rule is fixed here,
before any fitting: the transferred claim coefficients ship iff their held-out
span ECE is within `TRANSFER_MARGIN` of the span-level fit's — ties go to
transfer, because fewer fitted parameters is the smaller trust surface."""

import random
from collections.abc import Sequence

from bench.calibration import PlattFit, fit_platt
from bench.metrics import ece
from engine.calibration import span_confidence

TRANSFER_MARGIN = 0.01
BOOTSTRAP_RESAMPLES = 1000
BOOTSTRAP_SEED = 18


def fit_span_platt(unsupported: Sequence[int], raw_risks: Sequence[float]) -> PlattFit:
    """Fit the span map at span level. `fit_platt` fits the support-side event, and
    the span map is its complement (one arithmetic path), so the fit runs on the
    span's support-analog 1 − r against the supported-region outcome 1 − label."""
    return fit_platt([1 - u for u in unsupported], [1.0 - r for r in raw_risks])


def span_confidences(fit: PlattFit, raw_risks: Sequence[float]) -> list[float]:
    """Map raw span risks through the engine's own span map."""
    return [span_confidence(r, a=fit.a, b=fit.b) for r in raw_risks]


def span_ece(unsupported: Sequence[int], confidences: Sequence[float]) -> float:
    """ECE in the unsupported direction — the event the span confidence asserts."""
    return ece(unsupported, confidences)


def decide_source(transfer_ece: float, fitted_ece: float) -> str:
    """The pre-registered rule: transfer iff within the margin of the fit."""
    return "transferred" if transfer_ece - fitted_ece <= TRANSFER_MARGIN else "fitted"


def bootstrap_ece_difference(
    unsupported: Sequence[int],
    transfer_confidences: Sequence[float],
    fitted_confidences: Sequence[float],
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float]:
    """95% CI on ECE(transfer) − ECE(fit) by paired bootstrap over spans, so the
    thinness of the held-out span population is visible next to the point rule."""
    rng = random.Random(seed)
    n = len(unsupported)
    differences: list[float] = []
    for _ in range(resamples):
        indices = [rng.randrange(n) for _ in range(n)]
        resampled = [unsupported[i] for i in indices]
        differences.append(
            ece(resampled, [transfer_confidences[i] for i in indices])
            - ece(resampled, [fitted_confidences[i] for i in indices])
        )
    differences.sort()
    return (
        differences[round(0.025 * (len(differences) - 1))],
        differences[round(0.975 * (len(differences) - 1))],
    )
