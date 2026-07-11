"""Classification metrics for the benchmark, implemented directly so they are
unit-testable and free of silent library defaults.

Convention: label 1 = hallucinated (the positive class), risk scores are
higher-is-more-hallucinated. Adapters return support scores; callers pass
risk = 1 - support.

Calibration metrics use the opposite framing: `outcomes` are 1 iff the event the
confidence asserts occurred (for the groundedness calibrator, 1 = supported), so
a calibrated confidence c claims a fraction c of claims scored c are supported.
"""

from collections.abc import Sequence
from dataclasses import dataclass

RELIABILITY_BINS = 10


class MetricError(Exception):
    """The metric is undefined for the given inputs — refuse rather than guess."""


def _check_binary(labels: Sequence[int]) -> None:
    if not any(labels) or all(labels):
        raise MetricError("metric undefined: labels contain a single class")


def auroc(labels: Sequence[int], risk_scores: Sequence[float]) -> float:
    """Area under the ROC curve via the Mann-Whitney U statistic with midranks."""
    if len(labels) != len(risk_scores):
        raise MetricError(f"length mismatch: {len(labels)} labels, {len(risk_scores)} scores")
    _check_binary(labels)

    order = sorted(range(len(risk_scores)), key=lambda i: risk_scores[i])
    ranks = [0.0] * len(order)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and risk_scores[order[j + 1]] == risk_scores[order[i]]:
            j += 1
        midrank = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = midrank
        i = j + 1

    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    rank_sum_pos = sum(rank for rank, label in zip(ranks, labels, strict=True) if label)
    u_statistic = rank_sum_pos - n_pos * (n_pos + 1) / 2
    return u_statistic / (n_pos * n_neg)


def _confusion(labels: Sequence[int], predicted: Sequence[bool]) -> tuple[int, int, int, int]:
    if len(labels) != len(predicted):
        raise MetricError(f"length mismatch: {len(labels)} labels, {len(predicted)} predictions")
    tp = sum(1 for label, pred in zip(labels, predicted, strict=True) if label and pred)
    fp = sum(1 for label, pred in zip(labels, predicted, strict=True) if not label and pred)
    fn = sum(1 for label, pred in zip(labels, predicted, strict=True) if label and not pred)
    tn = sum(1 for label, pred in zip(labels, predicted, strict=True) if not label and not pred)
    return tp, fp, fn, tn


def balanced_accuracy(labels: Sequence[int], predicted: Sequence[bool]) -> float:
    """Mean of sensitivity and specificity — robust to the 35/65 class skew in RAGTruth."""
    _check_binary(labels)
    tp, fp, fn, tn = _confusion(labels, predicted)
    return (tp / (tp + fn) + tn / (tn + fp)) / 2


def f1_score(labels: Sequence[int], predicted: Sequence[bool]) -> float:
    """F1 for the hallucinated class."""
    tp, fp, fn, _ = _confusion(labels, predicted)
    if tp == 0:
        return 0.0
    return 2 * tp / (2 * tp + fp + fn)


def precision_recall(labels: Sequence[int], predicted: Sequence[bool]) -> tuple[float, float]:
    """Precision and recall for the hallucinated class."""
    tp, fp, fn, _ = _confusion(labels, predicted)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return precision, recall


@dataclass(frozen=True)
class ReliabilityBin:
    """One reliability-diagram bin: `[lo, hi)` (the last bin closes at 1.0 inclusive).
    `mean_confidence` and `outcome_rate` are None for an empty bin."""

    lo: float
    hi: float
    count: int
    mean_confidence: float | None
    outcome_rate: float | None


def _check_confidences(outcomes: Sequence[int], confidences: Sequence[float]) -> None:
    if len(outcomes) != len(confidences):
        raise MetricError(
            f"length mismatch: {len(outcomes)} outcomes, {len(confidences)} confidences"
        )
    if not confidences:
        raise MetricError("metric undefined: no confidences")
    for confidence in confidences:
        if not 0.0 <= confidence <= 1.0:
            raise MetricError(f"confidence {confidence} outside [0, 1]")


def reliability_bins(outcomes: Sequence[int], confidences: Sequence[float]) -> list[ReliabilityBin]:
    """Reliability-diagram data over equal-width confidence bins: per bin, how often
    the asserted event actually occurred versus the mean confidence claimed."""
    _check_confidences(outcomes, confidences)
    binned: list[list[tuple[int, float]]] = [[] for _ in range(RELIABILITY_BINS)]
    for outcome, confidence in zip(outcomes, confidences, strict=True):
        index = min(int(confidence * RELIABILITY_BINS), RELIABILITY_BINS - 1)
        binned[index].append((outcome, confidence))
    bins: list[ReliabilityBin] = []
    for i, members in enumerate(binned):
        if members:
            mean_confidence = sum(c for _, c in members) / len(members)
            outcome_rate = sum(o for o, _ in members) / len(members)
        else:
            mean_confidence = None
            outcome_rate = None
        bins.append(
            ReliabilityBin(
                lo=i / RELIABILITY_BINS,
                hi=(i + 1) / RELIABILITY_BINS,
                count=len(members),
                mean_confidence=mean_confidence,
                outcome_rate=outcome_rate,
            )
        )
    return bins


def ece(outcomes: Sequence[int], confidences: Sequence[float]) -> float:
    """Expected calibration error: the count-weighted mean absolute gap between
    each bin's mean confidence and its observed outcome rate."""
    bins = reliability_bins(outcomes, confidences)
    total = len(confidences)
    return sum(
        bin.count / total * abs(bin.outcome_rate - bin.mean_confidence)
        for bin in bins
        if bin.count and bin.outcome_rate is not None and bin.mean_confidence is not None
    )
