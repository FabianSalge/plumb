"""Classification metrics for the benchmark, implemented directly so they are
unit-testable and free of silent library defaults.

Convention: label 1 = hallucinated (the positive class), risk scores are
higher-is-more-hallucinated. Adapters return support scores; callers pass
risk = 1 - support.
"""

from collections.abc import Sequence


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
