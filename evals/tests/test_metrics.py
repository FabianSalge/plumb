"""Unit tests for the benchmark metrics — pure functions, no models."""

import pytest
from bench.metrics import MetricError, auroc, balanced_accuracy, f1_score


def test_auroc_perfect_separation():
    # labels: 1 = hallucinated; scores: higher = more likely hallucinated (risk)
    assert auroc([1, 1, 0, 0], [0.9, 0.8, 0.2, 0.1]) == 1.0


def test_auroc_inverted_separation_is_zero():
    assert auroc([1, 1, 0, 0], [0.1, 0.2, 0.8, 0.9]) == 0.0


def test_auroc_random_is_half():
    assert auroc([1, 0], [0.5, 0.5]) == 0.5


def test_auroc_ties_use_midrank():
    # one tie across classes counts half
    assert auroc([1, 1, 0, 0], [0.9, 0.5, 0.5, 0.1]) == pytest.approx(0.875)


def test_auroc_single_class_fails_loudly():
    with pytest.raises(MetricError):
        auroc([1, 1], [0.5, 0.6])


def test_balanced_accuracy():
    # sensitivity 1.0, specificity 0.5 -> 0.75
    assert balanced_accuracy([1, 1, 0, 0], [True, True, True, False]) == pytest.approx(0.75)


def test_balanced_accuracy_single_class_fails_loudly():
    with pytest.raises(MetricError):
        balanced_accuracy([0, 0], [False, False])


def test_f1_score_positive_class():
    # preds: TP=1, FP=1, FN=1 -> precision 0.5, recall 0.5, f1 0.5
    assert f1_score([1, 0, 1, 0], [True, True, False, False]) == pytest.approx(0.5)


def test_f1_score_no_predictions_is_zero():
    assert f1_score([1, 0], [False, False]) == 0.0
