"""Unit tests for the benchmark metrics — pure functions, no models."""

import pytest
from bench.metrics import (
    MetricError,
    auroc,
    balanced_accuracy,
    ece,
    f1_score,
    reliability_bins,
)


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


# --- Calibration metrics: outcome 1 = the event the confidence asserts ---


def test_ece_perfectly_calibrated_bin_is_zero():
    # all confidences in one bin, empirical rate equals mean confidence
    assert ece([1, 0, 0, 0], [0.25, 0.25, 0.25, 0.25]) == pytest.approx(0.0)


def test_ece_overconfident_gap():
    # mean confidence 0.95, empirical rate 0.5 -> gap 0.45
    assert ece([1, 0], [0.95, 0.95]) == pytest.approx(0.45)


def test_ece_weights_bins_by_count():
    # bin [0.9, 1.0]: 3 points, mean conf 0.95, rate 1.0 -> gap 0.05, weight 3/4
    # bin [0.0, 0.1): 1 point, conf 0.05, rate 0.0 -> gap 0.05, weight 1/4
    outcomes = [1, 1, 1, 0]
    confidences = [0.95, 0.95, 0.95, 0.05]
    assert ece(outcomes, confidences) == pytest.approx(0.05)


def test_ece_empty_fails_loudly():
    with pytest.raises(MetricError):
        ece([], [])


def test_ece_out_of_range_confidence_fails_loudly():
    with pytest.raises(MetricError):
        ece([1], [1.2])
    with pytest.raises(MetricError):
        ece([0], [-0.1])


def test_ece_length_mismatch_fails_loudly():
    with pytest.raises(MetricError):
        ece([1, 0], [0.5])


def test_reliability_bins_are_ten_equal_width():
    bins = reliability_bins([1, 0], [0.05, 0.95])
    assert len(bins) == 10
    assert bins[0].lo == pytest.approx(0.0)
    assert bins[-1].hi == pytest.approx(1.0)
    assert all(b.hi - b.lo == pytest.approx(0.1) for b in bins)


def test_reliability_bins_counts_means_and_rates():
    outcomes = [1, 0, 1, 1]
    confidences = [0.62, 0.68, 0.65, 0.91]
    bins = reliability_bins(outcomes, confidences)
    sixth = bins[6]  # [0.6, 0.7)
    assert sixth.count == 3
    assert sixth.mean_confidence == pytest.approx(0.65)
    assert sixth.outcome_rate == pytest.approx(2 / 3)
    ninth = bins[9]  # [0.9, 1.0]
    assert ninth.count == 1
    assert ninth.outcome_rate == pytest.approx(1.0)


def test_reliability_bins_empty_bin_has_no_rates():
    bins = reliability_bins([1], [0.95])
    empty = bins[0]
    assert empty.count == 0
    assert empty.mean_confidence is None
    assert empty.outcome_rate is None


def test_reliability_bins_boundary_confidences_land_inside():
    # 0.0 belongs to the first bin, 1.0 to the last — nothing falls off the edges
    bins = reliability_bins([0, 1], [0.0, 1.0])
    assert bins[0].count == 1
    assert bins[9].count == 1
    assert sum(b.count for b in bins) == 2


def test_ece_matches_reliability_bins():
    outcomes = [1, 0, 1, 1, 0, 1]
    confidences = [0.15, 0.15, 0.85, 0.85, 0.85, 0.55]
    bins = reliability_bins(outcomes, confidences)
    expected = sum(b.count / 6 * abs(b.outcome_rate - b.mean_confidence) for b in bins if b.count)
    assert ece(outcomes, confidences) == pytest.approx(expected)
