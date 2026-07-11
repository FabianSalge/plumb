"""Unit tests for sentence-level labelling — pure overlap logic, no model."""

from bench.sentence import sentence_hallucinated


def test_sentence_overlapping_a_span_is_hallucinated():
    # span [10, 20); a sentence touching any part of it is positive
    assert sentence_hallucinated(0, 15, [(10, 20)])
    assert sentence_hallucinated(15, 30, [(10, 20)])
    assert sentence_hallucinated(12, 18, [(10, 20)])  # sentence inside the span


def test_sentence_not_touching_any_span_is_supported():
    assert not sentence_hallucinated(0, 10, [(10, 20)])  # abuts, no overlap
    assert not sentence_hallucinated(20, 30, [(10, 20)])  # abuts on the right
    assert not sentence_hallucinated(0, 30, [])  # no spans at all


def test_any_overlapping_span_flags_the_sentence():
    assert sentence_hallucinated(0, 12, [(50, 60), (10, 20)])
