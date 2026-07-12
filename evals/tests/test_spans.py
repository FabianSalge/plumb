"""Unit tests for span-level labelling — pure overlap logic, no model."""

import pytest
from bench.data import Example
from bench.spans import labeled_spans, span_fully_covered, span_unsupported

from engine.signals import TokenScores


def test_span_overlapping_an_annotation_is_unsupported():
    # annotation [10, 20); a span touching any part of it is positive
    assert span_unsupported(0, 15, [(10, 20)])
    assert span_unsupported(15, 30, [(10, 20)])
    assert span_unsupported(12, 18, [(10, 20)])  # span inside the annotation


def test_span_not_touching_any_annotation_is_supported():
    assert not span_unsupported(0, 10, [(10, 20)])  # abuts, no overlap
    assert not span_unsupported(20, 30, [(10, 20)])  # abuts on the right
    assert not span_unsupported(0, 30, [])  # no annotations at all


def test_fully_covered_span_is_not_partial():
    assert span_fully_covered(12, 18, [(10, 20)])
    assert span_fully_covered(10, 20, [(10, 20)])  # exact match


def test_coverage_can_come_from_adjacent_annotations():
    assert span_fully_covered(12, 28, [(10, 20), (20, 30)])
    assert not span_fully_covered(12, 28, [(10, 20), (21, 30)])  # one-char gap


def test_overhanging_span_is_partial():
    assert not span_fully_covered(5, 15, [(10, 20)])
    assert not span_fully_covered(15, 25, [(10, 20)])


class _OneShotScorer:
    def __init__(self, scores: TokenScores):
        self._scores = scores

    def score(self, text: str, passages: list[str]) -> TokenScores:
        return self._scores


def _example(text: str, annotations: tuple[tuple[int, int], ...]) -> Example:
    return Example(
        id="x",
        task_type="QA",
        query="",
        context="doc",
        response=text,
        hallucinated=bool(annotations),
        spans=annotations,
    )


def test_labeled_spans_are_answer_relative_with_label_and_raw_risk():
    text = "Paris is nice. It has 50m people."
    second_start = len("Paris is nice.") + 1
    # Flag "50m people" (answer chars 22..32) at risk 0.9; everything else 0.1.
    scores = TokenScores(
        probs=[0.9 if 22 <= i < 32 else 0.1 for i in range(len(text))],
        offsets=[(i, i + 1) for i in range(len(text))],
    )
    example = _example(text, ((second_start, len(text)),))
    (span,) = labeled_spans(example, _OneShotScorer(scores), span_threshold=0.5)
    # The engine span is claim-relative; the labeled span is re-based to the answer.
    assert (span.start, span.end) == (22, 32)
    assert text[span.start : span.end] == "50m people"
    assert span.raw_risk == pytest.approx(0.9)
    assert span.unsupported == 1
    assert span.partial is False  # fully inside the annotated second sentence
    assert span.example_id == "x"


def test_labeled_span_missing_every_annotation_is_supported():
    text = "Paris is nice. It has 50m people."
    scores = TokenScores(
        probs=[0.9 if i < 5 else 0.1 for i in range(len(text))],  # flag "Paris"
        offsets=[(i, i + 1) for i in range(len(text))],
    )
    example = _example(text, ((15, len(text)),))  # only the second sentence annotated
    (span,) = labeled_spans(example, _OneShotScorer(scores), span_threshold=0.5)
    assert span.unsupported == 0
    assert span.partial is False


def test_labeled_span_grazing_an_annotation_is_partial():
    text = "Paris is nice. It has 50m people."
    scores = TokenScores(
        probs=[0.9 if 10 <= i < 20 else 0.1 for i in range(len(text))],
        offsets=[(i, i + 1) for i in range(len(text))],
    )
    example = _example(text, ((15, len(text)),))
    spans = list(labeled_spans(example, _OneShotScorer(scores), span_threshold=0.5))
    # The flagged region crosses the sentence boundary, so the engine clips it into
    # per-claim spans; the one overlapping the annotation only grazes it.
    overlapping = [s for s in spans if s.unsupported]
    assert overlapping
    assert all(s.partial for s in overlapping if s.start < 15)


def test_no_flagged_tokens_yield_no_spans():
    text = "Paris is nice."
    scores = TokenScores(
        probs=[0.1] * len(text),
        offsets=[(i, i + 1) for i in range(len(text))],
    )
    assert list(labeled_spans(_example(text, ()), _OneShotScorer(scores), 0.5)) == []
