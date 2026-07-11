"""Unit tests for sentence-level labelling — pure overlap logic, no model."""

import pytest
from bench.data import Example
from bench.sentence import scored_sentences, sentence_hallucinated

from engine.signals import TokenScores


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


class _OneShotScorer:
    def __init__(self, scores: TokenScores):
        self._scores = scores

    def score(self, text: str, passages: list[str]) -> TokenScores:
        return self._scores


def test_scored_sentences_yields_each_claim_with_its_span_label():
    text = "Paris is nice. It has 50m people."
    first_len = len("Paris is nice.")
    # The segmenter's first claim spans [0, 15) — the trailing space attaches to the
    # preceding sentence — so the risk split must land on that boundary, not on the
    # bare sentence length, or the space's token would drag claim 0's support down.
    second_start = first_len + 1
    scores = TokenScores(
        probs=[0.9 if i >= second_start else 0.1 for i in range(len(text))],
        offsets=[(i, i + 1) for i in range(len(text))],
    )
    example = Example(
        id="x",
        task_type="QA",
        query="",
        context="doc",
        response=text,
        hallucinated=True,
        spans=((second_start, len(text)),),
    )
    claims = list(scored_sentences(example, _OneShotScorer(scores), span_threshold=0.5))
    assert [(claim.text.strip(), label) for claim, label in claims] == [
        ("Paris is nice.", 0),
        ("It has 50m people.", 1),
    ]
    assert claims[0][0].support == 0.9
    assert claims[1][0].support == pytest.approx(0.1)
