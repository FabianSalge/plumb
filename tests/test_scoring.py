"""Unit tests for the LettuceDetect scoring wrapper.

Segment-after-score (ADR-0009) moves reduction and span attribution to
`engine.decomposition`; the scorer's job is now one whole-answer forward pass
returning per-token risk with answer-relative offsets. Those live in
`tests/test_decomposition.py`.
"""

import logging
import sys

import pytest

from engine.config import SignalModelConfig
from engine.scoring import LettuceDetectScorer, ScorerError, TokenScores, render_prompt


class FakePipeline:
    """Stands in for the torch-backed token classifier: one preset TokenScores per call."""

    def __init__(self, results: list[TokenScores]):
        self._results = list(results)
        self.calls: list[tuple[str, str]] = []

    def token_probs(self, prompt: str, claim: str) -> TokenScores:
        self.calls.append((prompt, claim))
        return self._results.pop(0)


def token_scores(probs: list[float], offsets: list[tuple[int, int]] | None = None) -> TokenScores:
    return TokenScores(probs=probs, offsets=offsets or [(i, i + 1) for i in range(len(probs))])


def scorer(results: list[TokenScores]):
    return LettuceDetectScorer(FakePipeline(results))


def test_prompt_format_is_golden():
    """The rendered prompt must match lettucedetect's training format byte-for-byte —
    any drift silently degrades scores instead of erroring."""
    assert render_prompt(["The sky is blue."]) == (
        "Summarize the following text:\npassage 1: The sky is blue.\noutput:"
    )


def test_multi_passage_prompt_format_is_golden():
    """All passages render jointly as enumerated lines, exactly as lettucedetect 0.2.1
    formats a question-less context."""
    assert render_prompt(["The sky is blue.", "Water is wet."]) == (
        "Summarize the following text:\n"
        "passage 1: The sky is blue.\n"
        "passage 2: Water is wet.\n"
        "output:"
    )


def test_answer_occupies_the_answer_slot_of_one_joint_pass():
    """All passages render into a single context and the whole answer is the second
    segment of the pair — one forward pass, not one per passage."""
    pipeline = FakePipeline([token_scores([0.1])])
    LettuceDetectScorer(pipeline).score("the answer", ["passage a", "passage b"])
    assert pipeline.calls == [(render_prompt(["passage a", "passage b"]), "the answer")]


def test_score_returns_the_whole_answer_token_output():
    """The scorer returns per-token risk with answer-relative offsets, unreduced —
    reduction is the decomposition step's work."""
    result = token_scores([0.1, 0.9, 0.2], offsets=[(0, 5), (6, 8), (9, 14)])
    out = scorer([result]).score("some answer", ["p"])
    assert out.probs == [0.1, 0.9, 0.2]
    assert out.offsets == [(0, 5), (6, 8), (9, 14)]


def test_no_token_probabilities_fails_loudly():
    with pytest.raises(ScorerError):
        scorer([token_scores([])]).score("c", ["p"])


def test_zero_passages_fails_loudly():
    """The HTTP layer rejects an empty context, but engine callers must not be
    able to score an answer against no evidence and get a plausible number back."""
    with pytest.raises(ScorerError, match="passages"):
        scorer([]).score("c", [])


def test_missing_dependency_error_names_the_extra(monkeypatch):
    monkeypatch.setitem(sys.modules, "transformers", None)
    cfg = SignalModelConfig(
        model="fake/model",
        revision="deadbeef",
        threshold=0.5,
        span_threshold=0.5,
        calibration="calibration.yaml",
    )
    with pytest.raises(ScorerError, match="'model' extra"):
        LettuceDetectScorer.load(cfg)


def test_truncated_context_is_logged_with_passage_count(caplog):
    result = TokenScores(probs=[0.1], offsets=[(0, 1)], truncated=True)
    with caplog.at_level(logging.WARNING, logger="plumb.engine.scoring"):
        scorer([result]).score("c", ["p1", "p2", "p3"])
    truncations = [r for r in caplog.records if "truncated" in r.getMessage()]
    assert len(truncations) == 1
    assert truncations[0].passage_count == 3
