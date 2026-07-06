"""Unit tests for the LettuceDetect scoring wrapper and verdict logic."""

import logging
import sys

import pytest

from engine.config import SignalModelConfig
from engine.scoring import LettuceDetectScorer, ScorerError, TokenScores, render_prompt
from engine.verdict import gate_decision, judge_claim

SPAN_THRESHOLD = 0.5


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


def scorer(results: list[TokenScores], span_threshold: float = SPAN_THRESHOLD):
    return LettuceDetectScorer(FakePipeline(results), span_threshold=span_threshold)


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


def test_claim_occupies_the_answer_slot_of_one_joint_pass():
    """All passages render into a single context and the claim is the second segment
    of the pair — one forward pass per claim, not one per passage."""
    pipeline = FakePipeline([token_scores([0.1])])
    LettuceDetectScorer(pipeline, span_threshold=SPAN_THRESHOLD).score(
        "the claim", ["passage a", "passage b"]
    )
    assert pipeline.calls == [(render_prompt(["passage a", "passage b"]), "the claim")]


def test_support_is_one_minus_max_token_prob():
    assert scorer([token_scores([0.1, 0.75, 0.2])]).score("c", ["p"]).support == 0.25


def test_score_outside_unit_interval_fails_loudly():
    with pytest.raises(ScorerError):
        scorer([token_scores([1.5])]).score("c", ["p"])
    with pytest.raises(ScorerError):
        scorer([token_scores([-0.5])]).score("c", ["p"])


def test_no_claim_tokens_fails_loudly():
    with pytest.raises(ScorerError):
        scorer([token_scores([])]).score("c", ["p"])


def test_zero_passages_fails_loudly():
    """The HTTP layer rejects an empty context, but engine callers must not be
    able to score a claim against no evidence and get a plausible number back."""
    with pytest.raises(ScorerError, match="passages"):
        scorer([]).score("c", [])


def test_missing_dependency_error_names_the_extra(monkeypatch):
    monkeypatch.setitem(sys.modules, "transformers", None)
    cfg = SignalModelConfig(
        model="fake/model", revision="deadbeef", threshold=0.5, span_threshold=0.5
    )
    with pytest.raises(ScorerError, match="'model' extra"):
        LettuceDetectScorer.load(cfg)


# One flagged region of "Paris is small." — shared by the span-shape and
# span-logging tests so their expectations cannot drift apart.
FLAGGED_CLAIM = "Paris is small."
FLAGGED_RESULT = token_scores(
    probs=[0.1, 0.9, 0.95, 0.2],
    offsets=[(0, 5), (6, 8), (9, 14), (14, 15)],
)


def test_flagged_tokens_become_spans():
    """Contiguous tokens at or above the span threshold merge into one character span
    over the claim, returned with the score."""
    spans = scorer([FLAGGED_RESULT]).score(FLAGGED_CLAIM, ["p"]).spans
    assert [(s.start, s.end, s.text) for s in spans] == [(6, 14, "is small")]


def test_span_threshold_is_injected_not_hardcoded():
    """The same token probabilities yield different spans under a different configured
    threshold — the knob lives in versioned config, not in the module."""
    result = token_scores(probs=[0.6, 0.2], offsets=[(0, 5), (5, 10)])
    assert scorer([result], span_threshold=0.5).score("hello world", ["p"]).spans
    assert not scorer([result], span_threshold=0.7).score("hello world", ["p"]).spans


def test_spans_are_logged_with_confidences(caplog):
    """Structured logs keep the raw span confidences; the returned spans are what the
    API renders and calibration hasn't blessed a confidence yet."""
    with caplog.at_level(logging.INFO, logger="plumb.engine.scoring"):
        scorer([FLAGGED_RESULT]).score(FLAGGED_CLAIM, ["p"])
    records = [r for r in caplog.records if hasattr(r, "spans")]
    assert len(records) == 1
    assert records[0].spans == [{"start": 6, "end": 14, "text": "is small", "confidence": 0.95}]


def test_special_tokens_never_enter_spans():
    """Zero-length offsets (special tokens like the trailing separator) count toward
    the support score but must not produce span characters."""
    result = token_scores(probs=[0.9, 0.9], offsets=[(0, 5), (0, 0)])
    spans = scorer([result]).score("Paris", ["p"]).spans
    assert [(s.start, s.end, s.text) for s in spans] == [(0, 5, "Paris")]


def test_supported_claim_yields_no_spans(caplog):
    with caplog.at_level(logging.INFO, logger="plumb.engine.scoring"):
        result = scorer([token_scores([0.1, 0.2])]).score("c", ["p"])
    assert result.spans == []
    assert not [r for r in caplog.records if hasattr(r, "spans")]


def test_truncated_context_is_logged_with_passage_count(caplog):
    result = TokenScores(probs=[0.1], offsets=[(0, 1)], truncated=True)
    with caplog.at_level(logging.WARNING, logger="plumb.engine.scoring"):
        scorer([result]).score("c", ["p1", "p2", "p3"])
    truncations = [r for r in caplog.records if "truncated" in r.getMessage()]
    assert len(truncations) == 1
    assert truncations[0].passage_count == 3


def test_judge_claim_maps_score_to_verdict():
    verdict = judge_claim("c", score=0.8, threshold=0.5)
    assert verdict.verdict == "supported"
    assert verdict.score == 0.8


def test_judge_claim_below_threshold_is_unsupported():
    verdict = judge_claim("c", score=0.4, threshold=0.5)
    assert verdict.verdict == "unsupported"
    assert verdict.score == 0.4


def test_gate_passes_only_when_all_supported():
    supported = judge_claim("a", 0.9, 0.5)
    unsupported = judge_claim("b", 0.1, 0.5)
    assert gate_decision([supported]) == "pass"
    assert gate_decision([supported, unsupported]) == "block"
