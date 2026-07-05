"""Unit tests for the LettuceDetect scoring wrapper and verdict logic."""

import logging
import sys

import pytest

from engine.config import SignalModelConfig
from engine.scoring import LettuceDetectScorer, ScorerError, TokenScores, render_prompt
from engine.verdict import gate_decision, judge_claim


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


def test_prompt_format_is_golden():
    """The rendered prompt must match lettucedetect's training format byte-for-byte —
    any drift silently degrades scores instead of erroring."""
    assert render_prompt("The sky is blue.") == (
        "Summarize the following text:\npassage 1: The sky is blue.\noutput:"
    )


def test_claim_occupies_the_answer_slot():
    """The passage is rendered into the context template and the claim is the second
    segment of the pair — swapping them scores the passage and returns plausible garbage."""
    pipeline = FakePipeline([token_scores([0.1]), token_scores([0.2])])
    LettuceDetectScorer(pipeline).score("the claim", ["passage a", "passage b"])
    assert pipeline.calls == [
        (render_prompt("passage a"), "the claim"),
        (render_prompt("passage b"), "the claim"),
    ]


def test_one_support_score_per_passage_in_order():
    pipeline = FakePipeline([token_scores([0.75]), token_scores([0.5])])
    scores = LettuceDetectScorer(pipeline).score("c", ["p1", "p2"])
    assert scores == [0.25, 0.5]
    assert all(isinstance(s, float) for s in scores)


def test_support_is_one_minus_max_token_prob():
    pipeline = FakePipeline([token_scores([0.1, 0.75, 0.2])])
    assert LettuceDetectScorer(pipeline).score("c", ["p"]) == [0.25]


def test_score_outside_unit_interval_fails_loudly():
    with pytest.raises(ScorerError):
        LettuceDetectScorer(FakePipeline([token_scores([1.5])])).score("c", ["p"])
    with pytest.raises(ScorerError):
        LettuceDetectScorer(FakePipeline([token_scores([-0.5])])).score("c", ["p"])


def test_no_claim_tokens_fails_loudly():
    with pytest.raises(ScorerError):
        LettuceDetectScorer(FakePipeline([token_scores([])])).score("c", ["p"])


def test_missing_dependency_error_names_the_extra(monkeypatch):
    monkeypatch.setitem(sys.modules, "transformers", None)
    cfg = SignalModelConfig(model="fake/model", revision="deadbeef", threshold=0.5)
    with pytest.raises(ScorerError, match="'model' extra"):
        LettuceDetectScorer.load(cfg)


def test_flagged_tokens_become_logged_spans(caplog):
    """Contiguous tokens classified as hallucinated merge into one character span
    over the claim, emitted as structured log detail."""
    claim = "Paris is small."
    result = token_scores(
        probs=[0.1, 0.9, 0.95, 0.2],
        offsets=[(0, 5), (6, 8), (9, 14), (14, 15)],
    )
    with caplog.at_level(logging.INFO, logger="plumb.engine.scoring"):
        LettuceDetectScorer(FakePipeline([result])).score(claim, ["p"])
    records = [r for r in caplog.records if hasattr(r, "spans")]
    assert len(records) == 1
    assert records[0].passage_index == 0
    assert records[0].spans == [{"start": 6, "end": 14, "text": "is small", "confidence": 0.95}]


def test_special_tokens_never_enter_spans(caplog):
    """Zero-length offsets (special tokens like the trailing separator) count toward
    the support score but must not produce span characters."""
    result = token_scores(probs=[0.9, 0.9], offsets=[(0, 5), (0, 0)])
    with caplog.at_level(logging.INFO, logger="plumb.engine.scoring"):
        LettuceDetectScorer(FakePipeline([result])).score("Paris", ["p"])
    records = [r for r in caplog.records if hasattr(r, "spans")]
    assert records[0].spans == [{"start": 0, "end": 5, "text": "Paris", "confidence": 0.9}]


def test_supported_claim_logs_no_spans(caplog):
    with caplog.at_level(logging.INFO, logger="plumb.engine.scoring"):
        LettuceDetectScorer(FakePipeline([token_scores([0.1, 0.2])])).score("c", ["p"])
    assert not [r for r in caplog.records if hasattr(r, "spans")]


def test_truncated_context_is_logged(caplog):
    result = TokenScores(probs=[0.1], offsets=[(0, 1)], truncated=True)
    with caplog.at_level(logging.WARNING, logger="plumb.engine.scoring"):
        LettuceDetectScorer(FakePipeline([result])).score("c", ["p"])
    assert any("truncated" in r.getMessage() for r in caplog.records)


def test_judge_claim_takes_max_over_passages():
    verdict = judge_claim("c", scores=[0.3, 0.8, 0.6], threshold=0.5)
    assert verdict.verdict == "supported"
    assert verdict.score == 0.8
    assert verdict.evidence_index == 1


def test_judge_claim_tie_picks_first_passage():
    assert judge_claim("c", scores=[0.7, 0.7], threshold=0.5).evidence_index == 0


def test_judge_claim_below_threshold_is_unsupported():
    verdict = judge_claim("c", scores=[0.2, 0.4], threshold=0.5)
    assert verdict.verdict == "unsupported"
    assert verdict.score == 0.4
    assert verdict.evidence_index == 1


def test_gate_passes_only_when_all_supported():
    supported = judge_claim("a", [0.9], 0.5)
    unsupported = judge_claim("b", [0.1], 0.5)
    assert gate_decision([supported]) == "pass"
    assert gate_decision([supported, unsupported]) == "block"
