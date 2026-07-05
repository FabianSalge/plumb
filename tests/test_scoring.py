"""Unit tests for the HHEM scoring wrapper and verdict logic."""

import pytest

from engine.scoring import HHEMScorer, ScorerError, evidence_claim_pairs
from engine.verdict import gate_decision, judge_claim


class FakeModel:
    """Mimics HHEM's nonstandard predict() bolted on by its remote code."""

    def __init__(self, scores: list[float]):
        self._scores = scores
        self.received_pairs: list[tuple[str, str]] | None = None

    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        self.received_pairs = pairs
        return self._scores


def test_pair_order_is_evidence_then_claim():
    """HHEM's pair order is positional and unchecked — flipping it returns
    plausible-looking garbage, so the wrapper must own the order."""
    assert evidence_claim_pairs("claim", ["p1", "p2"]) == [("p1", "claim"), ("p2", "claim")]


def test_scorer_sends_evidence_first():
    model = FakeModel([0.5, 0.5])
    HHEMScorer(model).score("the claim", ["passage a", "passage b"])
    assert model.received_pairs == [("passage a", "the claim"), ("passage b", "the claim")]


def test_scorer_returns_one_float_per_passage():
    scores = HHEMScorer(FakeModel([0.1, 0.9])).score("c", ["p1", "p2"])
    assert scores == [0.1, 0.9]
    assert all(isinstance(s, float) for s in scores)


def test_score_outside_unit_interval_fails_loudly():
    with pytest.raises(ScorerError):
        HHEMScorer(FakeModel([1.2])).score("c", ["p"])
    with pytest.raises(ScorerError):
        HHEMScorer(FakeModel([-0.1])).score("c", ["p"])


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
