"""Unit tests for verdict mapping and the conjunctive gate (engine/verdict.py)."""

from engine.verdict import gate_decision, judge_claim


def test_judge_claim_maps_score_to_verdict():
    verdict = judge_claim("c", score=0.8, threshold=0.5)
    assert verdict.verdict == "supported"
    assert verdict.score == 0.8


def test_judge_claim_below_threshold_is_unsupported():
    verdict = judge_claim("c", score=0.4, threshold=0.5)
    assert verdict.verdict == "unsupported"
    assert verdict.score == 0.4


def test_judge_claim_at_threshold_is_supported():
    assert judge_claim("c", score=0.5, threshold=0.5).verdict == "supported"


def test_gate_passes_only_when_all_supported():
    supported = judge_claim("a", 0.9, 0.5)
    unsupported = judge_claim("b", 0.1, 0.5)
    assert gate_decision([supported]) == "pass"
    assert gate_decision([supported, unsupported]) == "block"
