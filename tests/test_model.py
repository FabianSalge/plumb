"""Integration test against the real LettuceDetect weights.

Excluded by default (see pytest addopts); run with:
    uv run --extra model pytest -m model
"""

import pytest

from engine.config import load_config
from engine.decomposition import decompose
from engine.scoring import LettuceDetectScorer


@pytest.mark.model
def test_score_direction_through_real_model():
    cfg = load_config("config/verifier.yaml")
    scorer = LettuceDetectScorer.load(cfg.groundedness)
    evidence = ["Paris is the capital of France.", "The Eiffel Tower is in Paris."]
    span_threshold = cfg.groundedness.span_threshold

    supported_text = "The capital of France is Paris."
    contradicted_text = "The capital of France is Berlin."
    supported = decompose(supported_text, scorer.score(supported_text, evidence), span_threshold)
    contradicted = decompose(
        contradicted_text, scorer.score(contradicted_text, evidence), span_threshold
    )
    # each answer is a single sentence — one claim
    assert supported[0].support >= cfg.groundedness.threshold
    assert contradicted[0].support < cfg.groundedness.threshold
    assert contradicted[0].spans, "a contradicted claim should localize unsupported spans"
