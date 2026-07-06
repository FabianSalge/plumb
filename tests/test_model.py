"""Integration test against the real LettuceDetect weights.

Excluded by default (see pytest addopts); run with:
    uv run --extra model pytest -m model
"""

import pytest

from engine.config import load_config
from engine.scoring import LettuceDetectScorer


@pytest.mark.model
def test_score_direction_through_real_model():
    cfg = load_config("config/verifier.yaml")
    scorer = LettuceDetectScorer.load(cfg.groundedness)
    evidence = ["Paris is the capital of France.", "The Eiffel Tower is in Paris."]
    supported = scorer.score("The capital of France is Paris.", evidence)
    contradicted = scorer.score("The capital of France is Berlin.", evidence)
    assert supported.support >= cfg.groundedness.threshold
    assert contradicted.support < cfg.groundedness.threshold
    assert contradicted.spans, "a contradicted claim should localize unsupported spans"
