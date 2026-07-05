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
    evidence = ["Paris is the capital of France."]
    supported = scorer.score("The capital of France is Paris.", evidence)[0]
    contradicted = scorer.score("The capital of France is Berlin.", evidence)[0]
    assert supported >= cfg.groundedness.threshold
    assert contradicted < cfg.groundedness.threshold
