"""Tests for the reranker seam: protocol shape, and the real cross-encoder (marked)."""

import pytest

from engine.config import RerankerConfig
from engine.retrieval import Reranker


class FakeReranker:
    """Scores pairs by how many query words appear in the passage."""

    def score(self, pairs: list[tuple[str, str]]) -> list[float]:
        return [
            float(sum(word.lower() in passage.lower() for word in query.split()))
            for query, passage in pairs
        ]


def test_fake_reranker_satisfies_the_protocol():
    reranker: Reranker = FakeReranker()
    assert isinstance(reranker, Reranker)


def test_score_returns_one_score_per_pair_in_order():
    pairs = [
        ("eiffel tower height", "The Eiffel Tower is 330 metres tall."),
        ("eiffel tower height", "The Louvre is a museum."),
    ]
    scores = FakeReranker().score(pairs)
    assert len(scores) == 2
    assert scores[0] > scores[1]


def test_empty_pairs_score_to_empty():
    assert FakeReranker().score([]) == []


@pytest.mark.model
def test_pinned_cross_encoder_ranks_relevant_passage_higher():
    from engine.retrieval.rerank import CrossEncoderReranker

    cfg = RerankerConfig(
        model="BAAI/bge-reranker-v2-m3",
        revision="953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e",
    )
    reranker = CrossEncoderReranker.load(cfg)
    query = "How tall is the Eiffel Tower?"
    scores = reranker.score(
        [
            (query, "The Eiffel Tower is 330 metres tall and stands in Paris."),
            (query, "The Louvre is the world's most-visited museum."),
        ]
    )
    assert len(scores) == 2
    assert scores[0] > scores[1]


def test_cross_encoder_without_model_extra_fails_loudly(monkeypatch):
    import builtins

    from engine.retrieval.rerank import CrossEncoderReranker
    from engine.signals import ScorerError

    real_import = builtins.__import__

    def block_torch(name, *args, **kwargs):
        if name in {"torch", "transformers"}:
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", block_torch)
    with pytest.raises(ScorerError, match="model"):
        CrossEncoderReranker.load(RerankerConfig(model="x", revision="y"))
