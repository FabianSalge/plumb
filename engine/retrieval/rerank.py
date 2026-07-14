"""The precision half of recall-then-rerank: a pinned cross-encoder (ADR-0002).

Scores order candidates within and across claims (rank slots, then global
fill); only the ordering matters, so raw logits are returned untransformed.
The pin is provisional — #58 owns reranker selection — and a swap is a
config-version bump against the Reranker protocol.
"""

from typing import Any, Protocol, runtime_checkable

from engine.config import RerankerConfig
from engine.signals import ScorerError

# Pairs beyond this are scored in slices to bound peak memory on CPU.
_BATCH_SIZE = 32


@runtime_checkable
class Reranker(Protocol):
    def score(self, pairs: list[tuple[str, str]]) -> list[float]:
        """One relevance score per (query, passage) pair, higher is more relevant."""
        ...


class CrossEncoderReranker:
    def __init__(self, model: Any, tokenizer: Any) -> None:
        self._model = model
        self._tokenizer = tokenizer

    @classmethod
    def load(cls, cfg: RerankerConfig) -> "CrossEncoderReranker":
        try:
            import torch  # noqa: F401
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise ScorerError(
                "transformers/torch are not installed — "
                "install the 'model' extra to load the reranker"
            ) from exc
        # Import guard above is unit-tested; the download below is exercised
        # by `pytest -m model` only.
        tokenizer = AutoTokenizer.from_pretrained(  # pragma: no cover
            cfg.model, revision=cfg.revision
        )
        model = AutoModelForSequenceClassification.from_pretrained(  # pragma: no cover
            cfg.model, revision=cfg.revision
        )
        model.eval()  # pragma: no cover
        return cls(model, tokenizer)  # pragma: no cover

    def score(self, pairs: list[tuple[str, str]]) -> list[float]:  # pragma: no cover
        import torch

        scores: list[float] = []
        for start in range(0, len(pairs), _BATCH_SIZE):
            batch = pairs[start : start + _BATCH_SIZE]
            encoding = self._tokenizer(
                [query for query, _ in batch],
                [passage for _, passage in batch],
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            with torch.no_grad():
                logits = self._model(**encoding).logits
            scores.extend(float(v) for v in logits.view(-1).tolist())
        return scores
