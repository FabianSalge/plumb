"""Scoring wrapper around HHEM-2.1-open's remote-code predict() API."""

from collections.abc import Iterable
from typing import Protocol

from engine.config import SignalModelConfig


class ScorerError(Exception):
    """The scoring model is unavailable or returned something it must not."""


class Scorer(Protocol):
    def score(self, claim: str, passages: list[str]) -> list[float]: ...


class SupportsPredict(Protocol):
    def predict(self, pairs: list[tuple[str, str]]) -> Iterable[float]: ...


def evidence_claim_pairs(claim: str, passages: list[str]) -> list[tuple[str, str]]:
    # HHEM's predict() takes (premise, hypothesis) positionally and does not
    # check the order — flipped pairs still return plausible-looking scores.
    return [(passage, claim) for passage in passages]


class HHEMScorer:
    def __init__(self, model: SupportsPredict) -> None:
        self._model = model

    @classmethod
    def load(cls, cfg: SignalModelConfig) -> "HHEMScorer":  # pragma: no cover — `pytest -m model`
        try:
            from transformers import AutoModelForSequenceClassification
        except ImportError as exc:
            raise ScorerError(
                "transformers is not installed — install the 'hhem' extra to load the scoring model"
            ) from exc
        model = AutoModelForSequenceClassification.from_pretrained(
            cfg.model,
            revision=cfg.revision,
            trust_remote_code=True,
        )
        return cls(model)

    def score(self, claim: str, passages: list[str]) -> list[float]:
        raw = self._model.predict(evidence_claim_pairs(claim, passages))
        scores = [float(value) for value in raw]
        for value in scores:
            if not 0.0 <= value <= 1.0:
                raise ScorerError(f"model returned score {value} outside [0, 1]")
        return scores
