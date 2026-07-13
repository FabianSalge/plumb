"""Pluggable per-claim signals (ADR-0004): this package is the signal slot.

The seam is `Scorer` + `TokenScores`; each occupant lives in its own module
behind it (today: the groundedness cross-encoder in `groundedness`). Adopting
a new detector is an adapter module plus a benchmark run, never an
architecture change (ADR-0006).
"""

from dataclasses import dataclass
from typing import Protocol


class ScorerError(Exception):
    """The scoring model is unavailable or returned something it must not."""


@dataclass(frozen=True)
class TokenScores:
    """Per-token hallucination probabilities over the whole answer, aligned with
    answer-relative character offsets (zero-length offsets are special tokens);
    `truncated` marks a context that was cut to fit the model window. Reduction to
    per-claim support and spans happens in `engine.decomposition`."""

    probs: list[float]
    offsets: list[tuple[int, int]]
    truncated: bool = False


class TokenClassifier(Protocol):
    def token_probs(self, prompt: str, claim: str) -> TokenScores: ...

    def count_tokens(self, text: str) -> int: ...


class Scorer(Protocol):
    def score(self, text: str, passages: list[str]) -> TokenScores: ...

    def count_tokens(self, text: str) -> int:
        """Length of `text` in this scorer's own tokens — the currency the
        thorough-mode evidence pool budget is denominated in (ADR-0010)."""
        ...
