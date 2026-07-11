"""Fakes for the scorer seam: preset whole-answer TokenScores."""

from engine.signals import TokenScores


class FakeScorer:
    """Stands in for the scoring wrapper: returns one preset whole-answer TokenScores.
    The real segmenter and reducer run behind the API, so the contract tests exercise
    the actual decomposition wiring."""

    def __init__(self, scores: TokenScores):
        self.scores = scores
        self.calls: list[tuple[str, list[str]]] = []

    def score(self, text: str, passages: list[str]) -> TokenScores:
        self.calls.append((text, passages))
        return self.scores


def char_scores(
    text: str, *, base: float = 0.1, flag: tuple[int, int] | None = None, flag_prob: float = 0.8
) -> TokenScores:
    """Whole-answer token scores, one token per character. Characters inside `flag`
    carry `flag_prob`; the rest carry `base`. Support of any claim is 1 − its max."""
    lo, hi = flag or (-1, -1)
    probs = [flag_prob if lo <= i < hi else base for i in range(len(text))]
    return TokenScores(probs=probs, offsets=[(i, i + 1) for i in range(len(text))])
