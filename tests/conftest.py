from collections.abc import Iterator

import pytest
import yaml
from fastapi.testclient import TestClient

from api.app import create_app
from engine.scoring import TokenScores


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


def _config(*, threshold: float = 0.5, span_threshold: float = 0.5) -> dict:
    return {
        "version": "test-1",
        "signals": {
            "groundedness": {
                "model": "fake/model",
                "revision": "deadbeef",
                "threshold": threshold,
                "span_threshold": span_threshold,
            }
        },
    }


@pytest.fixture
def config_path(tmp_path):
    path = tmp_path / "verifier.yaml"
    path.write_text(yaml.safe_dump(_config()))
    return path


@pytest.fixture
def make_client(tmp_path):
    """Build a TestClient over an app wired to a FakeScorer returning the given scores.
    The verdict and span thresholds are overridable so the independence of the two
    knobs (an unsupported claim with no spans) is testable."""
    clients: list[TestClient] = []

    def _make(
        scores: TokenScores, *, threshold: float = 0.5, span_threshold: float = 0.5
    ) -> TestClient:
        path = tmp_path / f"verifier-{len(clients)}.yaml"
        path.write_text(yaml.safe_dump(_config(threshold=threshold, span_threshold=span_threshold)))
        app = create_app(
            config_path=path,
            scorer_factory=lambda cfg: FakeScorer(scores),
        )
        client = TestClient(app)
        client.__enter__()
        clients.append(client)
        return client

    yield _make
    for client in clients:
        client.__exit__(None, None, None)


@pytest.fixture
def client(make_client) -> Iterator[TestClient]:
    return make_client(char_scores("The sky is blue."))
