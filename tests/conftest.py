from collections.abc import Iterator
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

from api.app import create_app
from engine.decomposition import CLAIM_UNIT
from engine.scoring import INFERENCE_MODE, TokenScores


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
                "calibration": "calibration.yaml",
            }
        },
    }


def make_artifact(*, a: float = 1.0, b: float = 0.0, **binding_overrides) -> dict:
    """A valid calibration artifact matching `_config`'s fake model. The identity
    coefficients (a=1, b=0) make confidence equal raw support (up to the ε clamp),
    so threshold-behaviour tests read naturally."""
    bindings = {
        "model": "fake/model",
        "revision": "deadbeef",
        "inference_mode": INFERENCE_MODE,
        "claim_unit": CLAIM_UNIT,
    }
    bindings.update(binding_overrides)
    return {
        "schema": 1,
        "method": "platt",
        "coefficients": {"a": a, "b": b},
        "bindings": bindings,
        "fit": {
            "dataset": "test-fixture",
            "exclusion": "none",
            "responses": 3,
            "sentences": 12,
            "sha256": "0" * 64,
            "fitted_at": "2026-07-11",
        },
        "metrics": {
            "in_domain": {"dataset": "test-fixture", "slice": "s", "sentences": 12, "ece": 0.01},
            "out_of_domain": {
                "dataset": "ood-fixture",
                "subsets": ["a"],
                "excluded_subsets": {"RAGTruth": "fitted on RAGTruth"},
                "claims": 5,
                "ece": 0.06,
            },
        },
    }


def write_config(
    directory: Path, *, config: dict | None = None, artifact: dict | None = None
) -> Path:
    """Write a verifier config and its calibration artifact side by side."""
    path = directory / "verifier.yaml"
    path.write_text(yaml.safe_dump(config or _config()))
    (directory / "calibration.yaml").write_text(yaml.safe_dump(artifact or make_artifact()))
    return path


@pytest.fixture
def config_path(tmp_path):
    return write_config(tmp_path)


@pytest.fixture
def make_client(tmp_path):
    """Build a TestClient over an app wired to a FakeScorer returning the given scores.
    The verdict and span thresholds are overridable so the independence of the two
    knobs (an unsupported claim with no spans) is testable."""
    clients: list[TestClient] = []

    def _make(
        scores: TokenScores,
        *,
        threshold: float = 0.5,
        span_threshold: float = 0.5,
        a: float = 1.0,
        b: float = 0.0,
    ) -> TestClient:
        directory = tmp_path / f"cfg-{len(clients)}"
        directory.mkdir()
        path = write_config(
            directory,
            config=_config(threshold=threshold, span_threshold=span_threshold),
            artifact=make_artifact(a=a, b=b),
        )
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
