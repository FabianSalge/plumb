from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from api.app import create_app
from engine.signals import TokenScores
from tests.engine.factories import make_artifact, make_config, write_config
from tests.engine.signals.fakes import FakeScorer, char_scores


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
            config=make_config(threshold=threshold, span_threshold=span_threshold),
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
