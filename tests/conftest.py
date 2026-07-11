from collections.abc import Iterator

import pytest
import yaml
from fastapi.testclient import TestClient

from api.app import create_app
from engine.scoring import ClaimScore, Span


class FakeScorer:
    """Stands in for the scoring wrapper: returns one preset ClaimScore per claim."""

    def __init__(self, support: float, spans: list[Span]):
        self.result = ClaimScore(support=support, spans=spans)
        self.calls: list[tuple[str, list[str]]] = []

    def score(self, claim: str, passages: list[str]) -> ClaimScore:
        self.calls.append((claim, passages))
        return self.result


CONFIG = {
    "version": "test-1",
    "signals": {
        "groundedness": {
            "model": "fake/model",
            "revision": "deadbeef",
            "threshold": 0.5,
            "span_threshold": 0.5,
        }
    },
}


@pytest.fixture
def config_path(tmp_path):
    path = tmp_path / "verifier.yaml"
    path.write_text(yaml.safe_dump(CONFIG))
    return path


@pytest.fixture
def make_client(config_path):
    """Build a TestClient over an app wired to a FakeScorer with the given result."""
    clients: list[TestClient] = []

    def _make(support: float, spans: list[Span] | None = None) -> TestClient:
        app = create_app(
            config_path=config_path,
            scorer_factory=lambda cfg: FakeScorer(support, spans or []),
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
    return make_client(0.9)
