"""Liveness and readiness probe tests."""

from fastapi.testclient import TestClient

from api.app import create_app


class InstantScorer:
    def score(self, claim: str, passages: list[str]) -> list[float]:
        return [0.9 for _ in passages]


def test_healthz_ok_once_serving(client):
    assert client.get("/healthz").status_code == 200


def test_readyz_not_ready_before_model_load(config_path):
    app = create_app(config_path=config_path, scorer_factory=lambda cfg: InstantScorer())
    # No context manager: the lifespan has not run, so the model is not loaded.
    unstarted = TestClient(app)
    assert unstarted.get("/healthz").status_code == 200
    assert unstarted.get("/readyz").status_code == 503


def test_readyz_ok_after_model_load(client):
    assert client.get("/readyz").status_code == 200
