"""End-to-end smoke test against a deployed instance.

Excluded by default (see pytest addopts); `make e2e` and the CI e2e job run it with:
    uv run pytest -m e2e --no-cov
The target comes from PLUMB_URL (default http://127.0.0.1:8000).
"""

import os

import httpx2
import pytest

GOLDEN_REQUEST = {
    "text": "The capital of France is Paris.",
    "context": ["Paris is the capital of France."],
    "mode": "fast",
}


@pytest.mark.e2e
def test_golden_verify_request():
    base_url = os.environ.get("PLUMB_URL", "http://127.0.0.1:8000")
    response = httpx2.post(f"{base_url}/v1/verify", json=GOLDEN_REQUEST, timeout=60)
    assert response.status_code == 200
    body = response.json()
    assert body["gate"] == "pass"
    assert [claim["verdict"] for claim in body["claims"]] == ["supported"]
    # Calibrated confidence, never exact certainty (ADR-0008).
    assert all(0.0 < claim["confidence"] < 1.0 for claim in body["claims"])
    assert body["engine_version"]
    assert body["config_version"]
