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


@pytest.mark.e2e
def test_golden_thorough_request():
    """Against the store-enabled deployment (make deploy-thorough): the answer is
    verified with no caller context, and claims carry retrieval provenance from
    the seeded store (tests/e2e/postgres.yaml)."""
    if not os.environ.get("PLUMB_E2E_THOROUGH"):
        pytest.skip("needs the store-enabled deployment — run via `make e2e-thorough`")
    base_url = os.environ.get("PLUMB_URL", "http://127.0.0.1:8000")
    request = {"text": "The capital of France is Paris.", "mode": "thorough"}
    response = httpx2.post(f"{base_url}/v1/verify", json=request, timeout=120)
    assert response.status_code == 200
    body = response.json()
    assert body["gate"] == "pass"
    assert [claim["verdict"] for claim in body["claims"]] == ["supported"]
    assert all(0.0 < claim["confidence"] < 1.0 for claim in body["claims"])
    evidence = body["claims"][0]["evidence"]
    assert evidence, "thorough claims must carry retrieval provenance"
    top = evidence[0]
    assert top["source_id"] == "wiki/paris"
    assert top["chunk_id"] == "1"
    assert top["rank"] == 1
    assert top["snapshot_id"] == "e2e-1"
