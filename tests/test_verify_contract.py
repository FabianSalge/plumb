"""Contract tests for POST /v1/verify (openspec/changes/add-verify-endpoint/specs/verify-api)."""


def verify(client, **overrides):
    body = {"text": "The sky is blue.", "context": ["The sky is blue."], "mode": "fast"}
    body.update(overrides)
    return client.post("/v1/verify", json=body)


def test_supported_claim(make_client):
    client = make_client([0.2, 0.9])
    resp = verify(client, context=["irrelevant passage", "supporting passage"])
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["claims"]) == 1
    claim = body["claims"][0]
    assert claim["text"] == "The sky is blue."
    assert claim["verdict"] == "supported"
    assert claim["score"] == 0.9
    assert claim["evidence_index"] == 1
    assert body["gate"] == "pass"


def test_unsupported_claim(make_client):
    client = make_client([0.1, 0.2])
    resp = verify(client, context=["unrelated", "also unrelated"])
    assert resp.status_code == 200
    body = resp.json()
    claim = body["claims"][0]
    assert claim["verdict"] == "unsupported"
    assert claim["score"] == 0.2
    assert body["gate"] == "block"


def test_score_at_threshold_is_supported(make_client):
    client = make_client([0.5])
    body = verify(client).json()
    assert body["claims"][0]["verdict"] == "supported"


def test_missing_text_is_400(client):
    resp = client.post("/v1/verify", json={"context": ["evidence"], "mode": "fast"})
    assert resp.status_code == 400
    assert "text" in resp.text


def test_empty_text_is_400(client):
    assert verify(client, text="").status_code == 400


def test_missing_context_is_400(client):
    resp = client.post("/v1/verify", json={"text": "claim", "mode": "fast"})
    assert resp.status_code == 400
    assert "context" in resp.text


def test_empty_context_is_400(client):
    assert verify(client, context=[]).status_code == 400


def test_unknown_mode_is_400_not_silent_degradation(client):
    resp = verify(client, mode="thorough")
    assert resp.status_code == 400
    assert "fast" in resp.text


def test_version_fields_present(client):
    body = verify(client).json()
    assert body["engine_version"]
    assert body["config_version"] == "test-1"


def test_response_shape_carries_no_span_fields(client):
    """Span detail is observability (structured logs) only — it must not leak
    into the response contract before calibration (#29)."""
    body = verify(client).json()
    assert set(body) == {"claims", "gate", "engine_version", "config_version"}
    assert set(body["claims"][0]) == {"text", "verdict", "score", "evidence_index"}


def test_contradicted_absent_from_vocabulary(make_client):
    """Tier-1 verdicts are exactly supported/unsupported until the NLI signal lands."""
    for scores in ([0.0], [1.0]):
        body = verify(make_client(scores)).json()
        assert body["claims"][0]["verdict"] in {"supported", "unsupported"}
