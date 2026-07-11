"""Contract tests for POST /v1/verify (openspec/specs/verify-api)."""

import pytest

from tests.engine.signals.fakes import char_scores


def verify(client, **overrides):
    body = {"text": "The sky is blue.", "context": ["The sky is blue."], "mode": "fast"}
    body.update(overrides)
    return client.post("/v1/verify", json=body)


def test_supported_claim(make_client):
    client = make_client(char_scores("The sky is blue."))
    resp = verify(client, context=["irrelevant passage", "supporting passage"])
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["claims"]) == 1
    claim = body["claims"][0]
    assert claim["text"] == "The sky is blue."
    assert claim["start"] == 0
    assert claim["end"] == 16
    assert claim["verdict"] == "supported"
    assert claim["confidence"] == pytest.approx(0.9)
    assert claim["spans"] == []
    assert body["gate"] == "pass"


def test_unsupported_claim_carries_spans(make_client):
    client = make_client(char_scores("The sky is blue.", flag=(4, 15)))
    resp = verify(client, context=["unrelated", "also unrelated"])
    assert resp.status_code == 200
    body = resp.json()
    claim = body["claims"][0]
    assert claim["verdict"] == "unsupported"
    assert claim["confidence"] == pytest.approx(0.2)
    assert claim["spans"] == [{"start": 4, "end": 15, "text": "sky is blue"}]
    assert body["gate"] == "block"


def test_unsupported_claim_with_no_spans(make_client):
    """The verdict threshold and the span-flagging threshold are independent knobs —
    an unsupported claim with zero spans is legal. Every token risk is 0.6: support
    is 0.4 (below the 0.5 verdict threshold) yet under the 0.9 span threshold."""
    client = make_client(char_scores("The sky is blue.", base=0.6), span_threshold=0.9)
    body = verify(client).json()
    assert body["claims"][0]["verdict"] == "unsupported"
    assert body["claims"][0]["spans"] == []
    assert body["gate"] == "block"


def test_multi_sentence_answer_decomposes_into_claims(make_client):
    """A two-sentence answer returns one claim per sentence, each with answer-relative
    start/end satisfying the substring invariant, tiling the text with no gaps."""
    text = "The sky is blue. Grass is green."
    client = make_client(char_scores(text))
    body = verify(client, text=text).json()
    claims = body["claims"]
    assert [c["text"] for c in claims] == ["The sky is blue. ", "Grass is green."]
    assert [(c["start"], c["end"]) for c in claims] == [(0, 17), (17, 32)]
    assert all(text[c["start"] : c["end"]] == c["text"] for c in claims)
    assert body["gate"] == "pass"


def test_spans_are_claim_relative_in_a_later_claim(make_client):
    """A span in the second sentence is offset into that claim's text, not the answer."""
    text = "The sky is blue. Grass is green."
    client = make_client(char_scores(text, flag=(26, 31)))  # "green" in answer coords
    body = verify(client, text=text).json()
    second = body["claims"][1]
    assert second["verdict"] == "unsupported"
    assert second["spans"] == [{"start": 9, "end": 14, "text": "green"}]
    assert body["gate"] == "block"


def test_no_boundary_answer_is_one_whole_text_claim(make_client):
    client = make_client(char_scores("the sky is blue"))
    body = verify(client, text="the sky is blue").json()
    assert len(body["claims"]) == 1
    assert body["claims"][0]["text"] == "the sky is blue"
    assert (body["claims"][0]["start"], body["claims"][0]["end"]) == (0, 15)


def test_confidence_at_threshold_is_supported(make_client):
    client = make_client(char_scores("The sky is blue.", base=0.5))
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


def test_response_shape_is_exactly_the_contract(make_client):
    """Claims carry calibrated confidence, never the raw score; spans carry positions
    and text only — span confidences wait for #40."""
    client = make_client(char_scores("The sky is blue.", flag=(0, 3)))
    body = verify(client).json()
    assert set(body) == {"claims", "gate", "engine_version", "config_version"}
    assert set(body["claims"][0]) == {"text", "start", "end", "verdict", "confidence", "spans"}
    assert set(body["claims"][0]["spans"][0]) == {"start", "end", "text"}


def test_contradicted_absent_from_vocabulary(make_client):
    """Tier-1 verdicts are exactly supported/unsupported until the NLI signal lands."""
    for base in (0.0, 1.0):
        body = verify(make_client(char_scores("The sky is blue.", base=base))).json()
        assert body["claims"][0]["verdict"] in {"supported", "unsupported"}


def test_confidence_is_calibrated_not_raw(make_client):
    """A non-identity artifact visibly moves the number: with a=1, b=1 a raw support
    of 0.5 becomes sigmoid(1) ≈ 0.731 — and the verdict thresholds that."""
    client = make_client(char_scores("The sky is blue.", base=0.5), a=1.0, b=1.0)
    claim = verify(client).json()["claims"][0]
    assert claim["confidence"] == pytest.approx(0.7310585786300049)
    assert claim["verdict"] == "supported"


def test_calibration_can_flip_the_verdict_at_the_same_raw_score(make_client):
    """The gate thresholds calibrated confidence: a downward map turns a raw 0.6
    into a confidence below the 0.5 threshold."""
    client = make_client(char_scores("The sky is blue.", base=0.4), a=1.0, b=-1.0)
    body = verify(client).json()
    assert body["claims"][0]["verdict"] == "unsupported"
    assert body["gate"] == "block"


def test_confidence_never_exactly_zero_or_one(make_client):
    """Saturated raw supports stay strictly inside (0, 1) — the engine never
    prints certainty."""
    for base, bound in ((0.0, 1.0), (1.0, 0.0)):
        body = verify(make_client(char_scores("The sky is blue.", base=base))).json()
        confidence = body["claims"][0]["confidence"]
        assert confidence != bound
        assert 0.0 < confidence < 1.0


def test_raw_support_goes_to_logs_not_the_response(make_client, capsys):
    """Structured logs carry raw support alongside the calibrated confidence; the
    response carries only the confidence."""
    import json

    client = make_client(char_scores("The sky is blue.", base=0.5), a=1.0, b=1.0)
    body = verify(client).json()
    assert "score" not in body["claims"][0]
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line]
    calibrated = [line for line in lines if line.get("message") == "claims calibrated"]
    assert len(calibrated) == 1
    (logged,) = calibrated[0]["claims"]
    assert logged["raw_support"] == pytest.approx(0.5)
    assert logged["confidence"] == pytest.approx(0.7310585786300049)
    assert calibrated[0]["request_id"]
