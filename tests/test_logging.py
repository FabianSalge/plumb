"""Structured JSON logging and request-ID propagation tests."""

import json
import logging

from api.logging import JSONFormatter, request_id_var


def access_records(caplog) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.name == "plumb.request"]


def test_access_record_carries_caller_request_id(client, caplog):
    with caplog.at_level(logging.INFO):
        client.get("/healthz", headers={"X-Request-ID": "abc-123"})
    assert [r.request_id for r in access_records(caplog)] == ["abc-123"]


def test_request_id_generated_when_absent(client, caplog):
    with caplog.at_level(logging.INFO):
        response = client.get("/healthz")
    generated = access_records(caplog)[-1].request_id
    assert generated
    assert response.headers["X-Request-ID"] == generated


def test_access_record_carries_request_fields(client, caplog):
    with caplog.at_level(logging.INFO):
        client.post(
            "/v1/verify",
            json={"text": "claim", "context": ["evidence"], "mode": "fast"},
            headers={"X-Request-ID": "req-9"},
        )
    record = [r for r in access_records(caplog) if r.path == "/v1/verify"][-1]
    assert (record.request_id, record.method, record.status) == ("req-9", "POST", 200)


def test_formatted_lines_are_json_and_carry_the_request_id():
    """Any log line emitted during a request picks up the request ID via the
    contextvar — not just the access line."""
    token = request_id_var.set("ctx-42")
    try:
        record = logging.LogRecord(
            "plumb.api", logging.INFO, __file__, 1, "loading scoring model", None, None
        )
        line = json.loads(JSONFormatter().format(record))
    finally:
        request_id_var.reset(token)
    assert line["message"] == "loading scoring model"
    assert line["request_id"] == "ctx-42"
    assert line["level"] == "INFO"
    assert line["logger"] == "plumb.api"
