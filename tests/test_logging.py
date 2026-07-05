"""Structured JSON logging and request-ID propagation tests."""

import json


def request_log_lines(capfd) -> list[dict]:
    out, _ = capfd.readouterr()
    lines = []
    for line in out.splitlines():
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "request_id" in parsed:
            lines.append(parsed)
    return lines


def test_log_lines_are_json_with_caller_request_id(client, capfd):
    client.get("/healthz", headers={"X-Request-ID": "abc-123"})
    lines = request_log_lines(capfd)
    assert lines, "expected at least one JSON log line carrying a request_id"
    assert all(line["request_id"] == "abc-123" for line in lines)


def test_request_id_generated_when_absent(client, capfd):
    response = client.get("/healthz")
    lines = request_log_lines(capfd)
    assert lines
    generated = lines[-1]["request_id"]
    assert generated
    assert response.headers["X-Request-ID"] == generated


def test_access_line_carries_request_fields(client, capfd):
    client.post(
        "/v1/verify",
        json={"text": "claim", "context": ["evidence"], "mode": "fast"},
        headers={"X-Request-ID": "req-9"},
    )
    access = [line for line in request_log_lines(capfd) if line.get("path") == "/v1/verify"]
    assert access, "expected an access log line for /v1/verify"
    line = access[-1]
    assert line["request_id"] == "req-9"
    assert line["method"] == "POST"
    assert line["status"] == 200
