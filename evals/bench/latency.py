"""Fast-mode latency measurement logic (issue #36, ADR-0003).

The end-to-end bench sends real HTTP requests through `/v1/verify`; this module
holds the testable pieces — request construction, the latency summary judged
against the ADR-0003 contract, and the per-response sanity check that the
server stayed the same server for the whole run.
"""

import statistics
from collections.abc import Mapping

from bench.data import Example
from bench.harness import percentile


class LatencyError(Exception):
    """The run cannot produce an honest number — stop, don't publish garbage."""


def verify_payload(example: Example) -> dict:
    """The `/v1/verify` request for one RAGTruth response: whole answer, fast mode."""
    return {"text": example.response, "context": [example.context], "mode": "fast"}


def summarize(millis: list[float], contract_ms: float) -> dict:
    """p50/p95 over per-request wall-clock, judged against the latency contract."""
    if not millis:
        raise LatencyError("no timed requests — nothing to summarize")
    p95 = percentile(millis, 95)
    return {
        "n": len(millis),
        "p50_ms": statistics.median(millis),
        "p95_ms": p95,
        "contract_ms": contract_ms,
        "meets_contract": p95 < contract_ms,
    }


def check_identity(body: Mapping, expected: tuple[str, str] | None) -> tuple[str, str]:
    """Assert one response is well-formed and from the same server as the rest.

    Returns (engine_version, config_version); pass the previous return value as
    `expected` so a mid-run restart or config rollout aborts the measurement.
    """
    try:
        identity = (body["engine_version"], body["config_version"])
        claims = body["claims"]
    except KeyError as exc:
        raise LatencyError(f"response missing field: {exc}") from exc
    if not claims:
        raise LatencyError("response carries no claims — the server did no work")
    if expected is not None and identity != expected:
        raise LatencyError(f"server identity drifted mid-run: {expected} -> {identity}")
    return identity
