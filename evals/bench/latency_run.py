"""Measure fast-mode latency end to end through /v1/verify (#36, ADR-0003).

A pure HTTP client against a running server — no model runs in this process,
so wall-clock per request is the full serve path: HTTP, validation, one joint
forward pass, segmentation, calibration, gate. Requests go sequentially (this
measures latency, not throughput) over the seed-18 RAGTruth slice, whole
response as `text`, its source as the single context passage, mode "fast".
Warmup requests are sent first and excluded from the stats.

Usage: serve the API on this machine (`make run` from the repo root), then
from evals/:
    uv run python -m bench.latency_run --out results/latency-fast-lettucedetect-v2.json
"""

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

from bench.data import load_ragtruth_test, stratified_slice
from bench.harness import environment, progress, write_results
from bench.latency import LatencyError, check_identity, summarize, verify_payload

# ADR-0003: fast mode's contract is sub-second p95.
CONTRACT_P95_MS = 1000.0


def post_verify(url: str, payload: dict) -> dict:
    request = urllib.request.Request(
        f"{url}/v1/verify",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as exc:
        raise LatencyError(f"/v1/verify returned {exc.code}: {exc.read().decode()}") from exc


def require_ready(url: str) -> None:
    try:
        with urllib.request.urlopen(f"{url}/readyz") as response:
            if response.status != 200:
                raise LatencyError(f"server not ready: /readyz returned {response.status}")
    except urllib.error.URLError as exc:
        raise LatencyError(f"no server at {url} — start it with `make run` first") from exc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--per-task", type=int, default=200)
    parser.add_argument("--seed", type=int, default=18)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    require_ready(args.url)
    examples = stratified_slice(load_ragtruth_test(), per_task=args.per_task, seed=args.seed)
    print(f"slice: {len(examples)} responses", flush=True)

    for _ in range(args.warmup):
        post_verify(args.url, verify_payload(examples[0]))

    identity: tuple[str, str] | None = None
    millis: list[float] = []
    claims_per_response: list[int] = []
    for i, example in enumerate(examples):
        start = time.perf_counter()
        body = post_verify(args.url, verify_payload(example))
        millis.append((time.perf_counter() - start) * 1000)
        identity = check_identity(body, expected=identity)
        claims_per_response.append(len(body["claims"]))
        progress(i + 1, len(examples), every=50)

    assert identity is not None
    result = {
        "mode": "fast",
        "engine_version": identity[0],
        "config_version": identity[1],
        "slice": {
            "per_task": args.per_task,
            "seed": args.seed,
            "responses": len(examples),
            "warmup_requests": args.warmup,
        },
        "latency": summarize(millis, contract_ms=CONTRACT_P95_MS),
        "claims_per_response": {
            "mean": sum(claims_per_response) / len(claims_per_response),
            "max": max(claims_per_response),
        },
        # The machine stamp is the server's too — same host; the model runs in
        # the server process (`make run`, root project's `model` extra).
        "environment": environment(hardware=True, libs=False),
    }

    write_results(args.out, result)
    print(json.dumps({k: result[k] for k in ("slice", "latency", "claims_per_response")}, indent=1))


if __name__ == "__main__":
    main()
