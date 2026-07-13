"""Thorough-mode latency and calibration error end to end through /v1/verify (#66, ADR-0010).

One HTTP pass over the seed-18 RAGTruth slice serves both numbers:

- Latency: wall-clock per request against ADR-0010's p95 <= 10 s target. The
  request carries only `text` — retrieval does the evidence work: per-claim
  queries, Postgres FTS recall over the seeded corpus (bench.store_seed),
  cross-encoder rerank, pooling, one joint scoring pass.
- Calibration: the response's calibrated per-claim confidences against
  RAGTruth's sentence outcomes (supported iff the claim overlaps no annotated
  span). This is the honest-caveat measurement ADR-0010 demands: the artifact
  was fitted on provided-context traffic, and retrieved evidence is a
  different passage distribution. Caveat on the labels themselves: RAGTruth
  annotates support against each response's own source document, so a claim
  another corpus document happens to support still counts unsupported.

Usage: seed the store (bench.store_seed), serve the API with the store env
(see RESULTS.md), then from evals/:
    uv run python -m bench.thorough_run --out results/thorough-v1.json
"""

import argparse
import json
import time
import urllib.error
import urllib.request
from dataclasses import asdict
from pathlib import Path

from bench.data import Example, load_ragtruth_test, stratified_slice
from bench.harness import environment, progress, write_results
from bench.latency import LatencyError, check_identity, summarize
from bench.metrics import ece, reliability_bins

# ADR-0010: thorough mode's latency target, covering the full signal stack.
CONTRACT_P95_MS = 10_000.0


def thorough_payload(example: Example) -> dict:
    """Retrieval-only traffic: no caller context, the store supplies the evidence."""
    return {"text": example.response, "mode": "thorough"}


def post_verify(url: str, payload: dict) -> dict:
    request = urllib.request.Request(
        f"{url}/v1/verify",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as exc:
        raise LatencyError(f"/v1/verify returned {exc.code}: {exc.read().decode()}") from exc


def require_ready(url: str) -> None:
    try:
        with urllib.request.urlopen(f"{url}/readyz") as response:
            if response.status != 200:
                raise LatencyError(f"server not ready: /readyz returned {response.status}")
    except urllib.error.URLError as exc:
        raise LatencyError(f"no server at {url} — serve with the store env first") from exc


def claim_outcome(claim: dict, spans: tuple[tuple[int, int], ...]) -> int:
    """1 (supported) iff the claim overlaps no annotated hallucination span —
    the same labelling rule the calibration fit used."""
    overlaps = any(start < claim["end"] and claim["start"] < end for start, end in spans)
    return 0 if overlaps else 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--per-task", type=int, default=200)
    parser.add_argument("--seed", type=int, default=18)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    require_ready(args.url)
    examples = stratified_slice(load_ragtruth_test(), per_task=args.per_task, seed=args.seed)
    print(f"slice: {len(examples)} responses", flush=True)

    for _ in range(args.warmup):
        post_verify(args.url, thorough_payload(examples[0]))

    identity: tuple[str, str] | None = None
    millis: list[float] = []
    outcomes: list[int] = []
    confidences: list[float] = []
    evidence_counts: list[int] = []
    for i, example in enumerate(examples):
        start = time.perf_counter()
        body = post_verify(args.url, thorough_payload(example))
        millis.append((time.perf_counter() - start) * 1000)
        identity = check_identity(body, expected=identity)
        for claim in body["claims"]:
            outcomes.append(claim_outcome(claim, example.spans))
            confidences.append(claim["confidence"])
            evidence_counts.append(len(claim.get("evidence", [])))
        progress(i + 1, len(examples), every=10)

    assert identity is not None
    result = {
        "bench": "thorough-mode latency + calibration on retrieved evidence (#66, ADR-0010)",
        "engine_version": identity[0],
        "config_version": identity[1],
        "slice": {"per_task": args.per_task, "seed": args.seed, "responses": len(examples)},
        "latency": summarize(millis, CONTRACT_P95_MS),
        "calibration": {
            "claims": len(outcomes),
            "ece": ece(outcomes, confidences),
            "bins": [asdict(bin_) for bin_ in reliability_bins(outcomes, confidences)],
            "labels": "RAGTruth sentence outcomes against each response's own source",
        },
        "evidence": {
            "claims_with_evidence": sum(1 for count in evidence_counts if count),
            "mean_refs_per_claim": (
                sum(evidence_counts) / len(evidence_counts) if evidence_counts else 0.0
            ),
        },
        "environment": environment(hardware=True),
    }
    write_results(args.out, result)
    print(json.dumps({k: result[k] for k in ("latency", "calibration", "evidence")}, indent=2))


if __name__ == "__main__":
    main()
