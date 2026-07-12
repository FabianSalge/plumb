"""Shared run-script plumbing: config resolution, environment stamping, progress
printing, percentiles, and results writing. Every `*_run` entry point uses this —
the protocol details (slices, metrics, output shapes) stay in the scripts."""

import json
import platform
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "config" / "verifier.yaml"

# Fixed-pair latency protocol, shared by every model-decision run so the
# short/500-word columns stay comparable across benchmarks and the HHEM spike.
LATENCY_REPEATS = 30
SHORT_EVIDENCE = "The quick brown fox jumps over the lazy dog near the quiet river bank today."
SHORT_CLAIM = "A fox jumps over a dog."
LONG_EVIDENCE = " ".join(
    f"Fact number {i}: the archive records event {i} occurring in year {1500 + i}."
    for i in range(1, 51)
)  # 50 sentences x 10 words = 500 words
LONG_CLAIM = "The archive records event 7 occurring in year 1507."


def percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, round(pct / 100 * len(ordered) + 0.5) - 1)
    return ordered[index]


def environment(*, hardware: bool = False, libs: bool = True) -> dict:
    """Library and platform stamp for a results JSON; `hardware` adds CPU and
    memory (run-level benchmarks state their hardware, calibration runs don't);
    `libs=False` drops torch/transformers (a pure HTTP client doesn't run them)."""
    stamp = {
        "platform": platform.platform(),
        "python": platform.python_version(),
    }
    if libs:
        import torch
        import transformers

        stamp["torch"] = torch.__version__
        stamp["transformers"] = transformers.__version__
    if hardware:
        import psutil

        brand = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True
        ).stdout.strip()
        stamp["cpu"] = brand
        stamp["memory_gb"] = round(psutil.virtual_memory().total / 2**30)
    return stamp


def progress(
    done: int, total: int, *, noun: str = "responses", label: str | None = None, every: int = 50
) -> None:
    """Print a progress line every `every` items; call with 1-based `done`."""
    if done % every == 0:
        prefix = f"  [{label}] " if label else "  "
        print(f"{prefix}{done}/{total} {noun} scored", flush=True)


def write_results(path: Path, result: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=1) + "\n")


def weights_on_disk_bytes(repos: list[tuple[str, str | None]]) -> int | None:
    """Size of each repo's pinned revision in the local cache (not stray revisions)."""
    from huggingface_hub import scan_cache_dir

    cached_models = {c.repo_id: c for c in scan_cache_dir().repos if c.repo_type == "model"}
    total = 0
    for repo, revision in repos:
        if repo not in cached_models:
            return None
        revisions = cached_models[repo].revisions
        matching = [r for r in revisions if revision is None or r.commit_hash == revision]
        if not matching:
            return None
        total += max(r.size_on_disk for r in matching)
    return total
