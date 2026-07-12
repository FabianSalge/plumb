"""RAGTruth test-split loading and deterministic slice construction.

Source: wandb/RAGTruth-processed (MIT), the RAGTruth corpus with per-response
hallucination span counts. Label: a response is hallucinated iff it carries at
least one annotated span (evident conflict or baseless info).
"""

import json
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

# Published split statistics for wandb/RAGTruth-processed — the load fails
# loudly if the download does not match them.
EXPECTED_TEST_SIZE = 2700
EXPECTED_TEST_POSITIVES = 943


class DataError(Exception):
    """The dataset is not what the benchmark expects — stop, don't benchmark garbage."""


@dataclass(frozen=True)
class Example:
    id: str
    task_type: str
    query: str
    context: str
    response: str
    hallucinated: bool
    # Annotated hallucination spans as (start, end) code-point offsets into
    # `response`; empty for a supported response. Sentence-level labelling reads
    # these, the response-level label reads only whether any exist.
    spans: tuple[tuple[int, int], ...] = field(default=())
    # Parallel to `spans`: "conflict" (the span contradicts the evidence) or
    # "baseless" (it merely lacks support) — the distinction the NLI slot exists for.
    span_kinds: tuple[str, ...] = field(default=())


def is_hallucinated(span_counts: Mapping[str, int]) -> bool:
    try:
        return span_counts["evident_conflict"] + span_counts["baseless_info"] > 0
    except KeyError as exc:
        raise DataError(f"hallucination span counts missing key: {exc}") from exc


# RAGTruth's four annotation types collapse into the two kinds the signal stack
# distinguishes: a conflict span is refuted by the evidence, a baseless span
# merely lacks support.
_KIND_BY_LABEL_TYPE = {
    "Evident Conflict": "conflict",
    "Subtle Conflict": "conflict",
    "Evident Baseless Info": "baseless",
    "Subtle Baseless Info": "baseless",
}


def span_kind(label_type: str) -> str:
    try:
        return _KIND_BY_LABEL_TYPE[label_type]
    except KeyError as exc:
        raise DataError(f"unknown hallucination label_type: {label_type!r}") from exc


def hallucination_spans(raw: str) -> tuple[tuple[int, int, str], ...]:
    """Parse the raw `hallucination_labels` JSON into (start, end, kind) triples,
    with response code-point offsets and kind "conflict" or "baseless"."""
    try:
        items = json.loads(raw)
    except (json.JSONDecodeError, TypeError) as exc:
        raise DataError(f"hallucination_labels is not valid JSON: {exc}") from exc
    spans: list[tuple[int, int, str]] = []
    for item in items:
        try:
            spans.append((int(item["start"]), int(item["end"]), span_kind(item["label_type"])))
        except (KeyError, TypeError, ValueError) as exc:
            raise DataError(f"hallucination span missing start/end/label_type: {exc}") from exc
    return tuple(spans)


def stratified_slice(examples: Sequence[Example], per_task: int, seed: int) -> list[Example]:
    """Draw per_task examples from each task type, deterministically for a given seed."""
    by_task: dict[str, list[Example]] = {}
    for example in examples:
        by_task.setdefault(example.task_type, []).append(example)

    sliced: list[Example] = []
    for task_type in sorted(by_task):
        group = sorted(by_task[task_type], key=lambda e: e.id)
        if len(group) < per_task:
            raise DataError(f"task type {task_type!r} has {len(group)} examples, need {per_task}")
        random.Random(seed).shuffle(group)
        sliced.extend(group[:per_task])
    return sorted(sliced, key=lambda e: (e.task_type, e.id))


def load_ragtruth_test() -> list[Example]:
    from datasets import load_dataset

    rows = load_dataset("wandb/RAGTruth-processed", split="test")
    examples = []
    for row in rows:
        typed = hallucination_spans(row["hallucination_labels"])
        examples.append(
            Example(
                id=row["id"],
                task_type=row["task_type"],
                query=row["query"],
                context=row["context"],
                response=row["output"],
                hallucinated=is_hallucinated(row["hallucination_labels_processed"]),
                spans=tuple((start, end) for start, end, _ in typed),
                span_kinds=tuple(kind for _, _, kind in typed),
            )
        )
    positives = sum(e.hallucinated for e in examples)
    if len(examples) != EXPECTED_TEST_SIZE or positives != EXPECTED_TEST_POSITIVES:
        raise DataError(
            f"RAGTruth test split mismatch: {len(examples)} examples "
            f"({positives} hallucinated), expected {EXPECTED_TEST_SIZE} "
            f"({EXPECTED_TEST_POSITIVES} hallucinated)"
        )
    return examples
