"""LLM-AggreFact loading for the out-of-domain calibration check (ADR-0008).

Source: lytang/LLM-AggreFact (gated — needs an authenticated HF session), a
benchmark of (document, claim, label) triples from eleven grounding datasets.
The RAGTruth subset is always excluded (the calibrator is fitted on RAGTruth),
and every remaining subset is checked against the pinned model's actual
training mix — the `dataset` column of the KR Labs unified training sets — at
load time. Overlapping subsets are excluded and the exclusion is recorded, per
the calibration spec: nothing evaluated may have fed the model or the fit.
"""

import random
from collections.abc import Sequence
from dataclasses import dataclass

from bench.data import DataError

AGGREFACT_REPO = "lytang/LLM-AggreFact"
# Published test-split size of lytang/LLM-AggreFact — the load fails loudly on drift.
EXPECTED_TEST_SIZE = 29320
# The pinned model card's named training data: the KR Labs unified sets. Their
# `dataset` column names every constituent source, which is what we check against.
TRAINING_MIX_REPOS = (
    "KRLabsOrg/lettucedetect-prose-hallucination",
    "KRLabsOrg/lettucedetect-code-hallucination",
)
# Fitted on RAGTruth by construction — excluded before any overlap check runs.
ALWAYS_EXCLUDED = ("RAGTruth",)


@dataclass(frozen=True)
class ClaimExample:
    """One LLM-AggreFact row: `claim` checked against `doc`, at the benchmark's own
    claim granularity. `supported` is the human label (1 = supported)."""

    id: str
    subset: str
    doc: str
    claim: str
    supported: bool


def normalize_source_name(name: str) -> str:
    """Lower-case and strip non-alphanumerics so naming conventions can't hide overlap."""
    return "".join(char for char in name.lower() if char.isalnum())


def check_training_overlap(subsets: Sequence[str], training_sources: set[str]) -> dict[str, str]:
    """Map each LLM-AggreFact subset to the training source it collides with, if any.
    A collision is a substring match in either direction after normalization."""
    overlap: dict[str, str] = {}
    for subset in subsets:
        normalized_subset = normalize_source_name(subset)
        for source in sorted(training_sources):
            normalized_source = normalize_source_name(source)
            if normalized_subset in normalized_source or normalized_source in normalized_subset:
                overlap[subset] = source
                break
    return overlap


def load_training_mix_sources() -> set[str]:
    """Distinct source-dataset names across the pinned model's training repos."""
    from datasets import load_dataset

    sources: set[str] = set()
    for repo in TRAINING_MIX_REPOS:
        rows = load_dataset(repo, split="train", columns=["dataset"])
        sources.update(rows.unique("dataset"))
    if not sources:
        raise DataError("training-mix repos yielded no source-dataset names")
    return sources


def load_aggrefact_test() -> tuple[list[ClaimExample], dict[str, str]]:
    """Load the LLM-AggreFact test split with RAGTruth and every training-mix-overlapping
    subset excluded. Returns the examples and the record of exclusions (subset -> reason)."""
    from datasets import load_dataset

    rows = load_dataset(AGGREFACT_REPO, split="test")
    if len(rows) != EXPECTED_TEST_SIZE:
        raise DataError(
            f"LLM-AggreFact test split mismatch: {len(rows)} rows, expected {EXPECTED_TEST_SIZE}"
        )

    subsets = sorted(set(rows["dataset"]))
    overlap = check_training_overlap(
        [s for s in subsets if s not in ALWAYS_EXCLUDED], load_training_mix_sources()
    )
    exclusions = {subset: "fitted on RAGTruth" for subset in ALWAYS_EXCLUDED}
    exclusions.update(
        {subset: f"in the model's training mix ({source})" for subset, source in overlap.items()}
    )

    examples = [
        ClaimExample(
            id=f"{row['dataset']}:{i}",
            subset=row["dataset"],
            doc=row["doc"],
            claim=row["claim"],
            supported=bool(row["label"]),
        )
        for i, row in enumerate(rows)
        if row["dataset"] not in exclusions
    ]
    if not examples:
        raise DataError("every LLM-AggreFact subset was excluded — nothing to evaluate")
    return examples, exclusions


def stratified_claims(
    examples: Sequence[ClaimExample], per_subset: int, seed: int
) -> list[ClaimExample]:
    """Draw per_subset examples from each subset, deterministically for a given seed."""
    by_subset: dict[str, list[ClaimExample]] = {}
    for example in examples:
        by_subset.setdefault(example.subset, []).append(example)

    sliced: list[ClaimExample] = []
    for subset in sorted(by_subset):
        group = sorted(by_subset[subset], key=lambda e: e.id)
        if len(group) < per_subset:
            raise DataError(f"subset {subset!r} has {len(group)} examples, need {per_subset}")
        random.Random(seed).shuffle(group)
        sliced.extend(group[:per_subset])
    return sorted(sliced, key=lambda e: (e.subset, e.id))
