"""Unit tests for LLM-AggreFact loading logic — pure functions, no downloads."""

import pytest
from bench.aggrefact import (
    ClaimExample,
    check_training_overlap,
    normalize_source_name,
    stratified_claims,
)
from bench.data import DataError


def claim(subset: str, index: int) -> ClaimExample:
    return ClaimExample(
        id=f"{subset}:{index}",
        subset=subset,
        doc=f"doc {index}",
        claim=f"claim {index}",
        supported=index % 2 == 0,
    )


def test_normalize_source_name_strips_case_and_punctuation():
    assert normalize_source_name("AggreFact-CNN") == "aggrefactcnn"
    assert normalize_source_name("FactCheck-GPT") == "factcheckgpt"
    assert normalize_source_name("rag_truth") == "ragtruth"


def test_overlap_detected_both_directions():
    # a training source containing the subset name, and vice versa, both flag
    overlap = check_training_overlap(["Wice", "Reveal"], {"wice_v2", "psiloqa"})
    assert overlap == {"Wice": "wice_v2"}
    overlap = check_training_overlap(["TofuEval-MediaSum"], {"tofueval"})
    assert overlap == {"TofuEval-MediaSum": "tofueval"}


def test_no_overlap_is_empty():
    assert check_training_overlap(["Wice", "Lfqa"], {"psiloqa", "ragtruth"}) == {}


def test_stratified_claims_is_deterministic_and_balanced():
    examples = [claim("A", i) for i in range(30)] + [claim("B", i) for i in range(30)]
    sliced = stratified_claims(examples, per_subset=10, seed=18)
    assert len(sliced) == 20
    assert sum(1 for e in sliced if e.subset == "A") == 10
    assert sliced == stratified_claims(examples, per_subset=10, seed=18)


def test_stratified_claims_fails_on_small_subset():
    examples = [claim("A", i) for i in range(3)]
    with pytest.raises(DataError):
        stratified_claims(examples, per_subset=10, seed=18)
