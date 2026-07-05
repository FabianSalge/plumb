"""Unit tests for RAGTruth slice construction — pure logic, no downloads."""

import pytest
from bench.data import DataError, Example, is_hallucinated, stratified_slice


def make_example(id_: str, task_type: str) -> Example:
    return Example(
        id=id_,
        task_type=task_type,
        query="q",
        context="c",
        response="r",
        hallucinated=False,
    )


def test_label_derives_from_span_counts():
    assert is_hallucinated({"evident_conflict": 1, "baseless_info": 0})
    assert is_hallucinated({"evident_conflict": 0, "baseless_info": 2})
    assert not is_hallucinated({"evident_conflict": 0, "baseless_info": 0})


def test_label_missing_counts_fails_loudly():
    with pytest.raises(DataError):
        is_hallucinated({"evident_conflict": 1})


def test_stratified_slice_takes_per_task_from_each_type():
    task_types = ("QA", "Summary", "Data2txt")
    examples = [make_example(f"{t}-{i}", t) for t in task_types for i in range(10)]
    sliced = stratified_slice(examples, per_task=3, seed=18)
    assert len(sliced) == 9
    by_task = {t: sum(1 for e in sliced if e.task_type == t) for t in ("QA", "Summary", "Data2txt")}
    assert by_task == {"QA": 3, "Summary": 3, "Data2txt": 3}


def test_stratified_slice_is_deterministic():
    examples = [make_example(f"{t}-{i}", t) for t in ("QA", "Summary") for i in range(20)]
    a = stratified_slice(examples, per_task=5, seed=18)
    b = stratified_slice(list(reversed(examples)), per_task=5, seed=18)
    assert [e.id for e in a] == [e.id for e in b]


def test_stratified_slice_refuses_oversized_request():
    examples = [make_example(f"QA-{i}", "QA") for i in range(2)]
    with pytest.raises(DataError):
        stratified_slice(examples, per_task=3, seed=18)
