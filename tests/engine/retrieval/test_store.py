"""Unit tests for the store adapter seam: Chunk identity and the recall protocol."""

import pytest

from engine.retrieval import Chunk, EvidenceStore, StoreError


class FakeStore:
    """In-memory EvidenceStore: maps query substrings to canned chunks."""

    def __init__(self, results: dict[str, list[Chunk]]) -> None:
        self._results = results

    def recall(self, query: str, k: int) -> list[Chunk]:
        for needle, chunks in self._results.items():
            if needle in query:
                return chunks[:k]
        return []


def make_chunk(chunk_id: str = "1", **overrides: object) -> Chunk:
    fields: dict[str, object] = {
        "text": "The Eiffel Tower is 330 metres tall.",
        "source_id": "docs",
        "chunk_id": chunk_id,
    }
    fields.update(overrides)
    return Chunk(**fields)  # type: ignore[arg-type]


def test_chunk_carries_identity_and_text():
    chunk = make_chunk(source_id="wiki/eiffel", chunk_id="42")
    assert chunk.text == "The Eiffel Tower is 330 metres tall."
    assert chunk.source_id == "wiki/eiffel"
    assert chunk.chunk_id == "42"


def test_snapshot_identity_is_absent_unless_the_store_exposes_one():
    assert make_chunk().snapshot_id is None
    assert make_chunk(snapshot_id="v7").snapshot_id == "v7"


def test_chunk_is_immutable():
    chunk = make_chunk()
    with pytest.raises(AttributeError):
        chunk.text = "rewritten"  # type: ignore[misc]


def test_fake_store_satisfies_the_protocol():
    store: EvidenceStore = FakeStore({})
    assert isinstance(store, EvidenceStore)


def test_recall_respects_k():
    chunks = [make_chunk(chunk_id=str(i)) for i in range(5)]
    store = FakeStore({"eiffel": chunks})
    assert len(store.recall("the eiffel tower", k=2)) == 2


def test_store_error_is_an_exception_with_context():
    with pytest.raises(StoreError, match="unreachable"):
        raise StoreError("store unreachable at db:5432")
