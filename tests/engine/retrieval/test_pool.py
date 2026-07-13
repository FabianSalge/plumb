"""Unit tests for evidence pooling: dedupe, caller precedence, quota, fill, budget."""

import logging

from engine.retrieval import Chunk
from engine.retrieval.pool import pool_evidence


def chunk(chunk_id: str, text: str, source: str = "docs", snap: str | None = None) -> Chunk:
    return Chunk(text=text, source_id=source, chunk_id=chunk_id, snapshot_id=snap)


def words(text: str) -> int:
    return len(text.split())


def test_deduped_chunk_appears_once_but_in_both_claims_evidence():
    shared = chunk("1", "one two three")
    pool = pool_evidence(
        caller_passages=[],
        candidates=[[(shared, 0.9)], [(shared, 0.7)]],
        budget_tokens=100,
        per_claim_quota=1,
        count_tokens=words,
    )
    assert pool.passages == ["one two three"]
    assert [ref.chunk_id for ref in pool.evidence[0]] == ["1"]
    assert [ref.chunk_id for ref in pool.evidence[1]] == ["1"]
    assert pool.evidence[0][0].rank == 1
    assert pool.evidence[1][0].rank == 1


def test_caller_passages_come_first_and_are_never_displaced(caplog):
    strong = chunk("1", "alpha beta gamma delta")
    with caplog.at_level(logging.INFO, logger="plumb.engine.retrieval.pool"):
        pool = pool_evidence(
            caller_passages=["caller passage one two"],
            candidates=[[(strong, 99.0)]],
            budget_tokens=4,
            per_claim_quota=1,
            count_tokens=words,
        )
    assert pool.passages == ["caller passage one two"]
    assert pool.evidence == [[]]
    dropped = [record for record in caplog.records if "dropped" in record.getMessage()]
    assert dropped, "budget drop must be logged"


def test_quota_guarantees_the_minority_claims_top_chunk():
    # Claim 0 has two high-scoring chunks; claim 1's best scores far lower.
    a1, a2 = chunk("a1", "one two"), chunk("a2", "three four")
    b1 = chunk("b1", "five six")
    pool = pool_evidence(
        caller_passages=[],
        candidates=[[(a1, 0.9), (a2, 0.8)], [(b1, 0.1)]],
        budget_tokens=4,
        per_claim_quota=1,
        count_tokens=words,
    )
    assert "five six" in pool.passages, "minority claim's guaranteed slot was displaced"
    assert "three four" not in pool.passages


def test_remaining_budget_fills_by_global_rerank_score():
    a1, a2 = chunk("a1", "one two"), chunk("a2", "three four")
    b1, b2 = chunk("b1", "five six"), chunk("b2", "seven eight")
    pool = pool_evidence(
        caller_passages=[],
        candidates=[[(a1, 0.9), (a2, 0.3)], [(b1, 0.8), (b2, 0.7)]],
        budget_tokens=6,
        per_claim_quota=1,
        count_tokens=words,
    )
    # Quota takes a1 and b1; the last two slots go to b2 (0.7) over a2 (0.3).
    assert set(pool.passages) == {"one two", "five six", "seven eight"}


def test_passages_order_is_caller_then_score_descending():
    a1 = chunk("a1", "one two")
    b1 = chunk("b1", "five six")
    pool = pool_evidence(
        caller_passages=["caller text"],
        candidates=[[(a1, 0.2)], [(b1, 0.8)]],
        budget_tokens=100,
        per_claim_quota=1,
        count_tokens=words,
    )
    assert pool.passages == ["caller text", "five six", "one two"]


def test_every_drop_is_logged_with_identity_and_reason(caplog):
    a1, a2 = chunk("a1", "one two"), chunk("a2", "three four")
    with caplog.at_level(logging.INFO, logger="plumb.engine.retrieval.pool"):
        pool_evidence(
            caller_passages=[],
            candidates=[[(a1, 0.9), (a2, 0.8)]],
            budget_tokens=2,
            per_claim_quota=1,
            count_tokens=words,
        )
    drops = [record for record in caplog.records if "dropped" in record.getMessage()]
    assert [(record.chunk_id, record.reason) for record in drops] == [("a2", "budget")]


def test_evidence_rank_is_claim_local_rerank_order():
    a1, a2, a3 = chunk("a1", "one"), chunk("a2", "two"), chunk("a3", "three")
    pool = pool_evidence(
        caller_passages=[],
        candidates=[[(a2, 0.5), (a1, 0.9), (a3, 0.1)]],
        budget_tokens=100,
        per_claim_quota=1,
        count_tokens=words,
    )
    assert [(ref.chunk_id, ref.rank) for ref in pool.evidence[0]] == [
        ("a1", 1),
        ("a2", 2),
        ("a3", 3),
    ]


def test_evidence_carries_snapshot_identity_only_when_present():
    with_snap = chunk("s1", "one", snap="v7")
    without = chunk("s2", "two")
    pool = pool_evidence(
        caller_passages=[],
        candidates=[[(with_snap, 0.9), (without, 0.8)]],
        budget_tokens=100,
        per_claim_quota=2,
        count_tokens=words,
    )
    assert pool.evidence[0][0].snapshot_id == "v7"
    assert pool.evidence[0][1].snapshot_id is None


def test_no_candidates_yields_caller_passages_only():
    pool = pool_evidence(
        caller_passages=["only caller"],
        candidates=[[], []],
        budget_tokens=10,
        per_claim_quota=1,
        count_tokens=words,
    )
    assert pool.passages == ["only caller"]
    assert pool.evidence == [[], []]
