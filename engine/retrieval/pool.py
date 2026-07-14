"""Evidence pooling: dedupe, caller precedence, per-claim quota, global fill.

The pool — never the scorer's silent window — decides what gets scored
(ADR-0010): the budget is counted in scoring-tokenizer tokens with headroom
for the prompt template and the answer reserved by config, and every chunk
dropped at any step is logged with its identity and reason. The window quota
prices long answers honestly: many claims leave each claim little beyond its
guaranteed slot.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass

from engine.retrieval.store import Chunk

logger = logging.getLogger("plumb.engine.retrieval.pool")


@dataclass(frozen=True)
class EvidenceRef:
    """Retrieval provenance for one claim: 'retrieved for', not 'supports'."""

    source_id: str
    chunk_id: str
    # Claim-local rerank rank, 1-based: this chunk was this claim's rank-th
    # best candidate among those that made the scoring window.
    rank: int
    snapshot_id: str | None = None


@dataclass(frozen=True)
class Pool:
    # Caller passages first (never displaced), then chunks by global rerank
    # score descending — the deterministic order the scorer sees.
    passages: list[str]
    # Per claim, aligned with the candidates input: references to the chunks
    # its query retrieved that made the window, in claim-local rank order.
    evidence: list[list[EvidenceRef]]


def _identity(chunk: Chunk) -> tuple[str, str]:
    return (chunk.source_id, chunk.chunk_id)


def pool_evidence(
    caller_passages: list[str],
    candidates: list[list[tuple[Chunk, float]]],
    *,
    budget_tokens: int,
    per_claim_quota: int,
    count_tokens: Callable[[str], int],
) -> Pool:
    """Pool per-claim reranked candidates into one budgeted evidence set.

    `candidates` holds (chunk, rerank score) per claim, any order; scores are
    comparable across claims only approximately, which is why every claim's
    top chunks get guaranteed slots before the global fill.
    """
    ranked = [
        sorted(claim_candidates, key=lambda pair: (-pair[1], _identity(pair[0])))
        for claim_candidates in candidates
    ]

    budget = budget_tokens
    passages: list[str] = []
    for i, passage in enumerate(caller_passages):
        cost = count_tokens(passage)
        if cost > budget:
            logger.warning(
                "caller passage dropped: evidence pool budget exhausted",
                extra={"passage_index": i, "reason": "budget"},
            )
            continue
        budget -= cost
        passages.append(passage)

    best_score: dict[tuple[str, str], tuple[float, Chunk]] = {}
    for claim_candidates in ranked:
        for chunk, score in claim_candidates:
            key = _identity(chunk)
            if key not in best_score or score > best_score[key][0]:
                best_score[key] = (score, chunk)

    selected: set[tuple[str, str]] = set()

    def try_select(chunk: Chunk, reason_owner: str) -> None:
        nonlocal budget
        key = _identity(chunk)
        if key in selected:
            return
        cost = count_tokens(chunk.text)
        if cost > budget:
            logger.warning(
                "chunk dropped: evidence pool budget exhausted",
                extra={
                    "source_id": chunk.source_id,
                    "chunk_id": chunk.chunk_id,
                    "reason": "budget",
                    "stage": reason_owner,
                },
            )
            return
        budget -= cost
        selected.add(key)

    # Guaranteed slots: every claim's top-quota chunks, claims in answer order.
    for claim_candidates in ranked:
        for chunk, _score in claim_candidates[:per_claim_quota]:
            try_select(chunk, "quota")

    # Global fill: remaining candidates by rerank score, best first.
    fill = sorted(best_score.values(), key=lambda pair: (-pair[0], _identity(pair[1])))
    for _score, chunk in fill:
        try_select(chunk, "fill")

    ordered = [chunk for _score, chunk in fill if _identity(chunk) in selected]
    passages.extend(chunk.text for chunk in ordered)

    evidence: list[list[EvidenceRef]] = []
    for claim_candidates in ranked:
        refs = [
            EvidenceRef(
                source_id=chunk.source_id,
                chunk_id=chunk.chunk_id,
                rank=position + 1,
                snapshot_id=chunk.snapshot_id,
            )
            for position, (chunk, _score) in enumerate(claim_candidates)
            if _identity(chunk) in selected
        ]
        evidence.append(refs)
    return Pool(passages=passages, evidence=evidence)
