"""The thorough-mode retrieval pass: expand, recall, rerank, pool (ADR-0010).

Everything downstream of the pool is fast mode's unchanged scoring path —
mode changes what fills the context window, never the scoring semantics.
"""

from collections.abc import Callable

from engine.config import RetrievalConfig
from engine.retrieval.expansion import expand_queries
from engine.retrieval.pool import Pool, pool_evidence
from engine.retrieval.rerank import Reranker
from engine.retrieval.store import Chunk, EvidenceStore


def retrieve_pool(
    claim_texts: list[str],
    caller_passages: list[str],
    store: EvidenceStore,
    reranker: Reranker,
    cfg: RetrievalConfig,
    count_tokens: Callable[[str], int],
) -> Pool:
    queries = expand_queries(claim_texts, cfg.expansion_window)
    # One recall and one rerank per unique query: duplicate sentences share
    # their query's results.
    recalled: dict[str, list[Chunk]] = {}
    for query in queries:
        if query not in recalled:
            recalled[query] = store.recall(query, cfg.recall_depth)

    unique_queries = list(recalled)
    pairs = [(query, chunk.text) for query in unique_queries for chunk in recalled[query]]
    # One batched pass across all claims' pairs.
    scores = iter(reranker.score(pairs))
    scored: dict[str, list[tuple[Chunk, float]]] = {
        query: [(chunk, next(scores)) for chunk in recalled[query]] for query in unique_queries
    }

    candidates = [scored[query] for query in queries]
    return pool_evidence(
        caller_passages,
        candidates,
        budget_tokens=cfg.pool_budget_tokens,
        per_claim_quota=cfg.per_claim_quota,
        count_tokens=count_tokens,
    )
