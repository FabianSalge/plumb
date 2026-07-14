"""Thorough-mode retrieval: per-claim queries, recall-then-rerank, pooled evidence (ADR-0010)."""

from engine.retrieval.expansion import expand_queries
from engine.retrieval.rerank import Reranker
from engine.retrieval.store import Chunk, EvidenceStore, StoreError

__all__ = ["Chunk", "EvidenceStore", "Reranker", "StoreError", "expand_queries"]
