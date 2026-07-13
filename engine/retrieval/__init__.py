"""Thorough-mode retrieval: per-claim queries, recall-then-rerank, pooled evidence (ADR-0010)."""

from engine.retrieval.expansion import expand_queries
from engine.retrieval.store import Chunk, EvidenceStore, StoreError

__all__ = ["Chunk", "EvidenceStore", "StoreError", "expand_queries"]
