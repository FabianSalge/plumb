"""The store adapter seam: read-only recall against the tenant's knowledge base.

ADR-0002 fixes the architecture as recall-then-rerank; adapters implement
recall only, so the interface has no write path to misuse. Identity travels
with every chunk — snapshot identity only where the store exposes one, never
invented (it is the verdict-pinning seed, and a fabricated one would lie).
"""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


class StoreError(Exception):
    """The tenant store failed — connection, query, or schema. Always loud:
    an erroring store is a failed verification, never an empty result."""


@dataclass(frozen=True)
class Chunk:
    text: str
    # Where the chunk came from (configured source column, or the table name).
    source_id: str
    # The chunk's identity within its source (the configured id column).
    chunk_id: str
    # Store snapshot/version identity, only where the store exposes one.
    snapshot_id: str | None = None


@runtime_checkable
class EvidenceStore(Protocol):
    def recall(self, query: str, k: int) -> list[Chunk]:
        """Top-k lexical recall for one query, read-only, ranked by the store."""
        ...
