"""Deterministic query expansion: claim plus neighboring answer sentences (ADR-0010).

Queries are ephemeral — they never appear in the response, so the claim text
and ADR-0009's substring invariant stay untouched. This is expansion, not
rewriting: no model, no paraphrase, and golden tests pin the exact strings.
"""


def expand_queries(sentences: list[str], window: int) -> list[str]:
    """One query per claim: the answer's leading sentence, up to `window`
    immediately preceding sentences, then the claim itself — deduplicated
    preserving answer order and joined with single spaces."""
    stripped = [sentence.strip() for sentence in sentences]
    queries: list[str] = []
    for i, claim in enumerate(stripped):
        components = [stripped[0], *stripped[max(0, i - window) : i], claim]
        unique: list[str] = []
        for component in components:
            if component not in unique:
                unique.append(component)
        queries.append(" ".join(unique))
    return queries
