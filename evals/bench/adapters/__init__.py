"""Candidate model adapters.

Each adapter exposes NAME, REPO, REVISION and a load() function returning an
object with support_score(example) -> float in [0, 1], higher = more grounded.
Every model sees its own documented input protocol; the common currency is the
response-level support score.
"""
