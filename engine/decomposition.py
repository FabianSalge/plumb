"""Sentence decomposition and segment-after-score reduction (ADR-0009).

A claim is a verbatim sentence of the answer. `segment` partitions the answer
into claims by deterministic rule-based sentence segmentation — no model, pinned
by golden tests. `reduce_claim` scores one claim off the whole-answer forward
pass: support = 1 − the maximum token risk overlapping the claim's range, with
spans clipped to the claim and kept claim-relative. Because segmentation is a
total partition and a boundary-straddling token counts toward every claim it
overlaps, the maximum over per-claim risks equals the whole-answer maximum, so
the gate's decision boundary provably does not move.
"""

import logging
import re
from dataclasses import asdict, dataclass

from engine.scoring import TokenScores

logger = logging.getLogger("plumb.engine.decomposition")


class DecompositionError(Exception):
    """Segmentation produced claims that do not honor the substring invariant, or
    a claim's support fell outside [0, 1] — a bug the gate must not run on."""


@dataclass(frozen=True)
class Claim:
    """A verbatim sentence of the answer; `start`/`end` are answer-relative Unicode
    code-point offsets with `text == answer[start:end]`."""

    text: str
    start: int
    end: int


@dataclass(frozen=True)
class Span:
    """An unsupported region of a claim; `start`/`end` are claim-relative code-point
    offsets into the claim text. `confidence` is the raw maximum token probability —
    structured-log detail only until calibration (#32)."""

    start: int
    end: int
    text: str
    confidence: float


@dataclass(frozen=True)
class ScoredClaim:
    """One claim with its union support and the spans flagged as unsupported."""

    text: str
    start: int
    end: int
    support: float
    spans: list[Span]


# Tokens whose trailing period does not end a sentence. Lower-cased, dots kept for
# the acronym forms the segmenter matches (e.g / i.e / u.s / a.m / p.m).
_ABBREVIATIONS = frozenset(
    {
        "dr",
        "mr",
        "mrs",
        "ms",
        "prof",
        "sr",
        "jr",
        "st",
        "vs",
        "etc",
        "al",
        "no",
        "vol",
        "fig",
        "inc",
        "ltd",
        "co",
        "corp",
        "dept",
        "est",
        "gen",
        "gov",
        "sen",
        "rep",
        "capt",
        "lt",
        "col",
        "sgt",
        "approx",
        "min",
        "max",
        "cf",
        "pp",
        "ed",
        "eds",
        "repr",
        "trans",
        "rev",
        "jan",
        "feb",
        "mar",
        "apr",
        "jun",
        "jul",
        "aug",
        "sep",
        "sept",
        "oct",
        "nov",
        "dec",
        "e.g",
        "i.e",
        "u.s",
        "u.k",
        "a.m",
        "p.m",
    }
)

_FENCE = re.compile(r"```.*?```", re.DOTALL)
_LIST_MARKER = re.compile(r"(?:[-*•]|\d+[.)])\s")


def segment(text: str) -> list[Claim]:
    """Partition `text` into verbatim sentence claims. The claims tile `[0, len(text))`
    with no gaps; text with no detectable boundary yields one whole-text claim."""
    if text == "":
        return [Claim(text="", start=0, end=0)]
    bounds = [*_split_points(text), len(text)]
    claims = [
        Claim(text=text[a:b], start=a, end=b)
        for a, b in zip(bounds, bounds[1:], strict=False)
        if a < b
    ]
    _validate_partition(text, claims)
    return claims


def reduce_claim(claim: Claim, scores: TokenScores, span_threshold: float) -> ScoredClaim:
    """Score one claim off the whole-answer pass: support = 1 − max risk over the
    tokens overlapping the claim, spans clipped to the claim and kept claim-relative."""
    risks: list[float] = []
    overlapping: list[tuple[float, tuple[int, int]]] = []
    for prob, (ts, te) in zip(scores.probs, scores.offsets, strict=True):
        if ts == te:  # zero-width special token — covers no answer characters
            continue
        if ts < claim.end and te > claim.start:  # overlaps the claim's range
            risks.append(prob)
            rel_start = max(ts, claim.start) - claim.start
            rel_end = min(te, claim.end) - claim.start
            overlapping.append((prob, (rel_start, rel_end)))

    support = 1.0 - max(risks) if risks else 1.0
    if not 0.0 <= support <= 1.0:
        raise DecompositionError(f"claim support {support} outside [0, 1]")

    spans = _merge_spans(claim.text, overlapping, span_threshold)
    if spans:
        logger.info(
            "claim tokens flagged as unsupported",
            extra={"spans": [asdict(span) for span in spans]},
        )
    return ScoredClaim(
        text=claim.text, start=claim.start, end=claim.end, support=support, spans=spans
    )


def decompose(text: str, scores: TokenScores, span_threshold: float) -> list[ScoredClaim]:
    """Segment the answer, then reduce each claim off the single whole-answer pass."""
    return [reduce_claim(claim, scores, span_threshold) for claim in segment(text)]


# --- Segmentation internals ----------------------------------------------------


def _split_points(text: str) -> list[int]:
    """Sorted, unique claim-start indices (always including 0). A start marks where a
    new sentence begins; the character before it belongs to the preceding claim."""
    n = len(text)
    protected = [(m.start(), m.end()) for m in _FENCE.finditer(text)]
    starts = {0}
    for fence_start, fence_end in protected:
        starts.add(fence_start)  # a code fence begins its own claim
        after = _next_content(text, fence_end)
        if after < n:
            starts.add(after)

    i = 0
    while i < n:
        region_end = _protected_end(protected, i)
        if region_end is not None:  # skip inside code fences — no boundaries there
            i = region_end
            continue
        char = text[i]
        if char in ".!?":
            if _is_boundary_terminator(text, i):
                boundary = _next_content(text, i + 1)
                if boundary < n:
                    starts.add(boundary)
        elif char == "\n":
            boundary = _next_content(text, i)
            if boundary < n and (
                _is_list_marker(text, boundary) or _paragraph_break(text, i, boundary)
            ):
                starts.add(boundary)
        i += 1
    return sorted(starts)


def _next_content(text: str, pos: int) -> int:
    """First non-whitespace index at or after `pos` (len(text) if none remains)."""
    n = len(text)
    while pos < n and text[pos].isspace():
        pos += 1
    return pos


def _protected_end(protected: list[tuple[int, int]], i: int) -> int | None:
    for start, end in protected:
        if start <= i < end:
            return end
    return None


def _is_boundary_terminator(text: str, i: int) -> bool:
    """A `.`/`!`/`?` that ends a sentence: followed by whitespace, and — for a period —
    not an abbreviation, initial, or list-marker number."""
    if i + 1 >= len(text) or not text[i + 1].isspace():
        return False  # end of text, or mid-token (e.g. a decimal) — not a boundary
    if text[i] in "!?":
        return True
    return not _period_suppressed(text, i)


def _period_suppressed(text: str, i: int) -> bool:
    prev = text[i - 1] if i > 0 else ""
    # A single-letter initial or acronym segment: "M.I.T.", "J. Smith".
    if prev.isalpha() and (i - 2 < 0 or text[i - 2] == "." or text[i - 2].isspace()):
        return True
    # A known abbreviation token ending at this period.
    j = i - 1
    while j >= 0 and (text[j].isalnum() or text[j] == "."):
        j -= 1
    if text[j + 1 : i].lower().strip(".") in _ABBREVIATIONS:
        return True
    # A list-marker number at the start of its line: "1.", "12.".
    if prev.isdigit():
        k = i - 1
        while k >= 0 and text[k].isdigit():
            k -= 1
        while k >= 0 and text[k] in " \t":
            k -= 1
        if k < 0 or text[k] == "\n":
            return True
    return False


def _is_list_marker(text: str, pos: int) -> bool:
    return _LIST_MARKER.match(text, pos) is not None


def _paragraph_break(text: str, start: int, end: int) -> bool:
    return text[start:end].count("\n") >= 2


def _validate_partition(text: str, claims: list[Claim]) -> None:
    """Fail loud unless the claims tile `text` exactly and each honors the invariant."""
    if not claims:
        raise DecompositionError("segmentation produced no claims")
    cursor = 0
    for claim in claims:
        if claim.start != cursor:
            raise DecompositionError(f"segmentation gap or overlap at offset {cursor}")
        if text[claim.start : claim.end] != claim.text:
            raise DecompositionError(
                f"claim text does not match its offsets at [{claim.start}, {claim.end})"
            )
        cursor = claim.end
    if cursor != len(text):
        raise DecompositionError(f"segmentation left {len(text) - cursor} trailing characters")


def _merge_spans(
    claim_text: str, tokens: list[tuple[float, tuple[int, int]]], threshold: float
) -> list[Span]:
    """Merge contiguous flagged tokens into claim-relative character spans."""
    open_span: dict[str, float] | None = None
    closed: list[tuple[int, int, float]] = []

    def close() -> None:
        nonlocal open_span
        if open_span is not None:
            closed.append((int(open_span["start"]), int(open_span["end"]), open_span["conf"]))
            open_span = None

    for prob, (start, end) in tokens:
        if prob >= threshold:
            if open_span is None:
                open_span = {"start": start, "end": end, "conf": prob}
            else:
                open_span["end"] = end
                open_span["conf"] = max(open_span["conf"], prob)
        else:
            close()
    close()
    return [Span(start=s, end=e, text=claim_text[s:e], confidence=c) for s, e, c in closed]
