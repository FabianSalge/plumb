"""Scoring wrapper speaking LettuceDetect's token-classification protocol.

The prompt format and pair-tokenization layout are vendored from the
lettucedetect package (0.2.1) the pinned model ships with — the engine must not
depend on the package itself (ADR-0006). When the model revision is bumped,
re-verify this protocol against the package version that trained it.
"""

import logging
from dataclasses import asdict, dataclass
from typing import Any, Protocol

from engine.config import SignalModelConfig

logger = logging.getLogger("plumb.engine.scoring")

# lettucedetect's English summary template (the question-less input shape),
# verbatim from prompts/summary_prompt_en.txt. Any drift silently degrades
# scores instead of erroring, so a test pins the rendered output byte-for-byte.
_PROMPT_TEMPLATE = "Summarize the following text:\n{context}\noutput:"

# lettucedetect's training-time sequence window; not a tunable — inputs beyond
# it were never seen by the model, so raising it doesn't buy longer context.
_MAX_LENGTH = 4096


class ScorerError(Exception):
    """The scoring model is unavailable or returned something it must not."""


@dataclass(frozen=True)
class TokenScores:
    """Per-token hallucination probabilities for the claim, aligned with
    character offsets into the claim (zero-length offsets are special tokens);
    `truncated` marks a context that was cut to fit the model window."""

    probs: list[float]
    offsets: list[tuple[int, int]]
    truncated: bool = False


class TokenClassifier(Protocol):
    def token_probs(self, prompt: str, claim: str) -> TokenScores: ...


@dataclass(frozen=True)
class Span:
    """An unsupported region of the claim; start/end are Unicode code-point
    offsets into the claim text. `confidence` is the raw maximum token
    probability — structured-log detail only until calibration (#32)."""

    start: int
    end: int
    text: str
    confidence: float


@dataclass(frozen=True)
class ClaimScore:
    """One claim's support by the union of all passages, with the spans the
    model flagged as unsupported."""

    support: float
    spans: list[Span]


class Scorer(Protocol):
    def score(self, claim: str, passages: list[str]) -> ClaimScore: ...


def render_prompt(passages: list[str]) -> str:
    context = "\n".join(f"passage {i + 1}: {passage}" for i, passage in enumerate(passages))
    return _PROMPT_TEMPLATE.format(context=context)


def spans_from_token_scores(claim: str, scores: TokenScores, threshold: float) -> list[Span]:
    """Merge contiguous flagged tokens into character spans over the claim."""
    open_span: dict[str, float] | None = None
    closed: list[tuple[int, int, float]] = []

    def close() -> None:
        nonlocal open_span
        if open_span is not None:
            closed.append((int(open_span["start"]), int(open_span["end"]), open_span["conf"]))
            open_span = None

    for prob, (start, end) in zip(scores.probs, scores.offsets, strict=True):
        if start == end:  # special token — scores, but has no claim characters
            continue
        if prob >= threshold:
            if open_span is None:
                open_span = {"start": start, "end": end, "conf": prob}
            else:
                open_span["end"] = end
                open_span["conf"] = max(open_span["conf"], prob)
        else:
            close()
    close()
    return [Span(start=s, end=e, text=claim[s:e], confidence=c) for s, e, c in closed]


class LettuceDetectScorer:
    def __init__(self, pipeline: TokenClassifier, span_threshold: float) -> None:
        self._pipeline = pipeline
        self._span_threshold = span_threshold

    @classmethod
    def load(cls, cfg: SignalModelConfig) -> "LettuceDetectScorer":
        try:
            import torch  # noqa: F401
            from transformers import AutoModelForTokenClassification, AutoTokenizer
        except ImportError as exc:
            raise ScorerError(
                "transformers/torch are not installed — "
                "install the 'model' extra to load the scoring model"
            ) from exc
        # pragma-free import guard above is unit-tested; the download below is
        # exercised by `pytest -m model` only.
        tokenizer = AutoTokenizer.from_pretrained(  # pragma: no cover
            cfg.model, revision=cfg.revision
        )
        model = AutoModelForTokenClassification.from_pretrained(  # pragma: no cover
            cfg.model, revision=cfg.revision
        )
        model.eval()  # pragma: no cover
        return cls(  # pragma: no cover
            _TransformersTokenClassifier(model, tokenizer), span_threshold=cfg.span_threshold
        )

    def score(self, claim: str, passages: list[str]) -> ClaimScore:
        if not passages:
            raise ScorerError("cannot score a claim against zero evidence passages")
        result = self._pipeline.token_probs(render_prompt(passages), claim)
        if not result.probs:
            raise ScorerError("model returned no token probabilities for the claim")
        if result.truncated:
            logger.warning(
                "evidence context truncated to fit the model window",
                extra={"passage_count": len(passages)},
            )
        support = 1.0 - max(result.probs)
        if not 0.0 <= support <= 1.0:
            raise ScorerError(f"model produced support score {support} outside [0, 1]")
        spans = spans_from_token_scores(claim, result, self._span_threshold)
        if spans:
            logger.info(
                "claim tokens flagged as unsupported",
                extra={"spans": [asdict(span) for span in spans]},
            )
        return ClaimScore(support=support, spans=spans)


class _TransformersTokenClassifier:  # pragma: no cover — exercised by `pytest -m model`
    """Torch-backed TokenClassifier reproducing lettucedetect's tokenization:
    (context, claim) as a sentence pair, `only_first` truncation so the claim
    is never cut, claim tokens located by counting from the trailing separator."""

    def __init__(self, model: Any, tokenizer: Any) -> None:
        self._model = model
        self._tokenizer = tokenizer

    def token_probs(self, prompt: str, claim: str) -> TokenScores:
        import torch

        encoding = self._tokenizer(
            prompt,
            claim,
            truncation="only_first",
            max_length=_MAX_LENGTH,
            return_offsets_mapping=True,
            return_tensors="pt",
            add_special_tokens=True,
        )
        offsets = encoding.pop("offset_mapping")[0]
        claim_token_count = self._tokenizer(claim, add_special_tokens=False, return_tensors="pt")[
            "input_ids"
        ].shape[1]
        prompt_token_count = self._tokenizer(prompt, add_special_tokens=False, return_tensors="pt")[
            "input_ids"
        ].shape[1]
        total = encoding["input_ids"].shape[1]
        # Layout: [CLS] context [SEP] claim [SEP]; the trailing separator is
        # part of the scored region, exactly as lettucedetect labels it.
        answer_start = total - claim_token_count - 1
        truncated = prompt_token_count + claim_token_count + 3 > _MAX_LENGTH

        with torch.no_grad():
            logits = self._model(
                input_ids=encoding["input_ids"], attention_mask=encoding["attention_mask"]
            ).logits
        probs = torch.softmax(logits, dim=-1)[0, :, 1]

        claim_char_offset = int(offsets[answer_start][0].item())
        token_probs: list[float] = []
        token_offsets: list[tuple[int, int]] = []
        for i in range(answer_start, total):
            start, end = (int(v) for v in offsets[i].tolist())
            if start == end:
                token_offsets.append((0, 0))
            else:
                token_offsets.append((start - claim_char_offset, end - claim_char_offset))
            token_probs.append(float(probs[i].item()))
        return TokenScores(probs=token_probs, offsets=token_offsets, truncated=truncated)
