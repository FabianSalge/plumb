"""The groundedness signal: LettuceDetect's token-classification protocol.

The prompt format and pair-tokenization layout are vendored from the
lettucedetect package (0.2.1) the pinned model ships with — the engine must not
depend on the package itself (ADR-0006). When the model revision is bumped,
re-verify this protocol against the package version that trained it.
"""

import logging
from typing import Any

from engine.config import SignalModelConfig
from engine.signals import ScorerError, TokenClassifier, TokenScores

logger = logging.getLogger("plumb.engine.signals.groundedness")

# The scoring protocol's identity, bound into every calibration artifact
# (ADR-0008). Bump it whenever inference behaviour changes — prompt template,
# passage rendering, tokenization layout — so a calibrator fitted to the old
# behaviour refuses to serve; the golden prompt test changing in the same diff
# is the reviewer's signal.
INFERENCE_MODE = "joint-questionless-v1"

# lettucedetect's English summary template (the question-less input shape),
# verbatim from prompts/summary_prompt_en.txt. Any drift silently degrades
# scores instead of erroring, so a test pins the rendered output byte-for-byte.
_PROMPT_TEMPLATE = "Summarize the following text:\n{context}\noutput:"

# lettucedetect's training-time sequence window; not a tunable — inputs beyond
# it were never seen by the model, so raising it doesn't buy longer context.
_MAX_LENGTH = 4096


def render_prompt(passages: list[str]) -> str:
    context = "\n".join(f"passage {i + 1}: {passage}" for i, passage in enumerate(passages))
    return _PROMPT_TEMPLATE.format(context=context)


class LettuceDetectScorer:
    def __init__(self, pipeline: TokenClassifier) -> None:
        self._pipeline = pipeline

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
        return cls(_TransformersTokenClassifier(model, tokenizer))  # pragma: no cover

    def score(self, text: str, passages: list[str]) -> TokenScores:
        """One whole-answer joint pass: per-token risk with answer-relative offsets.
        Reduction to per-claim support and spans is the decomposition step's work."""
        if not passages:
            raise ScorerError("cannot score against zero evidence passages")
        result = self._pipeline.token_probs(render_prompt(passages), text)
        if not result.probs:
            raise ScorerError("model returned no token probabilities for the answer")
        if result.truncated:
            logger.warning(
                "evidence context truncated to fit the model window",
                extra={"passage_count": len(passages)},
            )
        return result

    def count_tokens(self, text: str) -> int:
        return self._pipeline.count_tokens(text)


class _TransformersTokenClassifier:  # pragma: no cover — exercised by `pytest -m model`
    """Torch-backed TokenClassifier reproducing lettucedetect's tokenization:
    (context, claim) as a sentence pair, `only_first` truncation so the claim
    is never cut, claim tokens located by counting from the trailing separator."""

    def __init__(self, model: Any, tokenizer: Any) -> None:
        self._model = model
        self._tokenizer = tokenizer

    def count_tokens(self, text: str) -> int:
        return int(
            self._tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"].shape[
                1
            ]
        )

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
