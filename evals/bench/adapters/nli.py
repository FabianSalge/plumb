"""Candidate NLI cross-encoders behind one generic adapter (issue #60).

Every candidate is a plain `AutoModelForSequenceClassification` three-class NLI
head loaded from a single pinned revision with no remote code — the operational
bar ADR-0006 set. Scoring is one (premise=evidence, hypothesis=sentence) pass;
when the pair exceeds the candidate's trained window, the premise truncates
(never the hypothesis) and the pair is flagged so the run can publish how much
evidence each window actually sees.
"""

from collections.abc import Mapping
from dataclasses import dataclass

CANDIDATES = {
    "modernbert-base-nli": {
        "repo": "tasksource/ModernBERT-base-nli",
        "revision": "de4ab7e77845098b7fab7f6ab9d370ddff27b19c",
        "max_length": 8192,
    },
    "modernbert-large-nli": {
        "repo": "tasksource/ModernBERT-large-nli",
        "revision": "ca476cb923a8637073d4ceb0f19f7fc236e260d4",
        "max_length": 8192,
    },
    "deberta-base-long-nli": {
        "repo": "tasksource/deberta-base-long-nli",
        "revision": "04dcf11f844b07bc57015169fca2b7d6df8299d5",
        "max_length": 1680,
    },
    "deberta-v3-base-mnli-fever-anli": {
        "repo": "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
        "revision": "6f5cf0a2b59cabb106aca4c287eed12e357e90eb",
        "max_length": 512,
    },
    "mdeberta-v3-base-mnli-xnli": {
        "repo": "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
        "revision": "8adb042d524ecd5c26d3e3ba0e3fbcf7e2d0864c",
        "max_length": 512,
    },
}


@dataclass(frozen=True)
class NliProbs:
    """Softmax over the three NLI classes, plus whether the premise truncated."""

    entailment: float
    neutral: float
    contradiction: float
    truncated: bool


def label_indices(id2label: Mapping[int, str]) -> tuple[int, int, int]:
    """(entailment, neutral, contradiction) logit indices resolved from the model's
    own head config — candidates disagree on class order, so never assume one."""
    by_name = {name.lower(): index for index, name in id2label.items()}
    if sorted(by_name) != ["contradiction", "entailment", "neutral"]:
        raise ValueError(f"not a three-class NLI head: {dict(id2label)!r}")
    return by_name["entailment"], by_name["neutral"], by_name["contradiction"]


class NliCandidate:
    def __init__(self, name: str, repo: str, revision: str, max_length: int) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.name = name
        self.repo = repo
        self.revision = revision
        self.max_length = max_length
        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(repo, revision=revision)
        self._model = AutoModelForSequenceClassification.from_pretrained(repo, revision=revision)
        self._model.eval()
        self._indices = label_indices(self._model.config.id2label)

    def probs(self, premise: str, hypothesis: str) -> NliProbs:
        untruncated = self._tokenizer(premise, hypothesis, truncation=False)
        truncated = len(untruncated["input_ids"]) > self.max_length
        inputs = self._tokenizer(
            premise,
            hypothesis,
            truncation="only_first",
            max_length=self.max_length,
            return_tensors="pt",
        )
        with self._torch.inference_mode():
            logits = self._model(**inputs).logits[0]
        softmax = self._torch.softmax(logits, dim=-1).tolist()
        entailment, neutral, contradiction = self._indices
        return NliProbs(
            entailment=softmax[entailment],
            neutral=softmax[neutral],
            contradiction=softmax[contradiction],
            truncated=truncated,
        )


def load(name: str) -> NliCandidate:
    return NliCandidate(name=name, **CANDIDATES[name])
