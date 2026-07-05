"""HHEM-2.1-open — the do-nothing baseline, pinned exactly as the engine pins it."""

from bench.data import Example

NAME = "hhem-2.1-open"
REPO = "vectara/hallucination_evaluation_model"
# Same pin as config/verifier.yaml.
REVISION = "8e4a2e6e96c708cc76c2344f7e4757df2515292c"
# HHEM's remote code fetches its tokenizer from a second Hub repo at runtime;
# that download is part of its deployment footprint.
EXTRA_REPOS = ["google/flan-t5-base"]


class HHEMCandidate:
    name = NAME

    def __init__(self) -> None:
        from transformers import AutoModelForSequenceClassification

        self._model = AutoModelForSequenceClassification.from_pretrained(
            REPO, revision=REVISION, trust_remote_code=True
        )

    def support_score(self, example: Example) -> float:
        # HHEM's documented protocol: one (premise, hypothesis) pair per check,
        # whole response as the hypothesis.
        return float(self._model.predict([(example.context, example.response)])[0])


def load() -> HHEMCandidate:
    return HHEMCandidate()
