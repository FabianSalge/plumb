"""LettuceDetect v1/v2 token classifiers, run through the maintained
lettucedetect package so prompt formatting matches how the models were trained.

Response-level support = 1 - max token hallucination probability, which is the
package's own example-level rule (a response is flagged iff any token crosses
0.5) expressed as a continuous score.
"""

from bench.data import Example

V1_LARGE = {
    "name": "lettucedetect-large-v1",
    "repo": "KRLabsOrg/lettucedect-large-modernbert-en-v1",
    "revision": "22296c700ef0ba4ab3e5c9afffa0185caaf61e52",
}
V2_MMBERT_BASE = {
    "name": "lettucedetect-v2-mmbert-base",
    "repo": "KRLabsOrg/lettucedect-v2-mmbert-base",
    "revision": "0f85c7a15b17aee6e8f794dae7cb4e42e2b8fdac",
}


class LettuceDetectCandidate:
    def __init__(self, name: str, repo: str, revision: str) -> None:
        from lettucedetect.models.inference import HallucinationDetector

        self.name = name
        self.repo = repo
        self.revision = revision
        self._detector = HallucinationDetector(
            method="transformer", model_path=repo, revision=revision
        )

    def support_score(self, example: Example) -> float:
        # QA and Data2txt carry a real instruction; RAGTruth summarisation
        # prompts are the bare passage, which the package models as question=None.
        question = example.query if example.task_type != "Summary" else None
        tokens = self._detector.predict(
            context=[example.context],
            question=question,
            answer=example.response,
            output_format="tokens",
        )
        risk = max((token["prob"] for token in tokens), default=0.0)
        return 1.0 - float(risk)


def load_v1_large() -> LettuceDetectCandidate:
    return LettuceDetectCandidate(**V1_LARGE)


def load_v2_mmbert_base() -> LettuceDetectCandidate:
    return LettuceDetectCandidate(**V2_MMBERT_BASE)
