"""MiniCheck-Flan-T5-Large, replicated from the reference implementation
(Liyan06/MiniCheck, minicheck/inference.py) rather than installed from git:
sentence-based document chunks of <= 500 words, input
'predict: {chunk}</s>{claim}', P(support) read from the first decoder step
over the label tokens (3 = unsupported, 209 = supported).

Response-level support follows the reference "sentence fusion" rule: split
the response into sentences, take max over chunks per sentence, then min
over sentences.
"""

import torch

from bench.data import Example

NAME = "minicheck-flan-t5-large"
REPO = "lytang/MiniCheck-Flan-T5-Large"
REVISION = "96eafd01cee2d16cf81aaa2fb226b14f422a37b3"

MAX_INPUT_TOKENS = 2048
CHUNK_SIZE_WORDS = 500
BATCH_SIZE = 16
# First-decoder-step vocabulary ids the model was trained to emit.
UNSUPPORTED_TOKEN_ID = 3
SUPPORTED_TOKEN_ID = 209


def chunk_sentences(sentences: list[str], chunk_size: int) -> list[str]:
    """Greedy sentence packing into chunks of at most chunk_size words."""
    chunks: list[str] = []
    current: list[str] = []
    current_words = 0
    for sentence in sentences:
        words = len(sentence.split())
        if current and current_words + words > chunk_size:
            chunks.append(" ".join(current))
            current = [sentence]
            current_words = words
        else:
            current.append(sentence)
            current_words += words
    if current:
        chunks.append(" ".join(current))
    chunks = [chunk.replace(" \n ", "\n").strip() for chunk in chunks]
    chunks = [chunk for chunk in chunks if chunk]
    return chunks or [""]


def aggregate_support(probs_per_sentence: list[list[float]]) -> float:
    """min over response sentences of (max over document chunks)."""
    return min(max(chunk_probs) for chunk_probs in probs_per_sentence)


class MiniCheckCandidate:
    name = NAME

    def __init__(self) -> None:
        import nltk
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        nltk.download("punkt_tab", quiet=True)
        self._sent_tokenize = nltk.sent_tokenize
        self._tokenizer = AutoTokenizer.from_pretrained(REPO, revision=REVISION)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(REPO, revision=REVISION)
        self._model.eval()

    def _split_document(self, document: str) -> list[str]:
        sentences: list[str] = []
        for block in document.split("\n"):
            sentences.extend(self._sent_tokenize(block))
            sentences.append("\n")
        return chunk_sentences(sentences[:-1], CHUNK_SIZE_WORDS)

    def _support_probs(self, pairs: list[tuple[str, str]]) -> list[float]:
        probs: list[float] = []
        for start in range(0, len(pairs), BATCH_SIZE):
            batch = pairs[start : start + BATCH_SIZE]
            texts = [f"predict: {doc}{self._tokenizer.eos_token}{claim}" for doc, claim in batch]
            inputs = self._tokenizer(
                texts,
                max_length=MAX_INPUT_TOKENS,
                truncation=True,
                padding=True,
                return_tensors="pt",
            )
            decoder_input_ids = torch.zeros((inputs["input_ids"].size(0), 1), dtype=torch.long)
            with torch.no_grad():
                outputs = self._model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    decoder_input_ids=decoder_input_ids,
                )
            label_logits = outputs.logits.squeeze(1)[
                :, torch.tensor([UNSUPPORTED_TOKEN_ID, SUPPORTED_TOKEN_ID])
            ]
            label_probs = torch.softmax(label_logits, dim=-1)
            probs.extend(label_probs[:, 1].tolist())
        return probs

    def support_score(self, example: Example) -> float:
        chunks = self._split_document(example.context)
        sentences = self._sent_tokenize(example.response) or [example.response]
        pairs = [(chunk, sentence) for sentence in sentences for chunk in chunks]
        flat = self._support_probs(pairs)
        per_sentence = [
            flat[i * len(chunks) : (i + 1) * len(chunks)] for i in range(len(sentences))
        ]
        return aggregate_support(per_sentence)


def load() -> MiniCheckCandidate:
    return MiniCheckCandidate()
