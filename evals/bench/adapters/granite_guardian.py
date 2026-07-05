"""Granite Guardian 3.2 3B-A800M, the smallest Granite Guardian with the
groundedness risk — included to price the LLM-judge shape on CPU, per its
model card protocol: guardian chat template, first Yes/No token decides,
token probability used as the continuous score.
"""

import torch

from bench.data import Example

NAME = "granite-guardian-3.2-3b-a800m"
REPO = "ibm-granite/granite-guardian-3.2-3b-a800m"
REVISION = "3de033d89b499a18d9a573b5192bf3b967ef48c5"


class GraniteGuardianError(Exception):
    """The guardian template or output did not behave as its model card documents."""


class GraniteGuardianCandidate:
    name = NAME

    def __init__(self) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(REPO, revision=REVISION)
        self._model = AutoModelForCausalLM.from_pretrained(
            REPO, revision=REVISION, dtype=torch.bfloat16
        )
        self._model.eval()
        vocab = self._tokenizer.get_vocab()
        self._yes_ids = {vocab[t] for t in ("Yes", "ĠYes", "yes", "Ġyes") if t in vocab}
        self._no_ids = {vocab[t] for t in ("No", "ĠNo", "no", "Ġno") if t in vocab}
        if not self._yes_ids or not self._no_ids:
            raise GraniteGuardianError("could not resolve Yes/No token ids in the vocabulary")

    def support_score(self, example: Example) -> float:
        messages = [
            {"role": "context", "content": example.context},
            {"role": "assistant", "content": example.response},
        ]
        input_ids = self._tokenizer.apply_chat_template(
            messages,
            guardian_config={"risk_name": "groundedness"},
            add_generation_prompt=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            generated = self._model.generate(
                input_ids,
                max_new_tokens=4,
                do_sample=False,
                output_scores=True,
                return_dict_in_generate=True,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        for step_scores in generated.scores:
            probs = torch.softmax(step_scores[0].float(), dim=-1)
            p_yes = float(sum(probs[i] for i in self._yes_ids))
            p_no = float(sum(probs[i] for i in self._no_ids))
            if p_yes + p_no > 0.5:
                # "Yes" answers "is there groundedness risk" — support is P(No).
                return p_no / (p_yes + p_no)
        text = self._tokenizer.decode(generated.sequences[0, input_ids.shape[1] :])
        raise GraniteGuardianError(f"no Yes/No token in guardian output: {text!r}")


def load() -> GraniteGuardianCandidate:
    return GraniteGuardianCandidate()
