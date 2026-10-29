import torch

from clausedraft.infer.complete import complete_text
from clausedraft.infer.presets import get_preset
from clausedraft.infer.request import make_request


class _Tokenizer:
    eos_token_id = 0

    def apply_chat_template(self, messages, add_generation_prompt=True, return_tensors="pt", return_dict=True):
        del messages, add_generation_prompt, return_tensors, return_dict
        return {"input_ids": torch.tensor([[1, 2, 3]])}

    def decode(self, token_ids, skip_special_tokens=True):
        del skip_special_tokens
        return "shall be governed by the laws of Delaware."


class _Model:
    def __init__(self):
        self.kwargs = None
        self._weight = torch.nn.Parameter(torch.zeros(1))

    def parameters(self):
        yield self._weight

    def generate(self, **kwargs):
        self.kwargs = kwargs
        extra = torch.tensor([[4, 5]])
        return torch.cat([kwargs["input_ids"], extra], dim=1)


def test_complete_text_returns_only_the_new_tokens():
    model = _Model()
    result = complete_text(model, _Tokenizer(), make_request("Governing Laws", "This Agreement shall be"), get_preset("greedy"))
    assert result.continuation == "shall be governed by the laws of Delaware."
    assert result.preset == "greedy"
    assert "temperature" not in model.kwargs


def test_draft_preset_passes_sampling_settings():
    model = _Model()
    complete_text(model, _Tokenizer(), make_request("Notices", "All notices shall be"), get_preset("draft"))
    assert model.kwargs["do_sample"] is True
    assert model.kwargs["temperature"] == 0.7
    assert model.kwargs["top_p"] == 0.9
    assert model.kwargs["max_new_tokens"] == 160
