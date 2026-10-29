from clausedraft.infer.engine import Drafter


class _Tokenizer:
    eos_token_id = 0

    def apply_chat_template(self, messages, add_generation_prompt=True, return_tensors="pt", return_dict=True):
        del messages, add_generation_prompt, return_tensors, return_dict
        import torch

        return {"input_ids": torch.tensor([[1]])}

    def decode(self, token_ids, skip_special_tokens=True):
        del token_ids, skip_special_tokens
        return "sent in writing."


class _Model:
    def parameters(self):
        import torch

        yield torch.nn.Parameter(torch.zeros(1))

    def generate(self, **kwargs):
        import torch

        return torch.cat([kwargs["input_ids"], torch.tensor([[2]])], dim=1)


def test_adapter_label_stays_relative_to_the_repo():
    from clausedraft.config import REPO_ROOT
    from clausedraft.infer.engine import _adapter_label

    assert _adapter_label(REPO_ROOT / "outputs" / "smoke") == "outputs/smoke"
    assert _adapter_label(None) is None


def test_drafter_completes_with_an_injected_model():
    drafter = Drafter(_Model(), _Tokenizer(), "outputs/smoke", cuda=False)
    result = drafter.complete("waivers", "No waiver shall be", preset="greedy")
    status = drafter.health()
    assert result.clause_type == "Waivers"
    assert result.continuation == "sent in writing."
    assert status.ok is True
    assert status.cuda is False
    assert status.adapter == "outputs/smoke"
