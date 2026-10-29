"""Hold a loaded model and complete one clause at a time."""

from __future__ import annotations

from pathlib import Path

from clausedraft.config import REPO_ROOT, AppConfig
from clausedraft.eval.generate import load_model
from clausedraft.infer.complete import DraftResult, complete_text
from clausedraft.infer.health import Health, health_report
from clausedraft.infer.presets import get_preset
from clausedraft.infer.request import make_request


class Drafter:
    def __init__(self, model, tokenizer, adapter: str | None, cuda: bool):
        self.model = model
        self.tokenizer = tokenizer
        self.adapter = adapter
        self.cuda = cuda

    @classmethod
    def load(cls, config: AppConfig, adapter_dir: Path | None, model_name: str | None = None) -> Drafter:
        import torch

        name = model_name or config.model.name
        model, tokenizer = load_model(name, config.quantization, adapter_dir)
        return cls(model, tokenizer, _adapter_label(adapter_dir), torch.cuda.is_available())

    def complete(self, clause_type: str, prefix: str, preset: str = "greedy") -> DraftResult:
        request = make_request(clause_type, prefix)
        return complete_text(self.model, self.tokenizer, request, get_preset(preset))

    def health(self) -> Health:
        return health_report(model_loaded=self.model is not None, adapter=self.adapter, cuda=self.cuda)


def _adapter_label(adapter_dir: Path | None) -> str | None:
    if adapter_dir is None:
        return None
    try:
        return adapter_dir.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return adapter_dir.as_posix()
