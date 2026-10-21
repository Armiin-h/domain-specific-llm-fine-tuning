from pathlib import Path

import pytest

from clausedraft.train.adapter import save_adapter


class _Adapter:
    def __init__(self) -> None:
        self.peft_config = {"default": object()}
        self.saved_to: Path | None = None

    def save_pretrained(self, path: str) -> None:
        self.saved_to = Path(path)
        target = Path(path)
        target.mkdir(parents=True, exist_ok=True)
        (target / "adapter_config.json").write_text("{}", encoding="utf-8")


class _Tokenizer:
    def save_pretrained(self, path: str) -> None:
        (Path(path) / "tokenizer_config.json").write_text("{}", encoding="utf-8")


def test_save_adapter_writes_adapter_files_only(tmp_path):
    model = _Adapter()
    saved = save_adapter(model, tmp_path / "adapter", tokenizer=_Tokenizer())
    assert model.saved_to == saved
    assert (saved / "adapter_config.json").is_file()
    assert (saved / "tokenizer_config.json").is_file()
    assert not (saved / "model.safetensors").exists()


def test_save_adapter_rejects_a_plain_model(tmp_path):
    with pytest.raises(ValueError, match="LoRA adapter"):
        save_adapter(object(), tmp_path / "full")
