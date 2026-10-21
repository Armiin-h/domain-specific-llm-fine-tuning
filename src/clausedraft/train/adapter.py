"""Save the LoRA adapter without the frozen base weights."""

from __future__ import annotations

from pathlib import Path


def save_adapter(model, output_dir: Path, tokenizer=None) -> Path:
    if not hasattr(model, "peft_config"):
        raise ValueError("refusing to save a full model; expected a LoRA adapter")
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    if tokenizer is not None:
        tokenizer.save_pretrained(output_dir)
    return output_dir
