"""LoRA settings passed to PEFT."""

from __future__ import annotations

from peft import LoraConfig as PeftLoraConfig

from clausedraft.config import LoraConfig


def peft_lora_config(lora: LoraConfig) -> PeftLoraConfig:
    if lora.task_type != "CAUSAL_LM":
        raise ValueError("LoRA task_type must be CAUSAL_LM")
    return PeftLoraConfig(
        r=lora.r,
        lora_alpha=lora.lora_alpha,
        lora_dropout=lora.lora_dropout,
        bias=lora.bias,
        task_type=lora.task_type,
        target_modules=list(lora.target_modules),
    )
