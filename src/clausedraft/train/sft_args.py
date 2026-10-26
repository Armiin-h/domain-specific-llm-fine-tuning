"""SFTConfig for one QLoRA run."""

from __future__ import annotations

import math
from pathlib import Path

import torch
from trl import SFTConfig

from clausedraft.config import AppConfig
from clausedraft.train.quant import native_bf16

_QWEN_EOS = "<|im_end|>"


def build_sft_config(
    config: AppConfig,
    output_dir: Path,
    *,
    model_name: str,
    train_rows: int,
    max_steps: int | None = None,
    do_eval: bool = True,
) -> SFTConfig:
    if max_steps is not None and max_steps < 1:
        raise ValueError("max_steps must be a positive integer")
    if train_rows < 1:
        raise ValueError("train_rows must be a positive integer")
    train = config.train
    use_bf16, use_fp16 = training_precision(
        train.bf16,
        cuda=torch.cuda.is_available(),
        bf16_supported=native_bf16(),
    )
    weight_dtype = "bfloat16" if use_bf16 else "float16"
    return SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=train.num_train_epochs,
        max_steps=max_steps if max_steps is not None else -1,
        per_device_train_batch_size=train.per_device_train_batch_size,
        per_device_eval_batch_size=train.per_device_train_batch_size,
        gradient_accumulation_steps=train.gradient_accumulation_steps,
        learning_rate=train.learning_rate,
        warmup_steps=warmup_steps_for(
            train_rows,
            batch_size=train.per_device_train_batch_size,
            grad_accum=train.gradient_accumulation_steps,
            epochs=train.num_train_epochs,
            warmup_ratio=train.warmup_ratio,
            max_steps=max_steps,
        ),
        logging_steps=1 if max_steps is not None else train.logging_steps,
        logging_first_step=True,
        eval_strategy="steps" if do_eval else "no",
        eval_steps=train.eval_steps if do_eval else None,
        save_strategy="no" if max_steps is not None else "steps",
        save_steps=train.save_steps,
        save_total_limit=2,
        gradient_checkpointing=train.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=use_bf16,
        fp16=use_fp16,
        optim="paged_adamw_8bit",
        assistant_only_loss=train.assistant_only_loss,
        max_length=config.model.max_seq_length,
        packing=False,
        dataset_num_proc=None,
        dataloader_num_workers=0,
        report_to="none",
        seed=config.data.seed,
        eos_token=_QWEN_EOS if "Qwen" in model_name else None,
        model_init_kwargs={"dtype": weight_dtype} if train.bf16 else None,
        remove_unused_columns=False,
    )


def training_precision(requested_bf16: bool, *, cuda: bool, bf16_supported: bool) -> tuple[bool, bool]:
    """Return (bf16, fp16). A T4 has no bfloat16 kernels, so the same config uses fp16 there."""
    if not requested_bf16:
        return False, False
    if cuda and not bf16_supported:
        return False, True
    return True, False


def warmup_steps_for(
    train_rows: int,
    *,
    batch_size: int,
    grad_accum: int,
    epochs: int,
    warmup_ratio: float,
    max_steps: int | None,
) -> int:
    if max_steps is not None:
        total = max_steps
    else:
        steps_per_epoch = max(1, math.ceil(train_rows / (batch_size * grad_accum)))
        total = steps_per_epoch * epochs
    if total <= 1:
        return 0
    return max(1, int(total * warmup_ratio))
