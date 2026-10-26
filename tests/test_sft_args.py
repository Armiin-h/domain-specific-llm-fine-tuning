from pathlib import Path

import pytest

from clausedraft.config import load_config
from clausedraft.train.sft_args import build_sft_config, training_precision


def test_sft_config_trains_the_assistant_tokens_only():
    config = load_config()
    args = build_sft_config(
        config,
        Path("outputs/adapter"),
        model_name=config.model.name,
        train_rows=12000,
    )
    assert args.assistant_only_loss is True
    assert args.max_length == 1024
    assert args.learning_rate == pytest.approx(0.0002)
    assert args.per_device_train_batch_size == 1
    assert args.gradient_accumulation_steps == 8
    assert args.bf16 is True
    assert args.optim == "paged_adamw_8bit"
    assert args.eos_token == "<|im_end|>"
    assert args.seed == 42
    assert args.warmup_steps == 45


def test_t4_trains_in_fp16():
    assert training_precision(True, cuda=True, bf16_supported=False) == (False, True)
    assert training_precision(True, cuda=True, bf16_supported=True) == (True, False)
    assert training_precision(True, cuda=False, bf16_supported=False) == (True, False)


def test_sft_config_short_run_skips_mid_training_checkpoints():
    config = load_config()
    args = build_sft_config(
        config,
        Path("outputs/smoke"),
        model_name=config.model.name,
        train_rows=32,
        max_steps=1,
        do_eval=False,
    )
    assert args.max_steps == 1
    assert args.eval_strategy == "no"
    assert args.save_strategy == "no"
    assert args.logging_steps == 1
    assert args.warmup_steps == 0
