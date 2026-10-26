"""Build the trainer, run it, and keep the adapter plus the loss log."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from trl import SFTTrainer

from clausedraft.config import REPO_ROOT, AppConfig
from clausedraft.train.adapter import save_adapter
from clausedraft.train.card import render_model_card
from clausedraft.train.chat_data import load_chat_split
from clausedraft.train.fingerprint import file_sha256
from clausedraft.train.history import loss_rows
from clausedraft.train.lora import peft_lora_config
from clausedraft.train.quant import bitsandbytes_config
from clausedraft.train.record import write_run_record
from clausedraft.train.sft_args import build_sft_config


@dataclass(frozen=True)
class TrainResult:
    output_dir: Path
    train_rows: int
    eval_rows: int
    metrics: dict


def resolve_output_dir(config: AppConfig, override: str | None) -> Path:
    if override:
        path = Path(override)
        return path if path.is_absolute() else REPO_ROOT / path
    return REPO_ROOT / config.train.output_dir


class ClauseSFTTrainer(SFTTrainer):
    """Use the model's own loss when Triton's fused head is unavailable.

    Current TRL builds the training loss with a fused LM head. That kernel is
    imported from Triton, and the Windows PyTorch wheel does not ship Triton.
    """

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        from trl.trainer.utils import _ChunkedLogProbFunction

        if _ChunkedLogProbFunction is not None:
            return super().compute_loss(
                model,
                inputs,
                return_outputs=return_outputs,
                num_items_in_batch=num_items_in_batch,
            )
        prepared = dict(inputs)
        prepared["use_cache"] = False
        prepared["fused_lm_head"] = False
        if num_items_in_batch is not None and getattr(self, "model_accepts_loss_kwargs", False):
            prepared["num_items_in_batch"] = num_items_in_batch
        outputs = model(**prepared)
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss


def run_sft(
    config: AppConfig,
    *,
    train_path: Path,
    output_dir: Path,
    eval_path: Path | None = None,
    model_name: str | None = None,
    limit: int | None = None,
    max_steps: int | None = None,
    resume: bool = False,
) -> TrainResult:
    chosen_model = model_name or config.model.name
    train_dataset = load_chat_split(train_path, limit)
    # A capped step count is a smoke run. Skip the validation file so one step
    # does not walk the holdout split. A full run scores the configured prefix
    # of that file, not every validation row.
    eval_dataset = None
    if eval_path is not None and max_steps is None:
        eval_dataset = load_chat_split(eval_path, config.train.eval_sample_cap)
    args = build_sft_config(
        config,
        output_dir,
        model_name=chosen_model,
        train_rows=len(train_dataset),
        max_steps=max_steps,
        do_eval=eval_dataset is not None,
    )
    trainer = ClauseSFTTrainer(
        model=chosen_model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_lora_config(config.lora),
        quantization_config=bitsandbytes_config(config.quantization),
    )
    train_output = trainer.train(resume_from_checkpoint=True if resume else None)
    save_adapter(trainer.model, output_dir, tokenizer=trainer.processing_class)
    metrics = {key: _jsonable(value) for key, value in dict(train_output.metrics).items()}
    history = loss_rows(trainer.state.log_history)
    card = render_model_card(
        config,
        base_model=chosen_model,
        train_sha256=file_sha256(train_path),
        train_rows=len(train_dataset),
        eval_rows=0 if eval_dataset is None else len(eval_dataset),
        train_loss=_metric(metrics, history, "train_loss", "loss"),
        eval_loss=_metric(metrics, history, "eval_loss", "eval_loss"),
        train_runtime=_as_float(metrics.get("train_runtime")),
        row_limit=limit,
        max_steps=max_steps,
    )
    write_run_record(output_dir, history, metrics, card)
    return TrainResult(
        output_dir=output_dir,
        train_rows=len(train_dataset),
        eval_rows=0 if eval_dataset is None else len(eval_dataset),
        metrics=metrics,
    )


def _jsonable(value):
    if isinstance(value, float):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return str(value)


def _metric(metrics: dict, history: list[dict], metric_key: str, history_key: str) -> float | None:
    found = _as_float(metrics.get(metric_key))
    if found is not None:
        return found
    for row in reversed(history):
        found = _as_float(row.get(history_key))
        if found is not None:
            return found
    return None


def _as_float(value) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None
