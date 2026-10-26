"""Short model card for one finished QLoRA run."""

from __future__ import annotations

from clausedraft.config import AppConfig


def render_model_card(
    config: AppConfig,
    *,
    base_model: str,
    train_sha256: str,
    train_rows: int,
    eval_rows: int,
    train_loss: float | None,
    eval_loss: float | None,
    train_runtime: float | None,
    row_limit: int | None = None,
    max_steps: int | None = None,
) -> str:
    train = config.train
    lora = config.lora
    quant = config.quantization
    used = f"{train_rows}"
    if row_limit is not None:
        used = f"{train_rows} (first {row_limit} rows of the train file)"
    stop = "one epoch"
    if max_steps == 1:
        stop = "capped at 1 optimizer step"
    elif max_steps is not None:
        stop = f"capped at {max_steps} optimizer steps"

    if eval_rows < 1:
        scored = "none"
    else:
        scored = (
            f"{eval_rows} (first {train.eval_sample_cap} rows of the validation file)"
        )
    lines = [
        "# ClauseDraft LoRA adapter",
        "",
        "This adapter continues a partial commercial-contract clause in the register of the LEDGAR provisions it was trained on. The output is a drafting demo, not legal advice.",
        "",
        "## Run",
        "",
        f"- Base model: `{base_model}`",
        f"- Train file sha256: `{train_sha256}`",
        f"- Train rows: {used}",
        f"- Validation rows scored: {scored}",
        f"- Stop: {stop}",
        "",
        "## Hyperparameters",
        "",
        f"- LoRA rank {lora.r}, alpha {lora.lora_alpha}, dropout {lora.lora_dropout}, bias {lora.bias}",
        f"- Target modules: {', '.join(lora.target_modules)}",
        f"- Quantization: {quant.bnb_4bit_quant_type}, double quant {quant.bnb_4bit_use_double_quant}, compute {quant.bnb_4bit_compute_dtype}",
        f"- Sequence length {config.model.max_seq_length}, batch {train.per_device_train_batch_size}, gradient accumulation {train.gradient_accumulation_steps}",
        f"- Learning rate {train.learning_rate}, warmup ratio {train.warmup_ratio}, epochs {train.num_train_epochs}",
        f"- Assistant-only loss: {train.assistant_only_loss}",
        f"- Seed: {config.data.seed}",
        "",
        "## Result",
        "",
    ]
    if train_loss is None:
        lines.append("- Train loss: not recorded")
    else:
        lines.append(f"- Train loss: {train_loss:.4f}")
    if eval_loss is None:
        lines.append("- Eval loss: not recorded")
    else:
        lines.append(f"- Eval loss: {eval_loss:.4f}")
    if train_runtime is not None:
        lines.append(f"- Train runtime (seconds): {train_runtime:.1f}")
    lines.extend(
        [
            "",
            "## Data",
            "",
            "LEDGAR provisions from LexGLUE (`coastalcph/lex_glue`, config `ledgar`). The official test split is not part of this run.",
            "",
            "LEDGAR: Tuggener, von Däniken, Peetz, and Cieliebak, 2020. CC BY-NC 4.0. LexGLUE: Chalkidis et al., 2022. The underlying filing text is public SEC material.",
            "",
        ]
    )
    return "\n".join(lines)
