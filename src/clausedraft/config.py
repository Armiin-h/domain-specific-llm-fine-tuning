"""Load and validate the training config."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "default.yaml"


@dataclass(frozen=True)
class ModelConfig:
    name: str
    fallback_name: str
    max_seq_length: int


@dataclass(frozen=True)
class QuantizationConfig:
    load_in_4bit: bool
    bnb_4bit_quant_type: str
    bnb_4bit_use_double_quant: bool
    bnb_4bit_compute_dtype: str


@dataclass(frozen=True)
class LoraConfig:
    r: int
    lora_alpha: int
    lora_dropout: float
    bias: str
    task_type: str
    target_modules: tuple[str, ...]


@dataclass(frozen=True)
class DataConfig:
    dataset: str
    subset: str
    train_sample_cap: int
    min_chars: int
    max_chars: int
    prefix_fraction: float
    seed: int


@dataclass(frozen=True)
class TrainConfigSection:
    num_train_epochs: int
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    warmup_ratio: float
    logging_steps: int
    eval_steps: int
    save_steps: int
    gradient_checkpointing: bool
    bf16: bool
    assistant_only_loss: bool
    output_dir: str


@dataclass(frozen=True)
class PathConfig:
    train_jsonl: str
    val_jsonl: str
    test_jsonl: str


@dataclass(frozen=True)
class AppConfig:
    model: ModelConfig
    quantization: QuantizationConfig
    lora: LoraConfig
    data: DataConfig
    train: TrainConfigSection
    paths: PathConfig


def load_config(path: Path | None = None) -> AppConfig:
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Config root must be a mapping: {config_path}")
    return _parse(raw)


def _parse(raw: dict) -> AppConfig:
    model = _section(raw, "model")
    quant = _section(raw, "quantization")
    lora = _section(raw, "lora")
    data = _section(raw, "data")
    train = _section(raw, "train")
    paths = _section(raw, "paths")

    max_seq_length = _positive_int(model, "max_seq_length")
    lora_r = _positive_int(lora, "r")
    min_chars = _positive_int(data, "min_chars")
    max_chars = _positive_int(data, "max_chars")
    if min_chars >= max_chars:
        raise ValueError("data.min_chars must be smaller than data.max_chars")

    prefix_fraction = float(data["prefix_fraction"])
    if not 0.05 <= prefix_fraction <= 0.8:
        raise ValueError("data.prefix_fraction must be between 0.05 and 0.8")

    name = _nonempty(model, "name")
    fallback_name = _nonempty(model, "fallback_name")
    if name == fallback_name:
        raise ValueError("model.fallback_name must differ from model.name")

    modules = lora.get("target_modules")
    if not isinstance(modules, list) or not modules or not all(isinstance(m, str) and m for m in modules):
        raise ValueError("lora.target_modules must be a non-empty list of strings")

    return AppConfig(
        model=ModelConfig(
            name=name,
            fallback_name=fallback_name,
            max_seq_length=max_seq_length,
        ),
        quantization=QuantizationConfig(
            load_in_4bit=bool(quant["load_in_4bit"]),
            bnb_4bit_quant_type=_nonempty(quant, "bnb_4bit_quant_type"),
            bnb_4bit_use_double_quant=bool(quant["bnb_4bit_use_double_quant"]),
            bnb_4bit_compute_dtype=_nonempty(quant, "bnb_4bit_compute_dtype"),
        ),
        lora=LoraConfig(
            r=lora_r,
            lora_alpha=_positive_int(lora, "lora_alpha"),
            lora_dropout=float(lora["lora_dropout"]),
            bias=_nonempty(lora, "bias"),
            task_type=_nonempty(lora, "task_type"),
            target_modules=tuple(modules),
        ),
        data=DataConfig(
            dataset=_nonempty(data, "dataset"),
            subset=_nonempty(data, "subset"),
            train_sample_cap=_positive_int(data, "train_sample_cap"),
            min_chars=min_chars,
            max_chars=max_chars,
            prefix_fraction=prefix_fraction,
            seed=int(data["seed"]),
        ),
        train=TrainConfigSection(
            num_train_epochs=_positive_int(train, "num_train_epochs"),
            per_device_train_batch_size=_positive_int(train, "per_device_train_batch_size"),
            gradient_accumulation_steps=_positive_int(train, "gradient_accumulation_steps"),
            learning_rate=float(train["learning_rate"]),
            warmup_ratio=float(train["warmup_ratio"]),
            logging_steps=_positive_int(train, "logging_steps"),
            eval_steps=_positive_int(train, "eval_steps"),
            save_steps=_positive_int(train, "save_steps"),
            gradient_checkpointing=bool(train["gradient_checkpointing"]),
            bf16=bool(train["bf16"]),
            assistant_only_loss=bool(train["assistant_only_loss"]),
            output_dir=_nonempty(train, "output_dir"),
        ),
        paths=PathConfig(
            train_jsonl=_nonempty(paths, "train_jsonl"),
            val_jsonl=_nonempty(paths, "val_jsonl"),
            test_jsonl=_nonempty(paths, "test_jsonl"),
        ),
    )


def _section(raw: dict, key: str) -> dict:
    section = raw.get(key)
    if not isinstance(section, dict):
        raise ValueError(f"Missing config section: {key}")
    return section


def _nonempty(section: dict, key: str) -> str:
    value = section.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _positive_int(section: dict, key: str) -> int:
    value = section.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{key} must be a positive integer")
    return value
