from pathlib import Path

import pytest
import yaml

from clausedraft.config import load_config

ROOT = Path(__file__).resolve().parents[1]


def test_default_config_matches_8gb_qlora_setup():
    cfg = load_config(ROOT / "configs" / "default.yaml")

    assert cfg.model.name == "Qwen/Qwen2.5-3B-Instruct"
    assert cfg.model.fallback_name == "Qwen/Qwen2.5-1.5B-Instruct"
    assert cfg.model.max_seq_length == 1024
    assert cfg.quantization.load_in_4bit is True
    assert cfg.quantization.bnb_4bit_quant_type == "nf4"
    assert cfg.lora.r == 16
    assert cfg.lora.lora_alpha == 32
    assert "q_proj" in cfg.lora.target_modules
    assert "down_proj" in cfg.lora.target_modules
    assert cfg.data.dataset == "coastalcph/lex_glue"
    assert cfg.data.subset == "ledgar"
    assert cfg.data.train_sample_cap == 12000
    assert cfg.train.per_device_train_batch_size == 1
    assert cfg.train.eval_steps == 50
    assert cfg.train.eval_sample_cap == 128
    assert cfg.train.assistant_only_loss is True
    assert cfg.paths.test_jsonl == "data/processed/test.jsonl"


def test_rejects_prefix_fraction_outside_range(tmp_path: Path):
    raw = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text(encoding="utf-8"))
    raw["data"]["prefix_fraction"] = 0.95
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="prefix_fraction"):
        load_config(path)


def test_rejects_min_chars_above_max(tmp_path: Path):
    raw = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text(encoding="utf-8"))
    raw["data"]["min_chars"] = 5000
    raw["data"]["max_chars"] = 100
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="min_chars"):
        load_config(path)
