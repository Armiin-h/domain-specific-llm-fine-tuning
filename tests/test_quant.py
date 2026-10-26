from clausedraft.config import QuantizationConfig, load_config
from clausedraft.train.quant import bitsandbytes_config, resolve_compute_dtype_name


def test_bitsandbytes_config_uses_nf4_and_bfloat16():
    config = bitsandbytes_config(load_config().quantization)
    assert config.load_in_4bit is True
    assert config.bnb_4bit_quant_type == "nf4"
    assert config.bnb_4bit_use_double_quant is True
    assert config.bnb_4bit_compute_dtype is not None


def test_t4_compute_dtype_falls_back_to_float16():
    assert resolve_compute_dtype_name("bfloat16", cuda=True, bf16_supported=False) == "float16"
    assert resolve_compute_dtype_name("bfloat16", cuda=True, bf16_supported=True) == "bfloat16"


def test_bitsandbytes_config_rejects_a_full_precision_load():
    import pytest

    quant = QuantizationConfig(
        load_in_4bit=False,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype="bfloat16",
    )
    with pytest.raises(ValueError, match="load_in_4bit"):
        bitsandbytes_config(quant)
