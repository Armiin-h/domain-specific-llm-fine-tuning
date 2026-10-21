"""4-bit load settings for QLoRA."""

from __future__ import annotations

import torch
from transformers import BitsAndBytesConfig

from clausedraft.config import QuantizationConfig

_COMPUTE_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def bitsandbytes_config(quant: QuantizationConfig) -> BitsAndBytesConfig:
    if not quant.load_in_4bit:
        raise ValueError("QLoRA training expects load_in_4bit")
    compute_dtype = _COMPUTE_DTYPES.get(quant.bnb_4bit_compute_dtype)
    if compute_dtype is None:
        known = ", ".join(sorted(_COMPUTE_DTYPES))
        raise ValueError(f"bnb_4bit_compute_dtype must be one of {known}")
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=quant.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=quant.bnb_4bit_use_double_quant,
        bnb_4bit_compute_dtype=compute_dtype,
    )
