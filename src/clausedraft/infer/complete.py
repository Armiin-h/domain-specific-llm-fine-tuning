"""Continue a clause with an already loaded model."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from clausedraft.infer.presets import DecodingPreset
from clausedraft.infer.request import DraftRequest, request_messages


@dataclass(frozen=True)
class DraftResult:
    clause_type: str
    prefix: str
    continuation: str
    preset: str


def complete_text(model, tokenizer, request: DraftRequest, preset: DecodingPreset) -> DraftResult:
    encoded = tokenizer.apply_chat_template(
        request_messages(request),
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    encoded = {key: value.to(_device(model)) for key, value in encoded.items() if torch.is_tensor(value)}
    prompt_len = encoded["input_ids"].shape[1]
    kwargs = {
        "max_new_tokens": preset.max_new_tokens,
        "do_sample": preset.do_sample,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if preset.do_sample:
        kwargs["temperature"] = preset.temperature
        kwargs["top_p"] = preset.top_p
    with torch.inference_mode():
        output = model.generate(**encoded, **kwargs)
    new_tokens = output[0, prompt_len:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return DraftResult(
        clause_type=request.clause_type,
        prefix=request.prefix,
        continuation=text,
        preset=preset.name,
    )


def _device(model) -> torch.device:
    return next(model.parameters()).device
