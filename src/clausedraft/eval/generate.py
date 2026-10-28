"""Generate continuations and score the gold tail."""

from __future__ import annotations

from pathlib import Path

import torch

from clausedraft.config import QuantizationConfig
from clausedraft.data.messages import SYSTEM_PROMPT, user_prompt
from clausedraft.eval.decode import DecodeSettings
from clausedraft.eval.prompts import PromptExample
from clausedraft.train.quant import bitsandbytes_config, native_bf16


def generation_messages(example: PromptExample) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt(example.clause_type, example.prefix)},
    ]


def prompt_labels(full_ids: list[int], prompt_ids: list[int]) -> list[int]:
    if not prompt_ids or len(prompt_ids) >= len(full_ids) or full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError("prompt tokens are not a prefix of the full conversation")
    return [-100] * len(prompt_ids) + full_ids[len(prompt_ids) :]


def load_model(model_name: str, quant: QuantizationConfig, adapter_dir: Path | None = None):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = torch.bfloat16 if native_bf16() else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bitsandbytes_config(quant),
        device_map="auto",
        dtype=dtype,
    )
    if adapter_dir is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()
    return model, tokenizer


def generate_continuation(model, tokenizer, example: PromptExample, decode: DecodeSettings) -> str:
    messages = generation_messages(example)
    encoded = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    encoded = {key: value.to(_device(model)) for key, value in encoded.items() if torch.is_tensor(value)}
    prompt_len = encoded["input_ids"].shape[1]
    with torch.inference_mode():
        output = model.generate(
            **encoded,
            max_new_tokens=decode.max_new_tokens,
            do_sample=decode.do_sample,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = output[0, prompt_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def gold_nll(model, tokenizer, example: PromptExample) -> float:
    prompt_messages = generation_messages(example)
    full_messages = prompt_messages + [{"role": "assistant", "content": example.tail}]
    prompt_ids = _token_ids(tokenizer, prompt_messages, add_generation_prompt=True)
    full_ids = _token_ids(tokenizer, full_messages, add_generation_prompt=False)
    labels = prompt_labels(full_ids, prompt_ids)
    device = _device(model)
    input_ids = torch.tensor([full_ids], device=device)
    label_ids = torch.tensor([labels], device=device)
    with torch.inference_mode():
        loss = model(input_ids=input_ids, labels=label_ids).loss
    return float(loss.item())


def _token_ids(tokenizer, messages: list[dict], *, add_generation_prompt: bool) -> list[int]:
    encoded = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=add_generation_prompt,
        tokenize=True,
        return_dict=True,
    )
    ids = encoded["input_ids"]
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return list(ids)


def unload(model) -> None:
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _device(model) -> torch.device:
    return next(model.parameters()).device
