"""Named decoding setups for a live completion."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DecodingPreset:
    name: str
    max_new_tokens: int
    do_sample: bool
    temperature: float
    top_p: float


PRESETS = {
    "greedy": DecodingPreset("greedy", max_new_tokens=128, do_sample=False, temperature=1.0, top_p=1.0),
    "draft": DecodingPreset("draft", max_new_tokens=160, do_sample=True, temperature=0.7, top_p=0.9),
}


def get_preset(name: str) -> DecodingPreset:
    key = name.strip().casefold()
    try:
        return PRESETS[key]
    except KeyError as exc:
        known = ", ".join(PRESETS)
        raise ValueError(f"unknown preset: {name}. Choose one of {known}.") from exc
