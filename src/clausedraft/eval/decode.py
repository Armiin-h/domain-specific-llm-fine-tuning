"""Decoding settings shared by the base model and the adapter."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DecodeSettings:
    max_new_tokens: int = 128
    do_sample: bool = False

    def __post_init__(self) -> None:
        if self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be a positive integer")
