"""Write the base-versus-adapter gallery."""

from __future__ import annotations

from clausedraft.eval.decode import DecodeSettings
from clausedraft.eval.metrics import Comparison, summarize


def render_gallery(
    rows: list[Comparison],
    *,
    model_name: str,
    adapter: str | None,
    decode: DecodeSettings,
    note: str | None = None,
) -> str:
    summary = summarize(rows)
    lines = [
        "# Held-out continuations",
        "",
        "Same prompts and the same greedy decoding for the untouched base model and the adapter. These are drafting demos, not legal advice.",
        "",
    ]
    if note:
        lines.extend([note, ""])
    lines.extend(
        [
            f"- Base model: `{model_name}`",
            f"- Adapter: `{adapter}`" if adapter else "- Adapter: not scored",
            f"- Decoding: greedy, at most {decode.max_new_tokens} new tokens",
            "",
            "## Summary",
            "",
            _summary_line("Base word F1", summary["base_word_f1"]),
            _summary_line("Adapter word F1", summary["adapter_word_f1"]),
            _summary_line("Base in register", summary["base_in_register"]),
            _summary_line("Adapter in register", summary["adapter_in_register"]),
            _summary_line("Base gold NLL", summary["base_nll"]),
            _summary_line("Adapter gold NLL", summary["adapter_nll"]),
            "",
        ]
    )
    for row in rows:
        lines.extend(
            [
                f"## {row.clause_type}",
                "",
                _block("Prefix", row.prefix),
                _block("Gold tail", row.gold),
                _block("Base", row.base.text),
                _score_line("Base", row.base.word_f1, row.base.in_register, row.base.nll),
                "",
            ]
        )
        if row.adapter is None:
            lines.extend(["Adapter was not scored for this prompt.", ""])
            continue
        lines.extend(
            [
                _block("Adapter", row.adapter.text),
                _score_line("Adapter", row.adapter.word_f1, row.adapter.in_register, row.adapter.nll),
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def _summary_line(label: str, value: float | None) -> str:
    if value is None:
        return f"- {label}: n/a"
    return f"- {label}: {value:.3f}"


def _score_line(name: str, f1: float, in_register: bool, nll: float | None) -> str:
    register = "in register" if in_register else "out of register"
    nll_text = "n/a" if nll is None else f"{nll:.3f}"
    return f"{name}: word F1 {f1:.3f}, {register}, gold NLL {nll_text}."


def _block(title: str, text: str) -> str:
    return f"**{title}**\n\n```text\n{text.strip()}\n```"
