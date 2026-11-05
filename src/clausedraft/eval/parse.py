"""Read a gallery written by render_gallery back into rows."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class GallerySide:
    text: str
    word_f1: float
    in_register: bool
    nll: float | None


@dataclass(frozen=True)
class GalleryRow:
    clause_type: str
    prefix: str
    gold: str
    base: GallerySide
    adapter: GallerySide | None


@dataclass(frozen=True)
class GalleryReport:
    model_name: str
    adapter: str | None
    max_new_tokens: int
    note: str | None
    summary: dict[str, float | None]
    rows: tuple[GalleryRow, ...]


_BLOCK = re.compile(r"\*\*(Prefix|Gold tail|Base|Adapter)\*\*\n\n```text\n(.*?)```", re.DOTALL)
_SCORE = re.compile(
    r"^(Base|Adapter): word F1 ([0-9.]+), (in register|out of register), gold NLL ([0-9.]+|n/a)\.$",
    re.MULTILINE,
)
_MODEL = re.compile(r"^- Base model: `([^`]+)`$", re.MULTILINE)
_ADAPTER = re.compile(r"^- Adapter: `([^`]*)`$", re.MULTILINE)
_DECODE = re.compile(r"^- Decoding: greedy, at most (\d+) new tokens$", re.MULTILINE)
_SUMMARY = re.compile(
    r"^- (Base word F1|Adapter word F1|Base in register|Adapter in register|Base gold NLL|Adapter gold NLL): ([0-9.]+|n/a)$",
    re.MULTILINE,
)
_SUMMARY_KEYS = {
    "Base word F1": "base_word_f1",
    "Adapter word F1": "adapter_word_f1",
    "Base in register": "base_in_register",
    "Adapter in register": "adapter_in_register",
    "Base gold NLL": "base_nll",
    "Adapter gold NLL": "adapter_nll",
}


def parse_gallery(text: str) -> GalleryReport:
    model = _MODEL.search(text)
    decode = _DECODE.search(text)
    if model is None or decode is None:
        raise ValueError("gallery is missing the model or the decoding line")
    if re.search(r"^- Adapter: not scored$", text, re.MULTILINE):
        adapter = None
    else:
        found = _ADAPTER.search(text)
        if found is None:
            raise ValueError("gallery is missing the adapter line")
        adapter = found.group(1)
    rows = tuple(_rows(text))
    return GalleryReport(
        model_name=model.group(1),
        adapter=adapter,
        max_new_tokens=int(decode.group(1)),
        note=_note(text),
        summary=_summary(text),
        rows=rows,
    )


def _note(text: str) -> str | None:
    head, _, _rest = text.partition("- Base model:")
    paragraphs = [part.strip() for part in head.split("\n\n") if part.strip()]
    extras = [part for part in paragraphs[2:] if not part.startswith("#")]
    if not extras:
        return None
    return "\n\n".join(extras)


def _summary(text: str) -> dict[str, float | None]:
    found = {label: value for label, value in _SUMMARY.findall(text)}
    summary: dict[str, float | None] = {}
    for label, key in _SUMMARY_KEYS.items():
        raw = found.get(label)
        if raw is None or raw == "n/a":
            summary[key] = None
        else:
            summary[key] = float(raw)
    return summary


def _rows(text: str) -> list[GalleryRow]:
    rows: list[GalleryRow] = []
    for part in re.split(r"\n## ", text)[1:]:
        title, _, body = part.partition("\n")
        clause_type = title.strip()
        if clause_type == "Summary" or not clause_type:
            continue
        blocks = {name: content.strip("\n") for name, content in _BLOCK.findall(body)}
        scores = {name: groups for name, *groups in _SCORE.findall(body)}
        if "Prefix" not in blocks or "Gold tail" not in blocks or "Base" not in blocks or "Base" not in scores:
            raise ValueError(f"gallery section {clause_type} is incomplete")
        adapter = None
        if "Adapter was not scored for this prompt." not in body:
            if "Adapter" not in blocks or "Adapter" not in scores:
                raise ValueError(f"gallery section {clause_type} is missing the adapter")
            adapter = _side(blocks["Adapter"], scores["Adapter"])
        rows.append(
            GalleryRow(
                clause_type=clause_type,
                prefix=blocks["Prefix"],
                gold=blocks["Gold tail"],
                base=_side(blocks["Base"], scores["Base"]),
                adapter=adapter,
            )
        )
    return rows


def _side(text: str, groups: list[str]) -> GallerySide:
    f1, register, nll = groups
    return GallerySide(
        text=text,
        word_f1=float(f1),
        in_register=register == "in register",
        nll=None if nll == "n/a" else float(nll),
    )
