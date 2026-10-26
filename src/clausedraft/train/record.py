"""Write the loss log, the curve, and the model card next to an adapter."""

from __future__ import annotations

from pathlib import Path

from clausedraft.train.curve import write_loss_curve
from clausedraft.train.history import write_loss_log


def write_run_record(output_dir: Path, history: list[dict], metrics: dict, card: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_loss_log(output_dir / "loss.json", history, metrics)
    (output_dir / "model_card.md").write_text(card, encoding="utf-8")
    if history:
        write_loss_curve(output_dir / "loss.svg", history)
