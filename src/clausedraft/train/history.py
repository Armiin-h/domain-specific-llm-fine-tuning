"""Write the loss history from a trainer log."""

from __future__ import annotations

import json
from pathlib import Path


def loss_rows(log_history: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for entry in log_history:
        if "loss" not in entry and "eval_loss" not in entry:
            continue
        row: dict = {}
        if "step" in entry:
            row["step"] = entry["step"]
        if "epoch" in entry:
            row["epoch"] = entry["epoch"]
        if "loss" in entry:
            row["loss"] = entry["loss"]
        if "eval_loss" in entry:
            row["eval_loss"] = entry["eval_loss"]
        rows.append(row)
    return rows


def write_loss_log(path: Path, rows: list[dict], metrics: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"metrics": metrics, "history": rows}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
