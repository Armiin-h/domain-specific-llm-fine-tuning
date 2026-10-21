import json

from clausedraft.train.history import loss_rows, write_loss_log


def test_loss_rows_keep_train_and_eval_loss():
    rows = loss_rows(
        [
            {"step": 1, "loss": 2.5, "epoch": 0.1, "learning_rate": 1e-4},
            {"step": 1, "eval_loss": 2.4, "epoch": 0.1},
            {"train_runtime": 3},
        ]
    )
    assert rows == [
        {"step": 1, "epoch": 0.1, "loss": 2.5},
        {"step": 1, "epoch": 0.1, "eval_loss": 2.4},
    ]


def test_write_loss_log_roundtrip(tmp_path):
    path = tmp_path / "loss.json"
    write_loss_log(path, [{"step": 1, "loss": 1.2}], {"train_loss": 1.2})
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["metrics"]["train_loss"] == 1.2
    assert payload["history"][0]["loss"] == 1.2
