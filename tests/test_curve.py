import pytest

from clausedraft.train.curve import loss_curve_svg, write_loss_curve


def test_loss_curve_draws_train_and_eval():
    svg = loss_curve_svg(
        [
            {"step": 10, "loss": 2.0},
            {"step": 20, "loss": 1.4},
            {"step": 20, "eval_loss": 1.6},
        ]
    )
    assert svg.startswith("<svg")
    assert "polyline" in svg
    assert "#1f4e79" in svg
    assert "#b85c38" in svg


def test_loss_curve_rejects_an_empty_history():
    with pytest.raises(ValueError, match="no points"):
        loss_curve_svg([{"learning_rate": 0.0002}])


def test_write_loss_curve_one_point(tmp_path):
    path = tmp_path / "loss.svg"
    write_loss_curve(path, [{"step": 1, "loss": 1.8}])
    text = path.read_text(encoding="utf-8")
    assert "<circle" in text
