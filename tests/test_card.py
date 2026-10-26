from clausedraft.config import load_config
from clausedraft.train.card import render_model_card


def test_model_card_records_the_run():
    config = load_config()
    text = render_model_card(
        config,
        base_model=config.model.name,
        train_sha256="abc123",
        train_rows=11933,
        eval_rows=128,
        train_loss=1.25,
        eval_loss=1.31,
        train_runtime=3600.0,
    )
    assert "Qwen/Qwen2.5-3B-Instruct" in text
    assert "`abc123`" in text
    assert "rank 16" in text
    assert "Train loss: 1.2500" in text
    assert "Eval loss: 1.3100" in text
    assert "not legal advice" in text
    assert "official test split is not part of this run" in text
    assert "128" in text


def test_model_card_marks_a_capped_smoke_run():
    config = load_config()
    text = render_model_card(
        config,
        base_model="Qwen/Qwen2.5-1.5B-Instruct",
        train_sha256="def",
        train_rows=32,
        eval_rows=0,
        train_loss=None,
        eval_loss=None,
        train_runtime=None,
        row_limit=32,
        max_steps=1,
    )
    assert "first 32 rows" in text
    assert "capped at 1 optimizer step" in text
    assert "Validation rows scored: none" in text
    assert "Train loss: not recorded" in text
