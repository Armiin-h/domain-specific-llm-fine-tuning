from clausedraft.train.record import write_run_record


def test_write_run_record_keeps_the_log_the_card_and_the_curve(tmp_path):
    write_run_record(
        tmp_path,
        [{"step": 1, "loss": 1.5}, {"step": 2, "loss": 1.2}],
        {"train_loss": 1.2},
        "# card\n",
    )
    assert "1.2" in (tmp_path / "loss.json").read_text(encoding="utf-8")
    assert (tmp_path / "model_card.md").read_text(encoding="utf-8") == "# card\n"
    assert (tmp_path / "loss.svg").read_text(encoding="utf-8").startswith("<svg")
