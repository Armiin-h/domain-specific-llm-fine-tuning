from clausedraft.data.corpus import Provision
from clausedraft.data.stats import format_stats, summarize


def _row(text: str, clause_type: str, split: str) -> Provision:
    return Provision(text, clause_type, split)


def test_summarize_counts_splits_labels_and_length_bounds():
    rows = [
        _row("a" * 10, "Notices", "train"),
        _row("b" * 100, "Notices", "train"),
        _row("c" * 50, "Waivers", "test"),
    ]
    stats = summarize(rows, min_chars=40, max_chars=80)
    assert stats.split_counts == {"train": 2, "test": 1}
    assert stats.label_counts == {"Notices": 2, "Waivers": 1}
    assert stats.char_min == 10
    assert stats.char_max == 100
    assert stats.below_min == 1
    assert stats.above_max == 1
    assert stats.char_p50 == 50


def test_format_stats_lists_common_and_rare_labels():
    rows = [
        _row("x" * 80, "Notices", "train"),
        _row("y" * 80, "Notices", "train"),
        _row("z" * 80, "Books", "validation"),
    ]
    text = format_stats(summarize(rows, min_chars=40, max_chars=4000), top_n=1)
    assert "train: 2" in text
    assert "most_common:\n  Notices: 2" in text
    assert "least_common:\n  Books: 1" in text


def test_summarize_empty_list():
    stats = summarize([], min_chars=80, max_chars=4000)
    assert stats.rows == 0
    assert stats.char_mean == 0
