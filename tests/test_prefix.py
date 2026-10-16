from clausedraft.data.prefix import split_clause


def test_split_lands_on_a_word_near_the_fraction():
    text = (
        "This Agreement shall be governed by the laws of the State of Delaware "
        "without regard to its conflict of laws principles and the parties consent."
    )
    parts = split_clause(text, 0.4)
    assert parts is not None
    assert parts.prefix + " " + parts.tail == text
    assert abs(len(parts.prefix) / len(text) - 0.4) < 0.08
    assert not parts.prefix.endswith(" ")
    assert text.split(" ")[len(parts.prefix.split(" ")) - 1] == parts.prefix.split(" ")[-1]


def test_split_returns_none_when_a_side_would_be_too_short():
    assert split_clause("Short clause only.", 0.4) is None
    assert split_clause("one two", 0.5, min_prefix_chars=4, min_tail_chars=20) is None


def test_split_rejects_a_fraction_outside_range():
    import pytest

    with pytest.raises(ValueError, match="prefix_fraction"):
        split_clause("one two three four five six seven", 0.01)
