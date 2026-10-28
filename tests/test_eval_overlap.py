import pytest

from clausedraft.eval.overlap import word_f1


def test_word_f1_is_one_for_an_exact_copy():
    text = "The parties shall agree in writing."
    assert word_f1(text, text) == pytest.approx(1.0)


def test_word_f1_penalizes_extra_words():
    score = word_f1("the parties shall agree today", "the parties shall agree")
    assert score == pytest.approx(2 * 0.8 * 1 / 1.8)


def test_word_f1_is_zero_when_either_side_is_empty():
    assert word_f1("", "the parties shall agree") == 0.0
    assert word_f1("shall", "") == 0.0
