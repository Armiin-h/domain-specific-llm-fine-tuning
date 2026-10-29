import pytest

from clausedraft.infer.presets import get_preset


def test_greedy_preset_does_not_sample():
    preset = get_preset("greedy")
    assert preset.do_sample is False
    assert preset.max_new_tokens == 128


def test_draft_preset_samples_with_a_low_temperature():
    preset = get_preset("Draft")
    assert preset.do_sample is True
    assert preset.temperature == pytest.approx(0.7)
    assert preset.top_p == pytest.approx(0.9)


def test_unknown_preset_names_the_choices():
    with pytest.raises(ValueError, match="greedy"):
        get_preset("creative")
