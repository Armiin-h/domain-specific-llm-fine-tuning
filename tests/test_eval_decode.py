import pytest

from clausedraft.eval.decode import DecodeSettings


def test_default_decoding_is_greedy():
    settings = DecodeSettings()
    assert settings.do_sample is False
    assert settings.max_new_tokens == 128


def test_decode_settings_reject_zero_tokens():
    with pytest.raises(ValueError, match="max_new_tokens"):
        DecodeSettings(max_new_tokens=0)
