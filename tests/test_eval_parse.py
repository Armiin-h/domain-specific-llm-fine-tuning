import pytest

from clausedraft.config import REPO_ROOT
from clausedraft.eval.decode import DecodeSettings
from clausedraft.eval.gallery import render_gallery
from clausedraft.eval.metrics import Comparison, score_continuation
from clausedraft.eval.parse import parse_gallery
from clausedraft.eval.prompts import FIXED_TYPES


def test_parse_round_trips_a_rendered_gallery():
    row = Comparison(
        clause_type="Notices",
        prefix="All notices shall be",
        gold="sent in writing.",
        base=score_continuation("Sure, I can help with that request.", "sent in writing."),
        adapter=score_continuation("sent in writing to each party.", "sent in writing."),
    )
    text = render_gallery(
        [row],
        model_name="Qwen/Qwen2.5-3B-Instruct",
        adapter="outputs/smoke",
        decode=DecodeSettings(),
        note="Smoke adapter only.",
    )
    report = parse_gallery(text)
    assert report.model_name == "Qwen/Qwen2.5-3B-Instruct"
    assert report.adapter == "outputs/smoke"
    assert report.max_new_tokens == 128
    assert report.note == "Smoke adapter only."
    assert report.rows[0].clause_type == "Notices"
    assert report.rows[0].prefix == "All notices shall be"
    assert report.rows[0].base.in_register is False
    assert report.rows[0].base.nll is None
    assert report.rows[0].adapter is not None
    assert report.rows[0].adapter.text == "sent in writing to each party."
    assert report.rows[0].adapter.in_register is True


def test_parse_allows_a_prompt_without_an_adapter_score():
    row = Comparison(
        clause_type="Waivers",
        prefix="No waiver shall be",
        gold="effective unless written.",
        base=score_continuation("effective unless it is written.", "effective unless written."),
        adapter=None,
    )
    report = parse_gallery(
        render_gallery([row], model_name="base", adapter=None, decode=DecodeSettings(max_new_tokens=32))
    )
    assert report.adapter is None
    assert report.rows[0].adapter is None
    assert report.max_new_tokens == 32


def test_saved_gallery_keeps_the_eight_held_out_types():
    text = (REPO_ROOT / "docs" / "gallery.md").read_text(encoding="utf-8")
    report = parse_gallery(text)
    assert report.model_name == "Qwen/Qwen2.5-3B-Instruct"
    assert report.adapter == "outputs/smoke"
    assert [row.clause_type for row in report.rows] == list(FIXED_TYPES)
    assert report.summary["base_word_f1"] == pytest.approx(0.392)
    assert report.summary["adapter_nll"] == pytest.approx(1.555)
    assert report.note is not None
    assert "one-step" in report.note


def test_parse_rejects_a_file_that_is_not_a_gallery():
    with pytest.raises(ValueError, match="model or the decoding line"):
        parse_gallery("not a gallery\n")
