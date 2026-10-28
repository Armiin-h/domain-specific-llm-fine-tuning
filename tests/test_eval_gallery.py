from clausedraft.eval.decode import DecodeSettings
from clausedraft.eval.gallery import render_gallery
from clausedraft.eval.metrics import Comparison, score_continuation


def test_gallery_shows_both_continuations():
    row = Comparison(
        clause_type="Notices",
        prefix="All notices shall be",
        gold="sent in writing.",
        base=score_continuation("Sure, I can help with that request.", "sent in writing."),
        adapter=score_continuation("sent in writing to each party.", "sent in writing."),
    )
    text = render_gallery([row], model_name="Qwen/Qwen2.5-3B-Instruct", adapter="outputs/smoke", decode=DecodeSettings())
    assert "## Notices" in text
    assert "sent in writing to each party." in text
    assert "out of register" in text
    assert "outputs/smoke" in text
    assert "not legal advice" in text
