from clausedraft.data.clean import clean_provisions, is_boilerplate, normalize_whitespace
from clausedraft.data.corpus import Provision


def test_normalize_whitespace_collapses_breaks_and_nbsp():
    assert normalize_whitespace(" Foo\u00a0\n\nbar\t baz ") == "Foo bar baz"


def test_boilerplate_catches_placeholders_and_short_headings():
    assert is_boilerplate("[Intentionally omitted.]")
    assert is_boilerplate("GOVERNING LAW")
    assert not is_boilerplate(
        "This Agreement shall be governed by the laws of the State of Delaware, "
        "without regard to its conflict of laws principles."
    )


def test_clean_provisions_applies_filters_in_order():
    rows = [
        Provision("  Reserved.  ", "General", "train"),
        Provision("short heading", "General", "train"),
        Provision("word " * 40, "Notices", "train"),
        Provision("The notice address is on the\n\nsignature page of this agreement.", "Notices", "validation"),
        Provision("x" * 50, "Waivers", "test"),
    ]
    result = clean_provisions(rows, min_chars=40, max_chars=80)
    assert result.dropped_boilerplate == 1
    assert result.dropped_short == 1
    assert result.dropped_long == 1
    assert len(result.kept) == 2
    assert result.kept[0].split == "validation"
    assert result.kept[0].text == "The notice address is on the signature page of this agreement."
    assert result.kept[1].split == "test"
