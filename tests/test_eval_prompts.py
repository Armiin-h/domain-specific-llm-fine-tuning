import json

import pytest

from clausedraft.eval.prompts import parse_chat_row, select_fixed_prompts


def _row(clause_type: str, prefix: str, tail: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": f"Clause type: {clause_type}\n\nComplete this clause:\n{prefix}"},
            {"role": "assistant", "content": tail},
        ]
    }


def test_parse_chat_row_reads_type_prefix_and_tail():
    example = parse_chat_row(_row("Notices", "All notices shall be", "sent in writing."))
    assert example.clause_type == "Notices"
    assert example.prefix == "All notices shall be"
    assert example.tail == "sent in writing."


def test_parse_chat_row_rejects_a_plain_user_message():
    with pytest.raises(ValueError, match="clause prompt"):
        parse_chat_row({"messages": [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "x"}, {"role": "system", "content": "s"}]})


def test_select_fixed_prompts_keeps_the_requested_order(tmp_path):
    path = tmp_path / "test.jsonl"
    rows = [
        _row("Notices", "first notices prefix that is long enough", "tail one"),
        _row("Notices", "later notices prefix that is long enough", "tail two"),
        _row("Waivers", "a waiver prefix that is long enough", "tail three"),
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    selected = select_fixed_prompts(path, ("Waivers", "Notices", "Missing"))
    assert [item.clause_type for item in selected] == ["Waivers", "Notices"]
    assert selected[1].prefix.startswith("first notices")
