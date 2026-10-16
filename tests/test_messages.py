import pytest

from clausedraft.data.messages import SYSTEM_PROMPT, ChatExample, render_messages, user_prompt


def test_user_prompt_names_the_type_and_the_prefix():
    text = user_prompt("Governing Laws", "This Agreement shall be governed by")
    assert "Clause type: Governing Laws" in text
    assert text.endswith("This Agreement shall be governed by")
    assert "Complete this clause:" in text


def test_chat_example_puts_the_tail_only_on_the_assistant():
    example = ChatExample(
        split="train",
        clause_type="Notices",
        prefix="The company shall notify",
        tail="the other party in writing.",
    )
    record = example.as_json()
    roles = [message["role"] for message in record["messages"]]
    assert roles == ["system", "user", "assistant"]
    assert record["messages"][0]["content"] == SYSTEM_PROMPT
    assert record["messages"][2]["content"] == "the other party in writing."
    assert "the other party in writing." not in record["messages"][1]["content"]


def test_render_messages_includes_each_role():
    example = ChatExample("validation", "Waivers", "Each party hereby", "waives trial by jury.")
    text = render_messages(example.as_json(), 1)
    assert "--- example 1 ---" in text
    assert "[assistant]" in text
    assert "waives trial by jury." in text


def test_user_prompt_rejects_an_empty_type():
    with pytest.raises(ValueError, match="clause type"):
        user_prompt("  ", "The company shall")
