import pytest

from clausedraft.eval.generate import generation_messages, prompt_labels
from clausedraft.eval.prompts import PromptExample


def test_generation_messages_stop_before_the_gold_tail():
    example = PromptExample("Notices", "All notices shall be", "sent in writing.")
    messages = generation_messages(example)
    assert [message["role"] for message in messages] == ["system", "user"]
    assert "All notices shall be" in messages[1]["content"]
    assert "sent in writing" not in messages[1]["content"]


def test_prompt_labels_mask_only_the_prompt():
    assert prompt_labels([1, 2, 3, 4], [1, 2]) == [-100, -100, 3, 4]


def test_prompt_labels_reject_a_mismatched_prefix():
    with pytest.raises(ValueError, match="prefix"):
        prompt_labels([1, 2, 3], [9])
