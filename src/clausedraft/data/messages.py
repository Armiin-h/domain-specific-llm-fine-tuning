"""Chat messages for a clause-completion example."""

from __future__ import annotations

from dataclasses import dataclass

SYSTEM_PROMPT = (
    "You complete commercial contract clauses. "
    "Continue in the same legal register. Do not add commentary."
)


def user_prompt(clause_type: str, prefix: str) -> str:
    if not clause_type.strip():
        raise ValueError("clause type is empty")
    if not prefix.strip():
        raise ValueError("prefix is empty")
    return f"Clause type: {clause_type}\n\nComplete this clause:\n{prefix}"


@dataclass(frozen=True)
class ChatExample:
    split: str
    clause_type: str
    prefix: str
    tail: str

    def as_json(self) -> dict:
        if not self.tail.strip():
            raise ValueError("tail is empty")
        return {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt(self.clause_type, self.prefix)},
                {"role": "assistant", "content": self.tail},
            ]
        }


def render_messages(record: dict, index: int) -> str:
    lines = [f"--- example {index} ---"]
    for message in record["messages"]:
        lines.append(f"[{message['role']}]")
        lines.append(message["content"])
        lines.append("")
    return "\n".join(lines).rstrip()
