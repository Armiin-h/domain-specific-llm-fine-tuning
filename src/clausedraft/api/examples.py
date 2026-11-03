"""Built-in prefixes for the page."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExamplePrefix:
    title: str
    clause_type: str
    prefix: str


EXAMPLES = (
    ExamplePrefix(
        title="Governing law",
        clause_type="Governing Laws",
        prefix="This Agreement shall be governed",
    ),
    ExamplePrefix(
        title="Notices",
        clause_type="Notices",
        prefix="All notices under this Agreement shall be",
    ),
    ExamplePrefix(
        title="Confidentiality",
        clause_type="Confidentiality",
        prefix="The Receiving Party shall hold in confidence",
    ),
    ExamplePrefix(
        title="Severability",
        clause_type="Severability",
        prefix="If any provision of this Agreement is held",
    ),
)
