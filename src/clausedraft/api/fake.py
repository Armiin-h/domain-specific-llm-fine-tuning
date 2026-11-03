"""A drafter that returns a fixed continuation, for tests and the page walkthrough."""

from __future__ import annotations

from clausedraft.infer.complete import DraftResult
from clausedraft.infer.health import health_report
from clausedraft.infer.presets import get_preset
from clausedraft.infer.request import make_request


class EchoDrafter:
    def complete(self, clause_type: str, prefix: str, preset: str = "greedy") -> DraftResult:
        request = make_request(clause_type, prefix)
        chosen = get_preset(preset)
        return DraftResult(
            clause_type=request.clause_type,
            prefix=request.prefix,
            continuation="shall be construed under the laws of the State of Delaware.",
            preset=chosen.name,
        )

    def health(self):
        return health_report(model_loaded=True, adapter=None, cuda=False)
