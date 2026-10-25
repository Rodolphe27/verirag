from __future__ import annotations

from collections.abc import Mapping

from verirag.ports import Tracer


class CompositeTracer:
    """Sends every record to several tracers, for example the screen and a file."""

    def __init__(self, *tracers: Tracer) -> None:
        self._tracers = tracers

    def record(self, step: str, data: Mapping[str, object]) -> None:
        for tracer in self._tracers:
            tracer.record(step, data)
