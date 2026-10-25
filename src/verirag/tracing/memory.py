from __future__ import annotations

from collections.abc import Mapping


class MemoryTracer:
    """Keeps the records in a list. Used in tests, and to show a trace after a run."""

    def __init__(self) -> None:
        self.records: list[tuple[str, dict[str, object]]] = []

    def record(self, step: str, data: Mapping[str, object]) -> None:
        # Copy the mapping so later changes by the caller cannot alter the log.
        self.records.append((step, dict(data)))

    def steps(self) -> list[str]:
        """The step names in order, handy for asserting the path a run took."""
        return [step for step, _ in self.records]
