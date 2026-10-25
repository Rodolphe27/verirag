from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    """A piece of a source document, as found by the search.

    `frozen=True` makes the object read-only after creation, so a value can
    never change while it travels between agents.
    """

    text: str
    source: str
    page: int | None = None
    score: float = 0.0

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError("a chunk needs some text")
        if not self.source.strip():
            raise ValueError("a chunk needs a source name")
        if self.page is not None and self.page < 1:
            raise ValueError("pages start at 1")

    @property
    def citation(self) -> str:
        """Short reference shown to the reader, for example 'act.md, p. 3'."""
        return self.source if self.page is None else f"{self.source}, p. {self.page}"
