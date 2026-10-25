"""Ports: what the core needs from the outside world.

A port is an interface written with `typing.Protocol`. Any class that has the
right methods fits, with no base class to inherit. Real implementations
(Gemini, the search index, a log file) are "adapters" and live in other
packages. Tests plug in small fakes instead.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

from verirag.domain.chunk import Chunk


class LLM(Protocol):
    """A language model: text in, text out."""

    def complete(self, prompt: str) -> str: ...


class Retriever(Protocol):
    """A search over the documents: question in, best chunks out."""

    def search(self, question: str, k: int = 4) -> list[Chunk]: ...


class Tracer(Protocol):
    """A log of what the system did, one record per step."""

    def record(self, step: str, data: Mapping[str, object]) -> None: ...
