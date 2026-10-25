from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from verirag.domain.verdict import Claim


class AnswerStatus(Enum):
    ANSWERED = "answered"
    REFUSED = "refused"


@dataclass(frozen=True)
class Draft:
    """An answer written by the writer agent, not yet checked."""

    text: str
    claims: tuple[Claim, ...]

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError("a draft needs some text")


@dataclass(frozen=True)
class Answer:
    """The final result shown to the user.

    An answered Answer must cite at least one source. A refused Answer carries
    the reason in `text` and may have no citations.
    """

    status: AnswerStatus
    text: str
    citations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError("an answer needs some text")
        if self.status is AnswerStatus.ANSWERED and not self.citations:
            raise ValueError("an answered question must cite its sources")
