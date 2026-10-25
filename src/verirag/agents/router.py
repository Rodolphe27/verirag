from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from verirag.domain.answer import Answer, AnswerStatus
from verirag.domain.chunk import Chunk

DEFAULT_MIN_SCORE = 0.30


class Route(Enum):
    DOCUMENTS = "documents"
    REFUSE = "refuse"


@dataclass(frozen=True)
class Decision:
    route: Route
    chunks: tuple[Chunk, ...]
    reason: str = ""


class Router:
    """Decides if the search found enough to answer from. Plain code, no model call.

    There is deliberately no "answer without documents" route: such an answer
    could not be verified, and verified answers are the point of the project.
    The decision uses the cosine score of the chunks (0 to 1). Questions
    unrelated to the documents score about 0.1 to 0.2 and real matches 0.4 and
    more, so 0.30 separates them; the value can be changed with `min_score`.
    """

    def __init__(self, min_score: float = DEFAULT_MIN_SCORE) -> None:
        self._min_score = min_score

    def decide(self, question: str, chunks: list[Chunk]) -> Decision:
        if not question.strip():
            return Decision(Route.REFUSE, (), "the question is empty")
        good = tuple(chunk for chunk in chunks if chunk.score >= self._min_score)
        if not good:
            return Decision(
                Route.REFUSE, (), "the documents contain nothing related to the question"
            )
        return Decision(Route.DOCUMENTS, good)


def refusal(reason: str, closest: list[Chunk] | tuple[Chunk, ...] = ()) -> Answer:
    """The standard "I am not sure" answer. It says why, and where the closest text was."""
    text = f"I am not sure, so I will not guess: {reason}."
    citations = tuple(dict.fromkeys(chunk.citation for chunk in closest))
    if citations:
        text += " The closest passages I found are listed as sources."
    return Answer(AnswerStatus.REFUSED, text, citations)
