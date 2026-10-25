from __future__ import annotations

from typing import Any

from verirag.agents.graph import build_graph
from verirag.agents.router import DEFAULT_MIN_SCORE, Router
from verirag.agents.verifier import Verifier
from verirag.agents.writer import Writer
from verirag.domain.answer import Answer
from verirag.ports import LLM, Retriever, Tracer


class Assistant:
    """The front door: ask a question, get a verified answer or a refusal.

    It only assembles the parts. The same model serves the writer and the
    verifier, but each has its own prompt, so the verifier does not simply
    repeat the writer's thinking.
    """

    def __init__(
        self,
        retriever: Retriever,
        llm: LLM,
        tracer: Tracer,
        min_score: float = DEFAULT_MIN_SCORE,
        max_retries: int = 2,
        k: int = 4,
    ) -> None:
        self._graph: Any = build_graph(
            retriever,
            Writer(llm),
            Verifier(llm),
            Router(min_score),
            tracer,
            max_retries=max_retries,
            k=k,
        )

    def ask(self, question: str) -> Answer:
        answer: Answer = self._graph.invoke({"question": question})["answer"]
        return answer
