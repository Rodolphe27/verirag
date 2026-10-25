"""The fakes below are never called through a base class.

mypy (run in CI) proves that each fake fits its port. If a port changes and a
fake no longer fits, the type check fails.
"""

from collections.abc import Mapping

from verirag.domain.chunk import Chunk
from verirag.ports import LLM, Retriever, Tracer


class EchoLLM:
    def complete(self, prompt: str) -> str:
        return prompt


class OneChunkRetriever:
    def search(self, question: str, k: int = 4) -> list[Chunk]:
        return [Chunk(question, "fake.md")]


class ListTracer:
    def __init__(self) -> None:
        self.records: list[tuple[str, Mapping[str, object]]] = []

    def record(self, step: str, data: Mapping[str, object]) -> None:
        self.records.append((step, data))


def use_ports(llm: LLM, retriever: Retriever, tracer: Tracer) -> str:
    chunks = retriever.search("question")
    tracer.record("search", {"count": len(chunks)})
    return llm.complete(chunks[0].text)


def test_fakes_fit_the_ports() -> None:
    tracer = ListTracer()
    assert use_ports(EchoLLM(), OneChunkRetriever(), tracer) == "question"
    assert tracer.records == [("search", {"count": 1})]
