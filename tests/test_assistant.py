import json

from tests.helpers.fake_llm import ScriptedLLM
from verirag.assistant import Assistant
from verirag.domain.answer import AnswerStatus
from verirag.domain.chunk import Chunk
from verirag.tracing import MemoryTracer


class OneChunk:
    def search(self, question: str, k: int = 4) -> list[Chunk]:
        return [Chunk("Fines can reach 35 million euros.", "article-99.md", score=0.8)]


def test_one_model_serves_the_writer_and_then_the_verifier() -> None:
    draft = {
        "answer": "Up to 35 million.",
        "claims": [{"text": "Up to 35 million.", "sources": [1]}],
    }
    verdict = {"verdicts": [{"claim": 1, "supported": True, "evidence": "35 million euros"}]}
    llm = ScriptedLLM(json.dumps(draft), json.dumps(verdict))
    tracer = MemoryTracer()
    answer = Assistant(OneChunk(), llm, tracer).ask("How high are fines?")
    assert answer.status is AnswerStatus.ANSWERED
    assert answer.citations == ("article-99.md",)
    assert "writer of a question answering system" in llm.prompts[0]
    assert "verifier of a question answering system" in llm.prompts[1]


def test_the_minimum_score_is_a_setting() -> None:
    answer = Assistant(OneChunk(), ScriptedLLM(), MemoryTracer(), min_score=0.9).ask("q?")
    assert answer.status is AnswerStatus.REFUSED
