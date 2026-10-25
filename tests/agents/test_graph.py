import json
from dataclasses import dataclass
from typing import Any

from tests.helpers.fake_llm import ScriptedLLM
from verirag.agents.graph import build_graph
from verirag.agents.router import Router
from verirag.agents.verifier import Verifier
from verirag.agents.writer import Writer
from verirag.domain.answer import Answer, AnswerStatus
from verirag.domain.chunk import Chunk
from verirag.tracing import MemoryTracer

FINE = Chunk("Fines can reach 35 million euros.", "article-99.md", score=0.8)
LOGS = Chunk("Providers must keep logs.", "article-12.md", page=2, score=0.6)


class FixedRetriever:
    def __init__(self, *chunks: Chunk) -> None:
        self.chunks = list(chunks)

    def search(self, question: str, k: int = 4) -> list[Chunk]:
        return self.chunks[:k]


def draft_reply(
    text: str = "Fines reach 35 million euros.", sources: tuple[int, ...] = (1,)
) -> str:
    claims = [{"text": text, "sources": list(sources)}]
    return json.dumps({"answer": text, "claims": claims})


def verdict_reply(supported: bool, evidence: str = "", reason: str = "") -> str:
    item = {"claim": 1, "supported": supported, "evidence": evidence, "reason": reason}
    return json.dumps({"verdicts": [item]})


PASS = verdict_reply(True, "fines can reach 35 million euros")
FAIL = verdict_reply(False, reason="the sources give no such figure")


@dataclass
class Run:
    answer: Answer
    writer_llm: ScriptedLLM
    verifier_llm: ScriptedLLM
    tracer: MemoryTracer


def run(
    writer_replies: list[str],
    verifier_replies: list[str],
    chunks: tuple[Chunk, ...] = (FINE, LOGS),
    max_retries: int = 2,
) -> Run:
    writer_llm, verifier_llm, tracer = (
        ScriptedLLM(*writer_replies),
        ScriptedLLM(*verifier_replies),
        MemoryTracer(),
    )
    graph: Any = build_graph(
        FixedRetriever(*chunks),
        Writer(writer_llm),
        Verifier(verifier_llm),
        Router(),
        tracer,
        max_retries=max_retries,
    )
    state = graph.invoke({"question": "How high are fines?"})
    return Run(state["answer"], writer_llm, verifier_llm, tracer)


def test_happy_path_returns_a_verified_answer_with_exact_citations() -> None:
    result = run([draft_reply()], [PASS])
    assert result.answer.status is AnswerStatus.ANSWERED
    assert result.answer.text == "Fines reach 35 million euros."
    assert result.answer.citations == ("article-99.md",)
    assert result.tracer.steps() == ["retriever", "router", "writer", "verifier", "finish"]


def test_citations_come_from_the_verified_evidence_not_from_the_writers_numbers() -> None:
    # The writer wrongly cites source 2 (the logs article); the evidence is in source 1.
    result = run([draft_reply(sources=(2,))], [PASS])
    assert result.answer.citations == ("article-99.md",)


def test_a_failed_check_sends_the_feedback_to_the_writer_and_can_recover() -> None:
    result = run([draft_reply("Fines are 1 euro."), draft_reply()], [FAIL, PASS])
    assert result.answer.status is AnswerStatus.ANSWERED
    assert "the sources give no such figure" in result.writer_llm.prompts[1]
    assert "NOT supported" not in result.writer_llm.prompts[0]
    assert result.tracer.steps().count("writer") == 2


def test_if_the_answer_is_never_verified_the_system_refuses_after_the_retry_limit() -> None:
    result = run([draft_reply()] * 3, [FAIL] * 3)
    assert result.answer.status is AnswerStatus.REFUSED
    assert "could not be verified" in result.answer.text
    assert len(result.writer_llm.prompts) == 3  # the first try plus two retries, never more
    assert result.answer.citations == ("article-99.md", "article-12.md, p. 2")


def test_the_retry_limit_is_a_setting() -> None:
    result = run([draft_reply()], [FAIL], max_retries=0)
    assert result.answer.status is AnswerStatus.REFUSED
    assert len(result.writer_llm.prompts) == 1


def test_weak_search_results_are_refused_without_calling_any_model() -> None:
    weak = Chunk("Unrelated text.", "other.md", score=0.12)
    result = run([], [], chunks=(weak,))
    assert result.answer.status is AnswerStatus.REFUSED
    assert result.answer.citations == ()
    assert result.writer_llm.prompts == [] and result.verifier_llm.prompts == []
    assert result.tracer.steps() == ["retriever", "router"]


def test_when_the_writer_finds_no_answer_in_the_sources_it_stops_at_once() -> None:
    empty = json.dumps({"answer": "The sources do not say.", "claims": []})
    result = run([empty], [])
    assert result.answer.status is AnswerStatus.REFUSED
    assert "do not contain an answer" in result.answer.text
    assert result.verifier_llm.prompts == []


def test_an_unreadable_writer_reply_is_retried_with_a_format_reminder() -> None:
    result = run(["sorry, no JSON", draft_reply()], [PASS])
    assert result.answer.status is AnswerStatus.ANSWERED
    assert "Follow the JSON format exactly" in result.writer_llm.prompts[1]


def test_a_writer_that_never_produces_readable_output_ends_in_a_refusal() -> None:
    result = run(["nope"] * 3, [])
    assert result.answer.status is AnswerStatus.REFUSED
    assert len(result.writer_llm.prompts) == 3


def test_the_trace_explains_why_claims_failed() -> None:
    result = run([draft_reply("Fines are 1 euro.")], [FAIL], max_retries=0)
    verifier_data = dict(result.tracer.records)["verifier"]
    assert verifier_data["failures"] == [
        {"claim": "Fines are 1 euro.", "reason": "the sources give no such figure"}
    ]
