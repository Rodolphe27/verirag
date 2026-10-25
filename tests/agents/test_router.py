from verirag.agents.router import Route, Router, refusal
from verirag.domain.answer import AnswerStatus
from verirag.domain.chunk import Chunk


def chunk(score: float, source: str = "a.md") -> Chunk:
    return Chunk("some text", source, score=score)


def test_strong_chunks_go_to_the_writer() -> None:
    decision = Router().decide("question?", [chunk(0.8), chunk(0.5)])
    assert decision.route is Route.DOCUMENTS
    assert len(decision.chunks) == 2


def test_weak_chunks_are_dropped_before_the_writer() -> None:
    decision = Router().decide("question?", [chunk(0.8), chunk(0.1, "weak.md")])
    assert [c.source for c in decision.chunks] == ["a.md"]


def test_only_weak_chunks_mean_refuse() -> None:
    decision = Router().decide("question?", [chunk(0.2), chunk(0.15)])
    assert decision.route is Route.REFUSE
    assert "nothing related" in decision.reason


def test_no_chunks_mean_refuse() -> None:
    assert Router().decide("question?", []).route is Route.REFUSE


def test_an_empty_question_is_refused() -> None:
    decision = Router().decide("   ", [chunk(0.9)])
    assert decision.route is Route.REFUSE
    assert decision.reason == "the question is empty"


def test_the_threshold_can_be_changed() -> None:
    assert Router(min_score=0.6).decide("q", [chunk(0.5)]).route is Route.REFUSE
    assert Router(min_score=0.4).decide("q", [chunk(0.5)]).route is Route.DOCUMENTS


def test_a_score_equal_to_the_threshold_passes() -> None:
    assert Router(min_score=0.3).decide("q", [chunk(0.3)]).route is Route.DOCUMENTS


def test_a_refusal_without_sources_has_no_citations() -> None:
    answer = refusal("the documents contain nothing related")
    assert answer.status is AnswerStatus.REFUSED
    assert answer.citations == ()
    assert "will not guess" in answer.text


def test_a_refusal_lists_the_closest_sources_once_each() -> None:
    answer = refusal("not verified", [chunk(0.5, "a.md"), chunk(0.4, "a.md"), chunk(0.3, "b.md")])
    assert answer.citations == ("a.md", "b.md")
    assert "closest passages" in answer.text
