import pytest

from verirag.domain.answer import Answer, AnswerStatus, Draft
from verirag.domain.verdict import Claim


def test_answered_answer_needs_a_citation() -> None:
    with pytest.raises(ValueError, match="cite"):
        Answer(AnswerStatus.ANSWERED, "text")


def test_refused_answer_may_have_no_citation() -> None:
    assert Answer(AnswerStatus.REFUSED, "I am not sure.").citations == ()


def test_answer_needs_text() -> None:
    with pytest.raises(ValueError, match="text"):
        Answer(AnswerStatus.REFUSED, " ")


def test_draft_keeps_its_claims() -> None:
    draft = Draft("A and B.", (Claim("A", (1,)), Claim("B", (2,))))
    assert len(draft.claims) == 2


def test_draft_needs_text() -> None:
    with pytest.raises(ValueError, match="text"):
        Draft("", ())
