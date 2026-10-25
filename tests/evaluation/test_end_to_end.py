from verirag.domain.answer import Answer, AnswerStatus
from verirag.evaluation.cases import Case, CaseSet
from verirag.evaluation.end_to_end import EndToEndReport, evaluate_end_to_end, format_end_to_end
from verirag.ports import LLMError

CASES = CaseSet(
    answerable=(
        Case("q1", ("a.md",)),
        Case("q2", ("b.md",)),
        Case("q3", ("c.md",)),
        Case("q4", ("d.md",)),
    ),
    unanswerable=(Case("off1"), Case("off2"), Case("off3")),
)


def answered(*citations: str) -> Answer:
    return Answer(AnswerStatus.ANSWERED, "text", citations)


REFUSED = Answer(AnswerStatus.REFUSED, "I am not sure.")


class Scripted:
    def __init__(self, results: dict[str, Answer | Exception]) -> None:
        self.results = results

    def ask(self, question: str) -> Answer:
        result = self.results[question]
        if isinstance(result, Exception):
            raise result
        return result


def report() -> EndToEndReport:
    return evaluate_end_to_end(
        Scripted(
            {
                "q1": answered("a.md", "x.md"),  # correct source
                "q2": answered("other.md"),  # answered, wrong article
                "q3": REFUSED,  # refused by mistake
                "q4": LLMError("model call failed with HTTP 503"),
                "off1": REFUSED,
                "off2": answered("a.md"),  # should have been refused
                "off3": REFUSED,
            }
        ),
        CASES,
    )


def test_each_kind_of_outcome_is_counted() -> None:
    r = report()
    assert r.answered_with_correct_source == 1
    assert r.answered_with_other_source == 1
    assert r.wrongly_refused == 1
    assert r.correctly_refused == 2
    assert r.wrongly_answered == 1
    assert r.errors == 1


def test_a_page_in_the_citation_is_ignored_when_matching_the_file() -> None:
    cases = CaseSet((Case("q", ("doc.pdf",)),), ())
    result = evaluate_end_to_end(Scripted({"q": answered("doc.pdf, p. 3")}), cases)
    assert result.answered_with_correct_source == 1


def test_model_errors_are_left_out_of_the_rates_and_shown_separately() -> None:
    table = format_end_to_end(report())
    assert "| Answerable: answered, cites a correct article | 1/3 (33%) |" in table
    assert "| Unanswerable: refused | 2/3 (67%) |" in table
    assert "| Runs that failed with a model error (left out above) | 1 |" in table
