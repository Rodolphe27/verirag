from pathlib import Path

import pytest

from verirag.domain.chunk import Chunk
from verirag.evaluation.cases import Case, CaseSet, load_cases
from verirag.evaluation.retrieval import RetrievalReport, evaluate_retrieval, format_table


class Scripted:
    """Returns prepared chunks for each question."""

    def __init__(self, results: dict[str, list[Chunk]]) -> None:
        self.results = results

    def search(self, question: str, k: int = 4) -> list[Chunk]:
        return self.results.get(question, [])[:k]


def chunk(source: str, score: float = 0.8) -> Chunk:
    return Chunk("text", source, score=score)


CASES = CaseSet(
    answerable=(Case("q1", ("a.md",)), Case("q2", ("b.md",)), Case("q3", ("c.md", "d.md"))),
    unanswerable=(Case("off1"), Case("off2")),
)


def test_hit_rate_and_mrr_count_the_rank_of_the_first_correct_file() -> None:
    retriever = Scripted(
        {
            "q1": [chunk("a.md"), chunk("x.md")],  # rank 1
            "q2": [chunk("x.md"), chunk("b.md")],  # rank 2
            "q3": [chunk("x.md"), chunk("y.md")],  # miss
        }
    )
    report = evaluate_retrieval(retriever, CASES)
    assert report.hit_rate == pytest.approx(2 / 3)
    assert report.mrr == pytest.approx((1 + 0.5 + 0) / 3)


def test_any_listed_source_counts() -> None:
    retriever = Scripted({"q3": [chunk("d.md")]})
    assert evaluate_retrieval(retriever, CaseSet((CASES.answerable[2],), ())).hit_rate == 1.0


def test_a_correct_file_outside_the_top_k_is_a_miss() -> None:
    retriever = Scripted({"q1": [chunk("x.md"), chunk("a.md")]})
    one = CaseSet((CASES.answerable[0],), ())
    assert evaluate_retrieval(retriever, one, k=1).hit_rate == 0.0
    assert evaluate_retrieval(retriever, one, k=2).hit_rate == 1.0


def test_the_refusal_gate_is_measured_in_both_directions() -> None:
    retriever = Scripted(
        {
            "q1": [chunk("a.md", 0.9)],
            "q2": [chunk("b.md", 0.1)],  # a real question that would be refused by mistake
            "q3": [chunk("c.md", 0.5)],
            "off1": [chunk("a.md", 0.2)],  # correctly refused
            "off2": [chunk("a.md", 0.6)],  # wrongly answered
        }
    )
    report = evaluate_retrieval(retriever, CASES, min_score=0.3)
    assert report.false_refusals == pytest.approx(1 / 3)
    assert report.correct_refusals == pytest.approx(1 / 2)


def test_an_empty_question_set_gives_zeros_not_a_crash() -> None:
    report = evaluate_retrieval(Scripted({}), CaseSet((), ()))
    assert (report.hit_rate, report.mrr, report.correct_refusals) == (0.0, 0.0, 0.0)


def test_load_cases_reads_the_project_file() -> None:
    cases = load_cases(Path("eval/questions.json"))
    assert len(cases.answerable) >= 20
    assert len(cases.unanswerable) >= 5
    assert all(case.sources for case in cases.answerable)


def test_an_answerable_question_without_a_source_is_rejected(tmp_path: Path) -> None:
    file = tmp_path / "q.json"
    file.write_text('{"answerable": [{"question": "q", "sources": []}], "unanswerable": []}')
    with pytest.raises(ValueError, match="at least one source"):
        load_cases(file)


def test_the_table_has_a_header_and_one_row_per_mode() -> None:
    report = RetrievalReport(
        k=4, min_score=0.3, hit_rate=0.9, mrr=0.75, false_refusals=0.0, correct_refusals=1.0
    )
    table = format_table({"dense": report, "hybrid": report}).splitlines()
    assert table[0].startswith("| Search mode | Hit rate@4")
    assert table[2] == "| dense | 90% | 0.75 | 0% | 100% |"
    assert len(table) == 4
