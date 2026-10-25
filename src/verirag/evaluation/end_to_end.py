from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from verirag.domain.answer import Answer, AnswerStatus
from verirag.evaluation.cases import Case, CaseSet
from verirag.ports import LLMError


class Asker(Protocol):
    """Anything that answers a question: the Assistant, or a fake in tests."""

    def ask(self, question: str) -> Answer: ...


@dataclass(frozen=True)
class Outcome:
    """What happened to one question."""

    question: str
    answerable: bool
    status: str  # "answered", "refused" or "error"
    correct_source: bool  # answered AND cited an expected file
    text: str
    citations: tuple[str, ...]


@dataclass(frozen=True)
class EndToEndReport:
    outcomes: tuple[Outcome, ...]

    def _count(self, answerable: bool, status: str) -> int:
        return sum(1 for o in self.outcomes if o.answerable is answerable and o.status == status)

    @property
    def errors(self) -> int:
        return sum(1 for o in self.outcomes if o.status == "error")

    @property
    def answered_with_correct_source(self) -> int:
        return sum(1 for o in self.outcomes if o.answerable and o.correct_source)

    @property
    def answered_with_other_source(self) -> int:
        return self._count(True, "answered") - self.answered_with_correct_source

    @property
    def wrongly_refused(self) -> int:
        return self._count(True, "refused")

    @property
    def correctly_refused(self) -> int:
        return self._count(False, "refused")

    @property
    def wrongly_answered(self) -> int:
        return self._count(False, "answered")

    def completed(self, answerable: bool) -> int:
        """Questions of this kind that ran to the end (model errors are excluded)."""
        return sum(1 for o in self.outcomes if o.answerable is answerable and o.status != "error")


def evaluate_end_to_end(
    asker: Asker, cases: CaseSet, previous: Sequence[Outcome] = ()
) -> EndToEndReport:
    """Ask every question. A model error (network, quota) is recorded as an
    error and left out of the rates, so a busy server cannot look like a refusal.

    `previous` holds the outcomes of an earlier run. Questions that finished
    there (answered or refused) are kept as they are; only questions that
    failed with an error are asked again. This lets a run be completed after
    a busy period without repeating, or hiding, the work already done.
    """
    finished = {o.question: o for o in previous if o.status != "error"}

    def outcome(case: Case, answerable: bool) -> Outcome:
        old = finished.get(case.question)
        if old is not None and old.answerable is answerable:
            return old
        return _run(asker, case, answerable)

    outcomes = [outcome(case, answerable=True) for case in cases.answerable]
    outcomes += [outcome(case, answerable=False) for case in cases.unanswerable]
    return EndToEndReport(tuple(outcomes))


def save_outcomes(report: EndToEndReport, path: Path) -> None:
    rows = [vars(outcome) for outcome in report.outcomes]
    path.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_outcomes(path: Path) -> list[Outcome]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    return [Outcome(**{**row, "citations": tuple(row["citations"])}) for row in rows]


def _run(asker: Asker, case: Case, answerable: bool) -> Outcome:
    try:
        answer = asker.ask(case.question)
    except LLMError as error:
        return Outcome(case.question, answerable, "error", False, str(error), ())
    cited_files = {citation.split(",")[0] for citation in answer.citations}
    correct = (
        answerable
        and answer.status is AnswerStatus.ANSWERED
        and bool(cited_files & set(case.sources))
    )
    return Outcome(
        case.question, answerable, answer.status.value, correct, answer.text, answer.citations
    )


def format_end_to_end(report: EndToEndReport) -> str:
    """A Markdown table. Rates are over the questions that ran to the end."""
    a, u = report.completed(True), report.completed(False)

    def share(count: int, total: int) -> str:
        return f"{count}/{total} ({count / total:.0%})" if total else "0/0"

    rows = [
        (
            "Answerable: answered, cites a correct article",
            share(report.answered_with_correct_source, a),
        ),
        (
            "Answerable: answered, cites another article",
            share(report.answered_with_other_source, a),
        ),
        ("Answerable: refused", share(report.wrongly_refused, a)),
        ("Unanswerable: refused", share(report.correctly_refused, u)),
        ("Unanswerable: answered anyway", share(report.wrongly_answered, u)),
        ("Runs that failed with a model error (left out above)", str(report.errors)),
    ]
    lines = ["| Measure | Result |", "|---|---|"]
    lines += [f"| {name} | {value} |" for name, value in rows]
    return "\n".join(lines)
