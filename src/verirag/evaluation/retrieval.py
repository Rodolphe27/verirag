from __future__ import annotations

from dataclasses import dataclass

from verirag.evaluation.cases import CaseSet
from verirag.ports import Retriever


@dataclass(frozen=True)
class RetrievalReport:
    """How well the search finds the right passages. All rates are between 0 and 1."""

    k: int
    min_score: float
    hit_rate: float  # answerable questions with a correct source among the top k chunks
    mrr: float  # mean reciprocal rank: 1 if the first chunk is right, 1/2 if the second, ...
    false_refusals: float  # answerable questions the score gate would wrongly refuse
    correct_refusals: float  # unanswerable questions the score gate refuses


def evaluate_retrieval(
    retriever: Retriever, cases: CaseSet, k: int = 4, min_score: float = 0.30
) -> RetrievalReport:
    """Run every question through the search and score the result.

    "Hit rate" asks: is a correct file somewhere in the top k? MRR asks how
    high it is ranked. The last two numbers measure the refusal gate: a
    question is refused when no chunk reaches `min_score`.
    """
    hits = 0
    reciprocal_ranks = 0.0
    false_refusals = 0
    for case in cases.answerable:
        chunks = retriever.search(case.question, k)
        rank = next(
            (position for position, chunk in enumerate(chunks, 1) if chunk.source in case.sources),
            None,
        )
        if rank is not None:
            hits += 1
            reciprocal_ranks += 1 / rank
        if not any(chunk.score >= min_score for chunk in chunks):
            false_refusals += 1

    correct_refusals = sum(
        1
        for case in cases.unanswerable
        if not any(chunk.score >= min_score for chunk in retriever.search(case.question, k))
    )
    return RetrievalReport(
        k=k,
        min_score=min_score,
        hit_rate=_rate(hits, len(cases.answerable)),
        mrr=_rate(reciprocal_ranks, len(cases.answerable)),
        false_refusals=_rate(false_refusals, len(cases.answerable)),
        correct_refusals=_rate(correct_refusals, len(cases.unanswerable)),
    )


def _rate(count: float, total: int) -> float:
    return count / total if total else 0.0


def format_table(reports: dict[str, RetrievalReport]) -> str:
    """A Markdown table, one row per search mode, ready to paste into the docs."""
    first = next(iter(reports.values()))
    lines = [
        f"| Search mode | Hit rate@{first.k} | MRR | False refusals | Correct refusals |",
        "|---|---|---|---|---|",
    ]
    for mode, r in reports.items():
        lines.append(
            f"| {mode} | {r.hit_rate:.0%} | {r.mrr:.2f} | {r.false_refusals:.0%} "
            f"| {r.correct_refusals:.0%} |"
        )
    return "\n".join(lines)
