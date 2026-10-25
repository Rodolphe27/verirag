from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Case:
    """A question and the files that correctly answer it (empty if it has no answer)."""

    question: str
    sources: tuple[str, ...] = ()


@dataclass(frozen=True)
class CaseSet:
    answerable: tuple[Case, ...]
    unanswerable: tuple[Case, ...]


def load_cases(path: Path) -> CaseSet:
    """Read a question file (see eval/questions.json for the format)."""
    data = json.loads(path.read_text(encoding="utf-8"))
    answerable = tuple(Case(c["question"], tuple(c["sources"])) for c in data["answerable"])
    unanswerable = tuple(Case(c["question"]) for c in data["unanswerable"])
    if any(not case.sources for case in answerable):
        raise ValueError("every answerable question needs at least one source")
    return CaseSet(answerable, unanswerable)
