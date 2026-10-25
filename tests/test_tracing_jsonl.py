import json
from pathlib import Path

from verirag.domain.answer import AnswerStatus
from verirag.tracing import JsonlTracer


def read_lines(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_each_record_is_one_valid_json_line(tmp_path: Path) -> None:
    path = tmp_path / "run.jsonl"
    tracer = JsonlTracer(path)
    tracer.record("router", {"route": "documents"})
    tracer.record("writer", {"attempt": 1})
    lines = read_lines(path)
    assert [line["step"] for line in lines] == ["router", "writer"]
    assert lines[0]["data"] == {"route": "documents"}
    assert "time" in lines[0]


def test_creates_missing_folders(tmp_path: Path) -> None:
    path = tmp_path / "deep" / "run.jsonl"
    JsonlTracer(path).record("x", {})
    assert path.exists()


def test_values_json_cannot_encode_are_stored_as_text(tmp_path: Path) -> None:
    path = tmp_path / "run.jsonl"
    JsonlTracer(path).record("x", {"status": AnswerStatus.REFUSED})
    assert read_lines(path)[0]["data"] == {"status": "AnswerStatus.REFUSED"}


def test_a_second_tracer_appends_instead_of_overwriting(tmp_path: Path) -> None:
    path = tmp_path / "run.jsonl"
    JsonlTracer(path).record("a", {})
    JsonlTracer(path).record("b", {})
    assert [line["step"] for line in read_lines(path)] == ["a", "b"]
