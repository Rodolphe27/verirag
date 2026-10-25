import io
import json
from pathlib import Path

import pytest
from mcp.server.mcpserver import MCPServer

from tests.helpers.fake_embedder import TopicEmbedder
from tests.helpers.fake_llm import ScriptedLLM
from verirag.cli import main

TOPICS = [{"fines", "euros", "penalties"}, {"logs", "records"}]
DRAFT = {
    "answer": "Up to 35 million euros.",
    "claims": [{"text": "Up to 35 million euros.", "sources": [1]}],
}
VERDICT = {
    "verdicts": [{"claim": 1, "supported": True, "evidence": "fines can reach 35 million euros"}]
}


class Result:
    def __init__(self, code: int, out: str, err: str) -> None:
        self.code, self.out, self.err = code, out, err


def run(*argv: str, llm: ScriptedLLM | None = None, env: dict[str, str] | None = None) -> Result:
    out, err = io.StringIO(), io.StringIO()
    code = main(
        list(argv),
        environ=env if env is not None else {},
        stdout=out,
        stderr=err,
        embedder=TopicEmbedder(TOPICS),
        llm=llm,
    )
    return Result(code, out.getvalue(), err.getvalue())


@pytest.fixture
def index(tmp_path: Path) -> Path:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "article-99.md").write_text("Fines can reach 35 million euros.", encoding="utf-8")
    (docs / "article-12.md").write_text("Providers must keep logs and records.", encoding="utf-8")
    path = tmp_path / "index.json"
    assert run("--index", str(path), "index", str(docs)).code == 0
    return path


def test_index_reports_what_it_did(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.md").write_text("Some text about fines.", encoding="utf-8")
    result = run("--index", str(tmp_path / "i.json"), "index", str(docs))
    assert result.code == 0
    assert "1 file(s) embedded, 0 reused, 0 removed" in result.out


def test_index_of_a_missing_folder_is_an_error(tmp_path: Path) -> None:
    result = run("--index", str(tmp_path / "i.json"), "index", str(tmp_path / "nope"))
    assert result.code == 1
    assert "not a folder" in result.err


def test_a_verified_answer_exits_with_zero_and_lists_its_sources(index: Path) -> None:
    llm = ScriptedLLM(json.dumps(DRAFT), json.dumps(VERDICT))
    result = run("--index", str(index), "ask", "How high are fines?", llm=llm)
    assert result.code == 0
    assert "Verified answer:" in result.out
    assert "Up to 35 million euros." in result.out
    assert "  - article-99.md" in result.out


def test_a_question_the_documents_cannot_answer_is_refused_with_exit_code_two(index: Path) -> None:
    result = run("--index", str(index), "ask", "What is the weather?", llm=ScriptedLLM())
    assert result.code == 2
    assert "No verified answer." in result.out


def test_asking_without_an_index_explains_how_to_build_one(tmp_path: Path) -> None:
    result = run("--index", str(tmp_path / "none.json"), "ask", "q?", llm=ScriptedLLM())
    assert result.code == 1
    assert "verirag index" in result.err


def test_asking_without_an_api_key_names_the_missing_variable(index: Path) -> None:
    result = run("--index", str(index), "ask", "How high are fines?")
    assert result.code == 1
    assert "GEMINI_API_KEY" in result.err


def test_trace_prints_every_step_to_the_error_stream(index: Path) -> None:
    llm = ScriptedLLM(json.dumps(DRAFT), json.dumps(VERDICT))
    result = run("--index", str(index), "ask", "How high are fines?", "--trace", llm=llm)
    steps = [line.split("]")[0] for line in result.err.splitlines()]
    assert steps == ["[retriever", "[router", "[writer", "[verifier", "[finish"]
    assert "[retriever" not in result.out


def test_trace_file_saves_the_steps_as_json_lines(index: Path, tmp_path: Path) -> None:
    llm = ScriptedLLM(json.dumps(DRAFT), json.dumps(VERDICT))
    trace = tmp_path / "logs" / "run.jsonl"
    run("--index", str(index), "ask", "How high are fines?", "--trace-file", str(trace), llm=llm)
    steps = [json.loads(line)["step"] for line in trace.read_text(encoding="utf-8").splitlines()]
    assert steps[0] == "retriever" and steps[-1] == "finish"


def test_the_index_location_can_come_from_the_environment(index: Path) -> None:
    llm = ScriptedLLM(json.dumps(DRAFT), json.dumps(VERDICT))
    result = run("ask", "How high are fines?", llm=llm, env={"VERIRAG_INDEX": str(index)})
    assert result.code == 0


def test_serve_starts_the_mcp_server_on_stdio(index: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    started: list[str] = []
    monkeypatch.setattr(
        MCPServer, "run", lambda self, transport="stdio", **kw: started.append(transport)
    )
    result = run("--index", str(index), "serve")
    assert result.code == 0
    assert started == ["stdio"]
    assert result.out == ""  # stdout belongs to the protocol


def test_serve_without_an_index_explains_how_to_build_one(tmp_path: Path) -> None:
    result = run("--index", str(tmp_path / "none.json"), "serve")
    assert result.code == 1
    assert "verirag index" in result.err


def test_serve_verified_needs_an_api_key(index: Path) -> None:
    result = run("--index", str(index), "serve", "--verified")
    assert result.code == 1
    assert "GEMINI_API_KEY" in result.err


def test_evaluate_prints_one_row_per_search_mode(index: Path, tmp_path: Path) -> None:
    questions = tmp_path / "q.json"
    questions.write_text(
        json.dumps(
            {
                "answerable": [{"question": "How high are fines?", "sources": ["article-99.md"]}],
                "unanswerable": [{"question": "What is the weather?"}],
            }
        ),
        encoding="utf-8",
    )
    result = run("--index", str(index), "evaluate", "--questions", str(questions))
    assert result.code == 0
    rows = [line.split("|")[1].strip() for line in result.out.splitlines()[2:5]]
    assert rows == ["dense", "keyword", "hybrid"]
    assert "1 answerable and 1 unanswerable" in result.out
