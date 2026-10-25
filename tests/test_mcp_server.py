import json
from typing import Any

import pytest

from tests.helpers.fake_llm import ScriptedLLM
from verirag.assistant import Assistant
from verirag.domain.chunk import Chunk
from verirag.mcp_server import create_server
from verirag.tracing import MemoryTracer


class FakeRetriever:
    def __init__(self) -> None:
        self.asked: list[tuple[str, int]] = []

    def search(self, question: str, k: int = 4) -> list[Chunk]:
        self.asked.append((question, k))
        return [
            Chunk("Fines can reach 35 million euros.", "article-99.md", page=None, score=0.8123)
        ]


async def call(server: Any, name: str, arguments: dict[str, Any]) -> Any:
    result = await server.call_tool(name, arguments)
    assert not result.is_error
    content = result.structured_content
    return content.get("result", content)


@pytest.mark.anyio
async def test_without_an_assistant_only_the_two_search_tools_exist() -> None:
    server = create_server(FakeRetriever(), ["b.md", "a.md"])
    assert {tool.name for tool in await server.list_tools()} == {
        "list_documents",
        "search_documents",
    }


@pytest.mark.anyio
async def test_list_documents_returns_sorted_sources() -> None:
    server = create_server(FakeRetriever(), ["b.md", "a.md"])
    assert await call(server, "list_documents", {}) == ["a.md", "b.md"]


@pytest.mark.anyio
async def test_search_documents_returns_source_page_score_and_text() -> None:
    server = create_server(FakeRetriever(), [])
    assert await call(server, "search_documents", {"question": "fines?"}) == [
        {
            "source": "article-99.md",
            "page": None,
            "score": 0.812,
            "text": "Fines can reach 35 million euros.",
        }
    ]


@pytest.mark.anyio
@pytest.mark.parametrize(("asked", "used"), [(3, 3), (0, 1), (500, 10)])
async def test_the_result_limit_is_kept_between_one_and_ten(asked: int, used: int) -> None:
    retriever = FakeRetriever()
    await call(create_server(retriever, []), "search_documents", {"question": "q", "limit": asked})
    assert retriever.asked == [("q", used)]


@pytest.mark.anyio
async def test_ask_verified_exists_only_with_an_assistant_and_returns_the_status() -> None:
    draft = {
        "answer": "Up to 35 million.",
        "claims": [{"text": "Up to 35 million.", "sources": [1]}],
    }
    verdict = {"verdicts": [{"claim": 1, "supported": True, "evidence": "35 million euros"}]}
    llm = ScriptedLLM(json.dumps(draft), json.dumps(verdict))
    assistant = Assistant(FakeRetriever(), llm, MemoryTracer())
    server = create_server(FakeRetriever(), [], assistant)
    assert "ask_verified" in {tool.name for tool in await server.list_tools()}
    assert await call(server, "ask_verified", {"question": "How high are fines?"}) == {
        "status": "answered",
        "text": "Up to 35 million.",
        "citations": ["article-99.md"],
    }
