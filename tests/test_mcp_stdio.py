"""End-to-end: a real MCP client talks to `verirag serve` running as a separate process."""

import sys
from pathlib import Path

import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from tests.helpers.fake_embedder import TopicEmbedder
from verirag.retrieval.index import build_index


@pytest.mark.anyio
async def test_a_client_can_list_the_tools_and_call_one(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "article-99.md").write_text("Fines can reach 35 million euros.", encoding="utf-8")
    index = tmp_path / "index.json"
    build_index(docs, index, TopicEmbedder([{"fines"}]))

    params = StdioServerParameters(
        command=sys.executable, args=["-m", "verirag", "--index", str(index), "serve"]
    )
    async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
        await session.initialize()
        tools = await session.list_tools()
        assert {tool.name for tool in tools.tools} == {"list_documents", "search_documents"}
        result = await session.call_tool("list_documents", {})
        assert result.structured_content == {"result": ["article-99.md"]}
