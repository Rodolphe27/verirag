"""An MCP server that exposes VeriRAG to any MCP client (a desktop AI app, an IDE, an agent).

MCP, the Model Context Protocol, is the standard way for an AI application to
discover and call tools. This module is an adapter: it holds no search or
verification logic, it only translates between MCP tool calls and the
Retriever port and the Assistant.

Tools:
- list_documents: which source files are indexed;
- search_documents: the best matching passages for a question (no model needed);
- ask_verified: a full answer whose claims were checked against the sources
  (only offered when an Assistant is given, because it needs a model).
"""

from __future__ import annotations

from typing import Any

from mcp.server.mcpserver import MCPServer

from verirag.assistant import Assistant
from verirag.ports import Retriever

MAX_RESULTS = 10


def create_server(
    retriever: Retriever, sources: list[str], assistant: Assistant | None = None
) -> MCPServer:
    server = MCPServer(
        "verirag",
        instructions="Search a private document collection and get answers checked against it.",
    )

    @server.tool()
    def list_documents() -> list[str]:
        """List the source files that can be searched."""
        return sorted(sources)

    @server.tool()
    def search_documents(question: str, limit: int = 4) -> list[dict[str, Any]]:
        """Find the passages of the documents that best match a question.

        Each result has the source file, the page (null for text files), a
        relevance score between 0 and 1, and the text.
        """
        chunks = retriever.search(question, max(1, min(limit, MAX_RESULTS)))
        return [
            {"source": c.source, "page": c.page, "score": round(c.score, 3), "text": c.text}
            for c in chunks
        ]

    if assistant is not None:

        @server.tool()
        def ask_verified(question: str) -> dict[str, Any]:
            """Answer a question from the documents; every claim is checked first.

            `status` is "answered" (with citations) or "refused" (the documents
            do not support an answer). A refusal is a result, not an error.
            """
            answer = assistant.ask(question)
            return {
                "status": answer.status.value,
                "text": answer.text,
                "citations": list(answer.citations),
            }

    return server
