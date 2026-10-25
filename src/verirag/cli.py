"""The command line: `verirag index <folder>`, `verirag ask "<question>"`, `verirag serve`.

This module is the composition root: the one place that creates the real
objects (embedder, model, tracers) and connects them. Everything else only
receives what it needs. `main` accepts replacements for the embedder and the
model, which is how the tests run the whole program without a network.

Exit codes: 0 = verified answer, 2 = refused (a normal outcome, not an error),
1 = error (missing index, missing key, network problem).
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import MutableMapping, Sequence
from pathlib import Path
from typing import TextIO

from verirag.agents.router import DEFAULT_MIN_SCORE
from verirag.assistant import Assistant
from verirag.config import load_env_file
from verirag.domain.answer import Answer, AnswerStatus
from verirag.llm.factory import create_llm
from verirag.mcp_server import create_server
from verirag.ports import LLM, LLMError, Tracer
from verirag.retrieval.embedding import DEFAULT_MODEL, Embedder, FastEmbedder
from verirag.retrieval.index import build_index, list_sources, load_retriever
from verirag.tracing import CompositeTracer, JsonlTracer, MemoryTracer

DEFAULT_INDEX = ".verirag/index.json"


def main(
    argv: Sequence[str] | None = None,
    *,
    environ: MutableMapping[str, str] = os.environ,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
    embedder: Embedder | None = None,
    llm: LLM | None = None,
) -> int:
    out, err = stdout or sys.stdout, stderr or sys.stderr
    if environ is os.environ:
        load_env_file(Path(".env"), environ)
    args = _parser().parse_args(argv)
    index_path = Path(args.index or environ.get("VERIRAG_INDEX", DEFAULT_INDEX))
    embedder = embedder or FastEmbedder()
    try:
        if args.command == "index":
            return _index(args, index_path, embedder, out)
        if args.command == "serve":
            return _serve(args, index_path, embedder, llm, environ)
        return _ask(args, index_path, embedder, llm, environ, out, err)
    except (LLMError, FileNotFoundError, NotADirectoryError) as error:
        print(f"error: {error}", file=err)
        return 1


def _index(args: argparse.Namespace, index_path: Path, embedder: Embedder, out: TextIO) -> int:
    folder = Path(args.folder)
    if not folder.is_dir():
        raise NotADirectoryError(f"{folder} is not a folder")
    report = build_index(folder, index_path, embedder, DEFAULT_MODEL)
    print(
        f"Index saved to {index_path}: {len(report.embedded)} file(s) embedded, "
        f"{len(report.reused)} reused, {len(report.removed)} removed.",
        file=out,
    )
    return 0


def _serve(
    args: argparse.Namespace,
    index_path: Path,
    embedder: Embedder,
    llm: LLM | None,
    environ: MutableMapping[str, str],
) -> int:
    """Run the MCP server on standard input/output. Nothing may be printed here:
    stdout carries the protocol messages."""
    retriever = load_retriever(index_path, embedder)
    assistant = None
    if args.verified:
        assistant = Assistant(retriever, llm or create_llm(environ), MemoryTracer())
    create_server(retriever, list_sources(index_path), assistant).run("stdio")
    return 0


def _ask(
    args: argparse.Namespace,
    index_path: Path,
    embedder: Embedder,
    llm: LLM | None,
    environ: MutableMapping[str, str],
    out: TextIO,
    err: TextIO,
) -> int:
    memory = MemoryTracer()
    tracers: list[Tracer] = [memory]
    if args.trace_file:
        tracers.append(JsonlTracer(Path(args.trace_file)))
    assistant = Assistant(
        load_retriever(index_path, embedder),
        llm or create_llm(environ),
        CompositeTracer(*tracers),
        min_score=args.min_score,
    )
    answer = assistant.ask(args.question)
    if args.trace:
        for step, data in memory.records:
            print(f"[{step}] {data}", file=err)
    print(format_answer(answer), file=out)
    return 0 if answer.status is AnswerStatus.ANSWERED else 2


def format_answer(answer: Answer) -> str:
    """Plain text for a terminal: the verdict first, then the text, then the sources."""
    if answer.status is AnswerStatus.ANSWERED:
        lines = ["Verified answer:", answer.text]
    else:
        lines = ["No verified answer.", answer.text]
    if answer.citations:
        lines += ["", "Sources:", *(f"  - {citation}" for citation in answer.citations)]
    return "\n".join(lines)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="verirag", description="Answers questions from documents and verifies every claim."
    )
    parser.add_argument("--index", help=f"index file (default: {DEFAULT_INDEX})")
    commands = parser.add_subparsers(dest="command", required=True)

    index = commands.add_parser("index", help="read the documents of a folder into the index")
    index.add_argument("folder", help="folder with .md, .txt and .pdf files")

    serve = commands.add_parser("serve", help="run an MCP server for AI applications")
    serve.add_argument(
        "--verified",
        action="store_true",
        help="also offer the ask_verified tool (needs a model and its API key)",
    )

    ask = commands.add_parser("ask", help="ask a question")
    ask.add_argument("question")
    ask.add_argument("--trace", action="store_true", help="print every agent step")
    ask.add_argument("--trace-file", help="also save the steps to this JSON-lines file")
    ask.add_argument(
        "--min-score",
        type=float,
        default=DEFAULT_MIN_SCORE,
        help=f"refuse when no source scores at least this (default: {DEFAULT_MIN_SCORE})",
    )
    return parser
