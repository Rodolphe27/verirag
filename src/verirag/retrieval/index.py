from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from verirag.domain.chunk import Chunk
from verirag.retrieval.documents import find_documents, load_chunks
from verirag.retrieval.embedding import DEFAULT_MODEL, Embedder
from verirag.retrieval.hybrid import HybridRetriever

FORMAT_VERSION = 1


@dataclass(frozen=True)
class BuildReport:
    """What a build did, file by file."""

    embedded: tuple[str, ...]
    reused: tuple[str, ...]
    removed: tuple[str, ...]


def build_index(
    folder: Path,
    index_path: Path,
    embedder: Embedder,
    model_name: str = DEFAULT_MODEL,
    size: int = 450,
    overlap: int = 150,
) -> BuildReport:
    """Embed the documents of `folder` and save them to `index_path` (a JSON file).

    Embedding is the slow part, so a file whose content did not change since
    the last build is copied from the old index instead of being embedded again.
    If the model or the chunk settings changed, everything is rebuilt, because
    old vectors would not fit the new settings.
    """
    settings = {"model": model_name, "size": size, "overlap": overlap}
    old_files = _read_files(index_path, settings)

    new_files: dict[str, Any] = {}
    embedded: list[str] = []
    reused: list[str] = []
    for path in find_documents(folder):
        key = path.relative_to(folder).as_posix()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if key in old_files and old_files[key]["hash"] == digest:
            new_files[key] = old_files[key]
            reused.append(key)
            continue
        chunks = load_chunks(path, size, overlap)
        vectors = embedder.embed([c.text for c in chunks]) if chunks else np.zeros((0, 0))
        new_files[key] = {
            "hash": digest,
            "chunks": [
                {"text": c.text, "source": c.source, "page": c.page, "embedding": _round(v)}
                for c, v in zip(chunks, vectors, strict=True)
            ],
        }
        embedded.append(key)

    removed = sorted(set(old_files) - set(new_files))
    document = {"version": FORMAT_VERSION, "settings": settings, "files": new_files}
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")
    return BuildReport(tuple(embedded), tuple(reused), tuple(removed))


def load_retriever(index_path: Path, embedder: Embedder, hybrid: bool = True) -> HybridRetriever:
    """Read an index file and return a retriever that searches it."""
    if not index_path.exists():
        raise FileNotFoundError(
            f"no index at {index_path}; build one with `verirag index <folder>`"
        )
    document = json.loads(index_path.read_text(encoding="utf-8"))
    entries = [e for name in sorted(document["files"]) for e in document["files"][name]["chunks"]]
    chunks = [Chunk(e["text"], e["source"], e["page"]) for e in entries]
    vectors = np.array([e["embedding"] for e in entries], dtype=np.float32).reshape(len(chunks), -1)
    return HybridRetriever(chunks, vectors, embedder, hybrid=hybrid)


def list_sources(index_path: Path) -> list[str]:
    """Names of the files stored in an index, sorted."""
    if not index_path.exists():
        raise FileNotFoundError(
            f"no index at {index_path}; build one with `verirag index <folder>`"
        )
    document = json.loads(index_path.read_text(encoding="utf-8"))
    return sorted(document["files"])


def _read_files(index_path: Path, settings: dict[str, object]) -> dict[str, Any]:
    """Files of an existing index, or {} if there is none or its settings differ."""
    if not index_path.exists():
        return {}
    document = json.loads(index_path.read_text(encoding="utf-8"))
    if document.get("version") != FORMAT_VERSION or document.get("settings") != settings:
        return {}
    files: dict[str, Any] = document["files"]
    return files


def _round(vector: np.ndarray[Any, Any]) -> list[float]:
    """Six decimals are plenty for cosine search and keep the file small."""
    return [round(float(x), 6) for x in vector]
