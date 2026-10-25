from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader

from verirag.domain.chunk import Chunk
from verirag.retrieval.chunking import split_text

TEXT_SUFFIXES = {".md", ".txt"}


def load_chunks(path: Path, size: int = 450, overlap: int = 150) -> list[Chunk]:
    """Read one file and return its chunks.

    Text and Markdown files have no pages. PDF chunks keep the page number
    they came from, so an answer can cite "file.pdf, p. 3".
    """
    suffix = path.suffix.lower()
    if suffix in TEXT_SUFFIXES:
        pages: list[tuple[int | None, str]] = [(None, path.read_text(encoding="utf-8"))]
    elif suffix == ".pdf":
        reader = PdfReader(path)
        pages = [(number, page.extract_text() or "") for number, page in enumerate(reader.pages, 1)]
    else:
        raise ValueError(f"unsupported file type: {path.suffix}")

    return [
        Chunk(text=piece, source=path.name, page=number)
        for number, text in pages
        for piece in split_text(text, size, overlap)
    ]


def find_documents(folder: Path) -> list[Path]:
    """Supported files in `folder` (sub-folders included), sorted by path."""
    supported = TEXT_SUFFIXES | {".pdf"}
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in supported)


def load_folder(folder: Path, size: int = 450, overlap: int = 150) -> list[Chunk]:
    """Chunks of every supported file in `folder`, in path order."""
    return [chunk for path in find_documents(folder) for chunk in load_chunks(path, size, overlap)]
