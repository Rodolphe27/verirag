from __future__ import annotations

import re

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+|\n{2,}")


def split_text(text: str, size: int = 450, overlap: int = 150) -> list[str]:
    """Cut `text` into chunks of at most `size` characters.

    Cuts fall on sentence or paragraph borders when possible. Each new chunk
    starts with the last sentences of the previous one (up to `overlap`
    characters), so an idea that crosses a border is still found by the search.
    """
    if size < 1:
        raise ValueError("size must be positive")
    if not 0 <= overlap < size:
        raise ValueError("overlap must be between 0 and size - 1")

    chunks: list[str] = []
    current: list[str] = []
    for piece in _pieces(text, size):
        if current and len(_join([*current, piece])) > size:
            chunks.append(_join(current))
            current = _tail(current, overlap)
            if current and len(_join([*current, piece])) > size:
                current = []  # the overlap would push the chunk over the limit
        current.append(piece)
    if current:
        chunks.append(_join(current))
    return chunks


def _pieces(text: str, size: int) -> list[str]:
    """Sentences and paragraphs. A piece longer than `size` is cut hard."""
    pieces: list[str] = []
    for part in _SENTENCE_END.split(text):
        part = part.strip()
        while len(part) > size:
            pieces.append(part[:size])
            part = part[size:]
        if part:
            pieces.append(part)
    return pieces


def _tail(pieces: list[str], overlap: int) -> list[str]:
    """The last pieces whose joined length stays within `overlap` characters."""
    tail: list[str] = []
    for piece in reversed(pieces):
        if len(_join([piece, *tail])) > overlap:
            break
        tail.insert(0, piece)
    return tail


def _join(pieces: list[str]) -> str:
    return " ".join(pieces)
