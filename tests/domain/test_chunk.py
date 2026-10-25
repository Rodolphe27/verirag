import pytest

from verirag.domain.chunk import Chunk


def test_citation_with_page() -> None:
    assert Chunk("text", "act.md", page=3).citation == "act.md, p. 3"


def test_citation_without_page() -> None:
    assert Chunk("text", "act.md").citation == "act.md"


@pytest.mark.parametrize("text", ["", "   "])
def test_empty_text_is_rejected(text: str) -> None:
    with pytest.raises(ValueError, match="text"):
        Chunk(text, "act.md")


def test_page_zero_is_rejected() -> None:
    with pytest.raises(ValueError, match="pages start at 1"):
        Chunk("text", "act.md", page=0)


def test_chunk_is_read_only() -> None:
    chunk = Chunk("text", "act.md")
    with pytest.raises(AttributeError):
        chunk.text = "other"  # type: ignore[misc]
