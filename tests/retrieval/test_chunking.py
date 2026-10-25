import pytest

from verirag.retrieval.chunking import split_text

TEXT = (
    "The first sentence is here. The second sentence follows it. "
    "A third one arrives now. Then a fourth sentence appears. "
    "The fifth closes the paragraph."
)


def test_short_text_stays_in_one_chunk() -> None:
    assert split_text("One short sentence.") == ["One short sentence."]


def test_empty_text_gives_no_chunks() -> None:
    assert split_text("  \n\n ") == []


def test_cuts_fall_on_sentence_borders() -> None:
    chunks = split_text(TEXT, size=70, overlap=0)
    assert len(chunks) > 1
    assert all(chunk.endswith(".") for chunk in chunks)


def test_no_chunk_is_longer_than_the_limit() -> None:
    for chunk in split_text(TEXT * 20, size=120, overlap=60):
        assert len(chunk) <= 120


def test_a_very_long_sentence_is_cut_hard() -> None:
    chunks = split_text("x" * 250, size=100, overlap=0)
    assert [len(c) for c in chunks] == [100, 100, 50]


def test_overlap_repeats_the_last_sentence() -> None:
    chunks = split_text(TEXT, size=100, overlap=60)
    assert chunks[0].split(". ")[-1].rstrip(".") in chunks[1]


def test_no_overlap_means_no_repetition() -> None:
    chunks = split_text(TEXT, size=70, overlap=0)
    assert " ".join(chunks) == TEXT


@pytest.mark.parametrize(("size", "overlap"), [(0, 0), (10, 10), (10, -1)])
def test_bad_settings_are_rejected(size: int, overlap: int) -> None:
    with pytest.raises(ValueError, match="must"):
        split_text("text", size=size, overlap=overlap)
