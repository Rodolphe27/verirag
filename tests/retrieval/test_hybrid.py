import numpy as np
import pytest

from tests.helpers.fake_embedder import TopicEmbedder
from verirag.domain.chunk import Chunk
from verirag.ports import Retriever
from verirag.retrieval.hybrid import HybridRetriever, Mode

TOPICS = [{"risk", "danger", "harm"}, {"data", "dataset", "training"}]
TEXTS = [
    "risk of harm to health",
    "danger for fundamental rights",
    "training dataset quality rules",
    "article 99 sets fines for infringements",
]


def build(mode: Mode = "hybrid") -> HybridRetriever:
    embedder = TopicEmbedder(TOPICS)
    chunks = [Chunk(text, f"file{i}.md") for i, text in enumerate(TEXTS)]
    return HybridRetriever(chunks, embedder.embed(TEXTS), embedder, mode=mode)


def test_it_fits_the_retriever_port() -> None:
    retriever: Retriever = build()
    assert retriever.search("risk")


def test_meaning_finds_the_right_topic() -> None:
    top = build().search("danger harm", k=2)
    assert {c.source for c in top} == {"file0.md", "file1.md"}


def test_score_is_the_cosine_similarity() -> None:
    top = build().search("training dataset", k=1)[0]
    assert top.source == "file2.md"
    assert 0.9 < top.score <= 1.0


def test_keyword_search_rescues_an_exact_term_the_meaning_misses() -> None:
    question = "article 99 fines"
    assert build("dense").search(question, k=1)[0].source != "file3.md"
    assert build("hybrid").search(question, k=1)[0].source == "file3.md"


def test_k_limits_the_result() -> None:
    assert len(build().search("risk", k=3)) == 3


def test_empty_index_returns_nothing() -> None:
    embedder = TopicEmbedder(TOPICS)
    assert HybridRetriever([], np.zeros((0, 2), dtype=np.float32), embedder).search("x") == []


def test_chunks_and_vectors_must_match() -> None:
    embedder = TopicEmbedder(TOPICS)
    with pytest.raises(ValueError, match="one vector"):
        HybridRetriever([Chunk("a", "a.md")], np.zeros((2, 2), dtype=np.float32), embedder)


def test_keyword_mode_ranks_by_shared_words_only() -> None:
    assert build("keyword").search("article 99 fines", k=1)[0].source == "file3.md"


def test_keyword_mode_still_reports_the_cosine_as_score() -> None:
    top = build("keyword").search("article 99 fines", k=1)[0]
    assert top.score == 0.0  # the fake embedder knows nothing about these words
