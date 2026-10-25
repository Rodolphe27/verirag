from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from tests.helpers.fake_embedder import TopicEmbedder
from verirag.retrieval.embedding import Kind
from verirag.retrieval.index import build_index, list_sources, load_retriever

TOPICS = [{"risk", "harm"}, {"data", "training"}]


class CountingEmbedder(TopicEmbedder):
    """TopicEmbedder that remembers every text it was asked to embed."""

    def __init__(self) -> None:
        super().__init__(TOPICS)
        self.seen: list[str] = []

    def embed(self, texts: list[str], kind: Kind = "passage") -> NDArray[np.float32]:
        self.seen.extend(texts)
        return super().embed(texts, kind)


@pytest.fixture
def docs(tmp_path: Path) -> Path:
    folder = tmp_path / "docs"
    folder.mkdir()
    (folder / "risk.md").write_text("Risk and harm rules.", encoding="utf-8")
    (folder / "data.md").write_text("Training data rules.", encoding="utf-8")
    return folder


def test_first_build_embeds_every_file(docs: Path, tmp_path: Path) -> None:
    report = build_index(docs, tmp_path / "index.json", CountingEmbedder())
    assert report.embedded == ("data.md", "risk.md")
    assert report.reused == ()


def test_second_build_of_unchanged_files_embeds_nothing(docs: Path, tmp_path: Path) -> None:
    index = tmp_path / "index.json"
    build_index(docs, index, CountingEmbedder())
    embedder = CountingEmbedder()
    report = build_index(docs, index, embedder)
    assert report.embedded == ()
    assert report.reused == ("data.md", "risk.md")
    assert embedder.seen == []


def test_only_the_changed_file_is_embedded_again(docs: Path, tmp_path: Path) -> None:
    index = tmp_path / "index.json"
    build_index(docs, index, CountingEmbedder())
    (docs / "risk.md").write_text("Risk and harm rules, now longer.", encoding="utf-8")
    embedder = CountingEmbedder()
    report = build_index(docs, index, embedder)
    assert report.embedded == ("risk.md",)
    assert report.reused == ("data.md",)
    assert embedder.seen == ["Risk and harm rules, now longer."]


def test_a_deleted_file_leaves_the_index(docs: Path, tmp_path: Path) -> None:
    index = tmp_path / "index.json"
    build_index(docs, index, CountingEmbedder())
    (docs / "data.md").unlink()
    report = build_index(docs, index, CountingEmbedder())
    assert report.removed == ("data.md",)
    assert {c.source for c in load_retriever(index, CountingEmbedder()).search("rules", k=5)} == {
        "risk.md"
    }


def test_a_new_model_name_rebuilds_everything(docs: Path, tmp_path: Path) -> None:
    index = tmp_path / "index.json"
    build_index(docs, index, CountingEmbedder(), model_name="model-a")
    report = build_index(docs, index, CountingEmbedder(), model_name="model-b")
    assert report.embedded == ("data.md", "risk.md")


def test_the_loaded_retriever_finds_the_right_file(docs: Path, tmp_path: Path) -> None:
    index = tmp_path / "index.json"
    embedder = CountingEmbedder()
    build_index(docs, index, embedder)
    top = load_retriever(index, embedder).search("training data", k=1)[0]
    assert top.source == "data.md"


def test_loading_a_missing_index_explains_what_to_do(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="verirag index"):
        load_retriever(tmp_path / "none.json", CountingEmbedder())


def test_list_sources_names_the_indexed_files(docs: Path, tmp_path: Path) -> None:
    index = tmp_path / "index.json"
    build_index(docs, index, CountingEmbedder())
    assert list_sources(index) == ["data.md", "risk.md"]


def test_list_sources_of_a_missing_index_explains_what_to_do(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="verirag index"):
        list_sources(tmp_path / "none.json")
