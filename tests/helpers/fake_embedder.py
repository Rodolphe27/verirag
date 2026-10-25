"""A tiny embedder for tests: one dimension per topic word, no model download."""

from __future__ import annotations

import re

import numpy as np
from numpy.typing import NDArray

from verirag.retrieval.embedding import Kind, normalise


class TopicEmbedder:
    """Counts how many words of each topic appear in the text.

    Texts about the same topic get similar vectors, which is all the search
    tests need. Each topic is a set of words. A text with no topic word gets the
    zero vector: the "model" knows nothing about it, like an unseen article number.
    """

    def __init__(self, topics: list[set[str]]) -> None:
        self._topics = topics

    def embed(self, texts: list[str], kind: Kind = "passage") -> NDArray[np.float32]:
        rows = []
        for text in texts:
            words = set(re.findall(r"\w+", text.lower()))
            rows.append([len(words & topic) for topic in self._topics])
        return normalise(np.array(rows, dtype=np.float32))
