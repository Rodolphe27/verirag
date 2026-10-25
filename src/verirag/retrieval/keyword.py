from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np
from numpy.typing import NDArray

_WORD = re.compile(r"\w{2,}")
_STEM_LENGTH = 6


def tokenize(text: str) -> list[str]:
    """Lowercase words, each cut to 6 letters.

    The cut is a very simple "stemmer": it lets word forms meet ("system" and
    "systems", "provider" and "providers"). It also works for German, where a
    real stemmer would need extra libraries. The price is some false matches
    between long words that share their first six letters.
    """
    return [word[:_STEM_LENGTH] for word in _WORD.findall(text.lower())]


class Bm25Index:
    """BM25 keyword scoring over a fixed list of texts.

    BM25 is the classic formula behind search engines. A chunk scores higher
    when it contains the query words often (term frequency) and when those
    words are rare in the whole corpus (inverse document frequency).
    Statistics are computed once here, then every query is cheap.
    """

    def __init__(self, texts: list[str], k1: float = 1.5, b: float = 0.75) -> None:
        self._k1 = k1
        self._b = b
        self._counts = [Counter(tokenize(text)) for text in texts]
        self._lengths = np.array([sum(c.values()) for c in self._counts], dtype=np.float64)
        self._average = float(self._lengths.mean()) if len(texts) else 0.0
        self._document_frequency: Counter[str] = Counter()
        for counts in self._counts:
            self._document_frequency.update(counts.keys())

    def scores(self, query: str) -> NDArray[np.float64]:
        """One score per text, in the original order. 0 means no useful word in common."""
        n = len(self._counts)
        result = np.zeros(n, dtype=np.float64)
        for term in set(tokenize(query)):
            df = self._document_frequency.get(term, 0)
            if df == 0:
                continue
            # A word found in more than half of the texts says little: weight 0, never negative.
            idf = max(0.0, math.log((n - df + 0.5) / (df + 0.5)))
            for i, counts in enumerate(self._counts):
                tf = counts.get(term, 0)
                if tf:
                    norm = tf + self._k1 * (
                        1 - self._b + self._b * self._lengths[i] / self._average
                    )
                    result[i] += idf * tf * (self._k1 + 1) / norm
        return result
