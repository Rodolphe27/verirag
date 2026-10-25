from __future__ import annotations

from dataclasses import replace

import numpy as np
from numpy.typing import NDArray

from verirag.domain.chunk import Chunk
from verirag.retrieval.embedding import Embedder
from verirag.retrieval.fusion import reciprocal_rank_fusion
from verirag.retrieval.keyword import Bm25Index


class HybridRetriever:
    """Finds chunks by meaning (embeddings) and by shared words (BM25), then fuses both.

    It implements the `Retriever` port. The score on each returned chunk is
    the cosine similarity (0 to 1): it is easy to interpret, so the router can
    use it to decide that the sources are too weak to answer. The ORDER comes
    from the fused ranking.
    """

    def __init__(
        self,
        chunks: list[Chunk],
        vectors: NDArray[np.float32],
        embedder: Embedder,
        hybrid: bool = True,
    ) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("one vector is needed for every chunk")
        self._chunks = chunks
        self._vectors = vectors
        self._embedder = embedder
        self._hybrid = hybrid
        self._keyword = Bm25Index([chunk.text for chunk in chunks])

    def search(self, question: str, k: int = 4) -> list[Chunk]:
        if not self._chunks:
            return []
        query = self._embedder.embed([question], kind="query")[0]
        cosine = (self._vectors @ query).astype(np.float64)
        if self._hybrid:
            ranking = reciprocal_rank_fusion(cosine, self._keyword.scores(question))
        else:
            ranking = cosine
        best = np.argsort(-ranking, kind="stable")[:k]
        return [replace(self._chunks[i], score=float(cosine[i])) for i in best]
