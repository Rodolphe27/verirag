from __future__ import annotations

from typing import Literal, Protocol

import numpy as np
from numpy.typing import NDArray

Kind = Literal["passage", "query"]
DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


class Embedder(Protocol):
    """Turns texts into vectors. Texts with a similar meaning get similar vectors."""

    def embed(self, texts: list[str], kind: Kind = "passage") -> NDArray[np.float32]:
        """One unit-length row per text. `kind` tells the model if the text is a
        document ("passage") or a question ("query"); some models need that."""
        ...


class FastEmbedder:
    """Local embeddings with the fastembed library: free, offline after the first download.

    The default model understands German and English, which matters for the
    documents this project is meant for. The model loads on first use, so
    importing this module stays fast and tests that never embed do not pay for it.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        self._model_name = model_name
        self._model: object | None = None

    def embed(self, texts: list[str], kind: Kind = "passage") -> NDArray[np.float32]:
        from fastembed import TextEmbedding

        if self._model is None:
            self._model = TextEmbedding(self._model_name)
        assert isinstance(self._model, TextEmbedding)
        if "e5" in self._model_name:  # E5 models were trained with these prefixes
            texts = [f"{kind}: {text}" for text in texts]
        vectors = np.array(list(self._model.embed(texts)), dtype=np.float32)
        return normalise(vectors)


def normalise(vectors: NDArray[np.float32]) -> NDArray[np.float32]:
    """Scale every row to length 1, so a dot product equals the cosine similarity."""
    lengths = np.linalg.norm(vectors, axis=1, keepdims=True)
    result: NDArray[np.float32] = vectors / np.clip(lengths, 1e-12, None)
    return result
