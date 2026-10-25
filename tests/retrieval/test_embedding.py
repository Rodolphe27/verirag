import os

import numpy as np
import pytest

from verirag.retrieval.embedding import FastEmbedder, normalise


def test_normalised_rows_have_length_one() -> None:
    rows = normalise(np.array([[3.0, 4.0], [0.0, 2.0]], dtype=np.float32))
    assert np.allclose(np.linalg.norm(rows, axis=1), 1.0)


def test_a_zero_vector_stays_zero_instead_of_dividing_by_zero() -> None:
    rows = normalise(np.zeros((1, 3), dtype=np.float32))
    assert not rows.any()


@pytest.mark.skipif(
    os.environ.get("VERIRAG_MODEL_TESTS") != "1",
    reason="downloads a 220 MB model; run with VERIRAG_MODEL_TESTS=1",
)
def test_the_real_model_puts_translations_close_and_unrelated_texts_far() -> None:
    embedder = FastEmbedder()
    english, german, unrelated = embedder.embed(
        ["high-risk AI system", "Hochrisiko-KI-System", "how to bake bread"]
    )
    assert float(english @ german) > 0.5
    assert float(english @ unrelated) < float(english @ german)
    assert len(english) == 384
    assert np.isclose(np.linalg.norm(english), 1.0, atol=1e-5)
