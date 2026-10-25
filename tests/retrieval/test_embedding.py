import numpy as np

from verirag.retrieval.embedding import normalise


def test_normalised_rows_have_length_one() -> None:
    rows = normalise(np.array([[3.0, 4.0], [0.0, 2.0]], dtype=np.float32))
    assert np.allclose(np.linalg.norm(rows, axis=1), 1.0)


def test_a_zero_vector_stays_zero_instead_of_dividing_by_zero() -> None:
    rows = normalise(np.zeros((1, 3), dtype=np.float32))
    assert not rows.any()
