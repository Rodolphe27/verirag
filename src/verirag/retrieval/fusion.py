from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

RRF_K = 60


def reciprocal_rank_fusion(*scores: NDArray[np.float64]) -> NDArray[np.float64]:
    """Combine several score lists into one by their RANKS (Reciprocal Rank Fusion).

    Each list gives every item the value 1 / (60 + rank), where rank 0 is the
    best item. The values are added up. Using ranks instead of raw scores
    avoids comparing numbers on different scales (cosine is 0 to 1, BM25 can
    be 20). An item with a score of 0 is treated as "not found" by that list
    and gets nothing from it.
    """
    total = np.zeros(len(scores[0]), dtype=np.float64)
    for values in scores:
        ranks = np.argsort(np.argsort(-values))
        total += np.where(values > 0, 1.0 / (RRF_K + ranks), 0.0)
    return total
