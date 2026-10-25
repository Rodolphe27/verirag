import numpy as np

from verirag.retrieval.fusion import reciprocal_rank_fusion


def test_an_item_good_in_both_lists_wins() -> None:
    meaning = np.array([0.9, 0.8, 0.1])
    keyword = np.array([0.0, 5.0, 4.0])
    assert int(np.argmax(reciprocal_rank_fusion(meaning, keyword))) == 1


def test_scale_of_the_scores_does_not_matter() -> None:
    meaning = np.array([0.9, 0.8, 0.7])
    keyword = np.array([20.0, 10.0, 5.0])
    scaled = keyword / 1000
    assert np.allclose(
        reciprocal_rank_fusion(meaning, keyword), reciprocal_rank_fusion(meaning, scaled)
    )


def test_a_zero_score_adds_nothing() -> None:
    result = reciprocal_rank_fusion(np.array([1.0, 0.5]), np.array([0.0, 0.0]))
    assert result[0] > result[1] > 0
    assert result[0] == 1 / 60


def test_one_list_alone_keeps_its_order() -> None:
    result = reciprocal_rank_fusion(np.array([0.2, 0.9, 0.5]))
    assert list(np.argsort(-result)) == [1, 2, 0]
