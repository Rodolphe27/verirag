import numpy as np

from verirag.retrieval.keyword import Bm25Index, tokenize

CORPUS = [
    "providers of high-risk systems must keep technical documentation",
    "deployers must monitor the operation of the system",
    "general purpose models need a summary of the training content",
    "the authority supervises the market and the providers",
]


def test_tokenize_lowercases_and_cuts_to_six_letters() -> None:
    assert tokenize("Providers SYSTEMS a") == ["provid", "system"]


def test_word_forms_meet() -> None:
    assert tokenize("system") == tokenize("systems")


def test_chunk_with_the_rare_word_ranks_first() -> None:
    scores = Bm25Index(CORPUS).scores("training")
    assert int(np.argmax(scores)) == 2


def test_chunks_without_any_query_word_score_zero() -> None:
    scores = Bm25Index(CORPUS).scores("training")
    assert scores[0] == scores[1] == scores[3] == 0.0


def test_a_word_in_most_chunks_adds_nothing() -> None:
    corpus = ["the cat", "the dog", "the bird"]
    assert not Bm25Index(corpus).scores("the").any()


def test_unknown_words_give_all_zero() -> None:
    assert not Bm25Index(CORPUS).scores("blockchain").any()


def test_empty_corpus_is_allowed() -> None:
    assert Bm25Index([]).scores("anything").shape == (0,)
