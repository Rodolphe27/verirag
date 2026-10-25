import pytest

from verirag.agents.output import BadModelOutput, extract_json


def test_plain_json_is_parsed() -> None:
    assert extract_json('{"a": 1}') == {"a": 1}


def test_a_markdown_fence_is_ignored() -> None:
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}


def test_words_around_the_json_are_ignored() -> None:
    assert extract_json('Here you go: {"a": {"b": 2}} Hope it helps!') == {"a": {"b": 2}}


@pytest.mark.parametrize("reply", ["no json here", "} backwards {", '{"a": 1', "{'a': 1}"])
def test_broken_replies_raise_a_clear_error(reply: str) -> None:
    with pytest.raises(BadModelOutput):
        extract_json(reply)
