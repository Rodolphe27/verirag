import pytest

from tests.helpers.fake_llm import ScriptedLLM
from verirag.ports import LLM


def test_replies_come_back_in_order() -> None:
    llm = ScriptedLLM("one", "two")
    assert [llm.complete("a"), llm.complete("b")] == ["one", "two"]


def test_prompts_are_remembered() -> None:
    llm = ScriptedLLM("x")
    llm.complete("what is this?")
    assert llm.prompts == ["what is this?"]


def test_a_reply_can_depend_on_the_prompt() -> None:
    llm = ScriptedLLM(lambda prompt: prompt.upper())
    assert llm.complete("abc") == "ABC"


def test_an_unexpected_extra_call_fails_loudly() -> None:
    with pytest.raises(AssertionError, match="extra model call"):
        ScriptedLLM().complete("anything")


def test_it_fits_the_llm_port() -> None:
    llm: LLM = ScriptedLLM("ok")
    assert llm.complete("hi") == "ok"


def test_the_fixture_builds_the_fake(make_llm: type[ScriptedLLM]) -> None:
    assert make_llm("hello").complete("x") == "hello"
