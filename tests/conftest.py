import pytest

from tests.helpers.fake_llm import ScriptedLLM


@pytest.fixture
def make_llm() -> type[ScriptedLLM]:
    """Fixture form of the fake: `make_llm("reply 1", "reply 2")` builds a ScriptedLLM."""
    return ScriptedLLM
