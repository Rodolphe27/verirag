import pytest

from verirag.llm.anthropic import AnthropicLLM
from verirag.llm.factory import create_llm
from verirag.llm.gemini import GeminiLLM
from verirag.ports import LLMError

KEYS = {"GEMINI_API_KEY": "g", "ANTHROPIC_API_KEY": "a"}


def test_gemini_is_the_default() -> None:
    assert isinstance(create_llm(KEYS), GeminiLLM)


def test_one_setting_switches_the_provider() -> None:
    assert isinstance(create_llm({**KEYS, "VERIRAG_LLM": "anthropic"}), AnthropicLLM)


def test_the_setting_ignores_case() -> None:
    assert isinstance(create_llm({**KEYS, "VERIRAG_LLM": "Anthropic"}), AnthropicLLM)


def test_an_unknown_provider_lists_the_choices() -> None:
    with pytest.raises(LLMError, match="gemini, anthropic"):
        create_llm({**KEYS, "VERIRAG_LLM": "other"})


def test_a_missing_key_is_reported_by_name() -> None:
    with pytest.raises(LLMError, match="ANTHROPIC_API_KEY"):
        create_llm({"VERIRAG_LLM": "anthropic"})
