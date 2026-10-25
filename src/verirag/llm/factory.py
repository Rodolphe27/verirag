from __future__ import annotations

import os
from collections.abc import Mapping

from verirag.llm.anthropic import AnthropicLLM
from verirag.llm.gemini import GeminiLLM
from verirag.ports import LLM, LLMError

PROVIDERS = ("gemini", "anthropic")


def create_llm(environ: Mapping[str, str] = os.environ) -> LLM:
    """Build the model chosen by the VERIRAG_LLM setting (default: gemini).

    This is the only place that knows the list of providers. To add one, write
    an adapter and add one line here; the core does not change.
    """
    provider = environ.get("VERIRAG_LLM", "gemini").lower()
    if provider == "gemini":
        return GeminiLLM.from_env(environ)
    if provider == "anthropic":
        return AnthropicLLM.from_env(environ)
    raise LLMError(f"unknown VERIRAG_LLM '{provider}'; choose one of: {', '.join(PROVIDERS)}")
