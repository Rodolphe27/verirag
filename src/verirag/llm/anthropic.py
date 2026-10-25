from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from typing import Any

import requests

from verirag.llm.http import Post, post_json
from verirag.ports import LLMError

DEFAULT_MODEL = "claude-haiku-5-5"
_URL = "https://api.anthropic.com/v1/messages"
_API_VERSION = "2023-06-01"


class AnthropicLLM:
    """Anthropic models through the Messages REST API. Implements the `LLM` port."""

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        max_tokens: int = 2048,
        post: Post = requests.post,
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        if not api_key:
            raise LLMError("ANTHROPIC_API_KEY is empty")
        self._api_key = api_key
        self._model = model
        self._max_tokens = max_tokens
        self._post = post
        self._sleep = sleep

    @classmethod
    def from_env(cls, environ: Mapping[str, str] = os.environ) -> AnthropicLLM:
        """Build from ANTHROPIC_API_KEY and the optional ANTHROPIC_MODEL."""
        key = environ.get("ANTHROPIC_API_KEY", "")
        if not key:
            raise LLMError("set the ANTHROPIC_API_KEY environment variable")
        return cls(key, environ.get("ANTHROPIC_MODEL", DEFAULT_MODEL))

    def complete(self, prompt: str) -> str:
        payload = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "temperature": 0,
            "messages": [{"role": "user", "content": prompt}],
        }
        headers = {
            "x-api-key": self._api_key,
            "anthropic-version": _API_VERSION,
            "content-type": "application/json",
        }
        kwargs: dict[str, Any] = {} if self._sleep is None else {"sleep": self._sleep}
        return _text(post_json(_URL, headers, payload, self._post, **kwargs))


def _text(body: Mapping[str, Any]) -> str:
    try:
        blocks = body["content"]
        text = "".join(block.get("text", "") for block in blocks if block.get("type") == "text")
    except (KeyError, TypeError, AttributeError) as error:
        raise LLMError("unexpected reply format from Anthropic") from error
    if not text.strip():
        raise LLMError("Anthropic returned an empty answer")
    return text
