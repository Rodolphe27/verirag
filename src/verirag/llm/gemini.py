from __future__ import annotations

import os
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import requests

from verirag.llm.http import Post, post_json
from verirag.ports import LLMError

DEFAULT_MODELS = ("gemini-3.5-flash", "gemini-3.1-flash-lite")
_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


class GeminiLLM:
    """Google Gemini through its REST API. Implements the `LLM` port.

    If a model stays unavailable after the retries, the next model of the list
    is tried: free models are often busy.
    """

    def __init__(
        self,
        api_key: str,
        models: Sequence[str] = DEFAULT_MODELS,
        post: Post = requests.post,
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        if not api_key:
            raise LLMError("GEMINI_API_KEY is empty")
        self._api_key = api_key
        self._models = list(models)
        self._post = post
        self._sleep = sleep

    @classmethod
    def from_env(cls, environ: Mapping[str, str] = os.environ) -> GeminiLLM:
        """Build from GEMINI_API_KEY and the optional GEMINI_MODEL. The key is never
        written in code or in the repository."""
        key = environ.get("GEMINI_API_KEY", "")
        if not key:
            raise LLMError("set the GEMINI_API_KEY environment variable")
        preferred = environ.get("GEMINI_MODEL")
        models = [preferred, *DEFAULT_MODELS] if preferred else list(DEFAULT_MODELS)
        return cls(key, tuple(dict.fromkeys(models)))

    def complete(self, prompt: str) -> str:
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0},  # 0: the most predictable output
        }
        # The key travels in a header, not in the URL, so it cannot show up in error text.
        headers = {"x-goog-api-key": self._api_key, "content-type": "application/json"}
        last_error: LLMError | None = None
        for model in self._models:
            try:
                kwargs: dict[str, Any] = {} if self._sleep is None else {"sleep": self._sleep}
                body = post_json(_URL.format(model=model), headers, payload, self._post, **kwargs)
                return _text(body)
            except LLMError as error:
                last_error = error
        raise LLMError(f"all models failed, last error: {last_error}")


def _text(body: Mapping[str, Any]) -> str:
    try:
        parts = body["candidates"][0]["content"]["parts"]
        text = "".join(part.get("text", "") for part in parts)
    except (KeyError, IndexError, TypeError) as error:
        raise LLMError("unexpected reply format from Gemini") from error
    if not text.strip():
        raise LLMError("Gemini returned an empty answer")
    return text
