from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from typing import Any, Protocol

import requests

from verirag.ports import LLMError

RETRY_STATUS = {429, 500, 502, 503, 504}


class Post(Protocol):
    """The part of `requests.post` we use. Tests pass a fake instead."""

    def __call__(
        self, url: str, *, headers: Mapping[str, str], json: Mapping[str, Any], timeout: float
    ) -> requests.Response: ...


def post_json(
    url: str,
    headers: Mapping[str, str],
    payload: Mapping[str, Any],
    post: Post = requests.post,
    sleep: Callable[[float], None] = time.sleep,
    attempts: int = 4,
    timeout: float = 60.0,
) -> dict[str, Any]:
    """POST JSON and return the JSON reply, retrying when the server is busy.

    Providers answer 429 (too many requests) or 5xx (server trouble) from time
    to time. Those are temporary, so we wait 1, 2, 4... seconds and try again.
    Other errors (a wrong key, a bad request) will not fix themselves, so they
    fail at once. Error messages never contain the headers, which hold the key.
    """
    for attempt in range(1, attempts + 1):
        try:
            response = post(url, headers=headers, json=payload, timeout=timeout)
        except requests.RequestException as error:
            if attempt == attempts:
                raise LLMError(f"network error: {type(error).__name__}") from error
        else:
            if response.ok:
                body: dict[str, Any] = response.json()
                return body
            if response.status_code not in RETRY_STATUS or attempt == attempts:
                raise LLMError(f"model call failed with HTTP {response.status_code}")
        sleep(2 ** (attempt - 1))
    raise AssertionError("unreachable")  # the loop always returns or raises
