import json
from typing import Any

import pytest
import requests

from verirag.llm.http import post_json
from verirag.ports import LLMError


def reply(status: int, body: dict[str, Any] | None = None) -> requests.Response:
    response = requests.Response()
    response.status_code = status
    response._content = b"{}" if body is None else json.dumps(body).encode()
    return response


class FakePost:
    """Plays a list of outcomes (a response or an exception) and counts the calls."""

    def __init__(self, *outcomes: requests.Response | Exception) -> None:
        self.outcomes = list(outcomes)
        self.calls = 0

    def __call__(self, url: str, **kwargs: Any) -> requests.Response:
        self.calls += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def run(post: FakePost, waits: list[float]) -> dict[str, Any]:
    return post_json("https://x", {"k": "secret"}, {}, post=post, sleep=waits.append)


def test_a_good_reply_is_returned_at_once() -> None:
    waits: list[float] = []
    assert run(FakePost(reply(200, {"ok": 1})), waits) == {"ok": 1}
    assert waits == []


def test_a_busy_server_is_retried_with_growing_waits() -> None:
    waits: list[float] = []
    post = FakePost(reply(429), reply(503), reply(200, {"ok": 1}))
    assert run(post, waits) == {"ok": 1}
    assert waits == [1, 2]


def test_gives_up_after_the_last_attempt() -> None:
    post = FakePost(*[reply(503)] * 4)
    with pytest.raises(LLMError, match="503"):
        run(post, [])
    assert post.calls == 4


def test_a_wrong_key_fails_without_retry() -> None:
    post = FakePost(reply(401))
    with pytest.raises(LLMError, match="401"):
        run(post, [])
    assert post.calls == 1


def test_a_network_error_is_retried_then_reported() -> None:
    post = FakePost(*[requests.ConnectionError("down")] * 4)
    with pytest.raises(LLMError, match="ConnectionError"):
        run(post, [])
    assert post.calls == 4


def test_the_key_never_appears_in_the_error() -> None:
    with pytest.raises(LLMError) as info:
        run(FakePost(reply(401)), [])
    assert "secret" not in str(info.value)
