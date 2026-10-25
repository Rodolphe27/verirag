import json
from typing import Any

import pytest
import requests

from verirag.llm.anthropic import DEFAULT_MODEL, AnthropicLLM
from verirag.ports import LLM, LLMError


def reply(status: int, body: dict[str, Any]) -> requests.Response:
    response = requests.Response()
    response.status_code = status
    response._content = json.dumps(body).encode()
    return response


def message(text: str) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": text}]}


class Recorder:
    def __init__(self, *outcomes: requests.Response) -> None:
        self.outcomes = list(outcomes)
        self.sent: list[dict[str, Any]] = []

    def __call__(self, url: str, *, headers: Any, json: Any, timeout: float) -> requests.Response:
        self.sent.append({"url": url, "headers": dict(headers), "json": json})
        return self.outcomes.pop(0)


def make(post: Recorder) -> AnthropicLLM:
    return AnthropicLLM("KEY", post=post, sleep=lambda _: None)


def test_it_fits_the_llm_port() -> None:
    llm: LLM = make(Recorder())
    assert llm is not None


def test_the_prompt_is_sent_as_a_user_message() -> None:
    post = Recorder(reply(200, message("hello")))
    assert make(post).complete("Say hello") == "hello"
    sent = post.sent[0]["json"]
    assert sent["messages"] == [{"role": "user", "content": "Say hello"}]
    assert sent["model"] == DEFAULT_MODEL
    assert sent["temperature"] == 0


def test_the_key_and_version_travel_in_headers() -> None:
    post = Recorder(reply(200, message("x")))
    make(post).complete("q")
    headers = post.sent[0]["headers"]
    assert headers["x-api-key"] == "KEY"
    assert headers["anthropic-version"] == "2023-06-01"


def test_several_text_blocks_are_joined() -> None:
    body = {"content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}
    assert make(Recorder(reply(200, body))).complete("q") == "ab"


def test_a_busy_server_is_retried() -> None:
    post = Recorder(reply(503, {}), reply(200, message("ok")))
    assert make(post).complete("q") == "ok"


@pytest.mark.parametrize("bad", [{}, {"content": []}, {"content": [{"type": "image"}]}])
def test_a_reply_without_text_is_reported(bad: dict[str, Any]) -> None:
    with pytest.raises(LLMError):
        make(Recorder(reply(200, bad))).complete("q")


def test_from_env_needs_a_key() -> None:
    with pytest.raises(LLMError, match="ANTHROPIC_API_KEY"):
        AnthropicLLM.from_env({})


def test_from_env_reads_the_model() -> None:
    llm = AnthropicLLM.from_env({"ANTHROPIC_API_KEY": "k", "ANTHROPIC_MODEL": "my-model"})
    assert llm._model == "my-model"
