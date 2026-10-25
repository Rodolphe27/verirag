import json
from typing import Any

import pytest
import requests

from verirag.llm.gemini import DEFAULT_MODELS, GeminiLLM
from verirag.ports import LLM, LLMError


def reply(status: int, text: str | None = None) -> requests.Response:
    response = requests.Response()
    response.status_code = status
    body = {"candidates": [{"content": {"parts": [{"text": text}]}}]} if text is not None else {}
    response._content = json.dumps(body).encode()
    return response


class Recorder:
    def __init__(self, *outcomes: requests.Response) -> None:
        self.outcomes = list(outcomes)
        self.urls: list[str] = []
        self.sent: list[dict[str, Any]] = []

    def __call__(self, url: str, *, headers: Any, json: Any, timeout: float) -> requests.Response:
        self.urls.append(url)
        self.sent.append({"headers": dict(headers), "json": json})
        return self.outcomes.pop(0)


def make(post: Recorder) -> GeminiLLM:
    return GeminiLLM("KEY", post=post, sleep=lambda _: None)


def test_it_fits_the_llm_port() -> None:
    llm: LLM = make(Recorder())
    assert llm is not None


def test_the_prompt_is_sent_and_the_text_returned() -> None:
    post = Recorder(reply(200, "hello"))
    assert make(post).complete("Say hello") == "hello"
    assert post.sent[0]["json"]["contents"][0]["parts"][0]["text"] == "Say hello"
    assert post.sent[0]["json"]["generationConfig"]["temperature"] == 0


def test_the_key_is_in_a_header_and_not_in_the_url() -> None:
    post = Recorder(reply(200, "x"))
    make(post).complete("q")
    assert post.sent[0]["headers"]["x-goog-api-key"] == "KEY"
    assert "KEY" not in post.urls[0]


def test_the_next_model_is_tried_when_the_first_stays_busy() -> None:
    post = Recorder(*[reply(503)] * 4, reply(200, "from second"))
    assert make(post).complete("q") == "from second"
    assert DEFAULT_MODELS[0] in post.urls[0]
    assert DEFAULT_MODELS[1] in post.urls[-1]


def test_all_models_failing_gives_one_clear_error() -> None:
    post = Recorder(*[reply(503)] * 8)
    with pytest.raises(LLMError, match="all models failed"):
        make(post).complete("q")


@pytest.mark.parametrize("bad", [{}, {"candidates": []}])
def test_an_unexpected_reply_format_is_reported(bad: dict[str, Any]) -> None:
    response = requests.Response()
    response.status_code = 200
    response._content = json.dumps(bad).encode()
    with pytest.raises(LLMError, match="unexpected reply"):
        make(Recorder(response, response)).complete("q")


def test_an_empty_answer_is_an_error() -> None:
    with pytest.raises(LLMError, match="empty"):
        make(Recorder(reply(200, " "), reply(200, " "))).complete("q")


def test_from_env_needs_a_key() -> None:
    with pytest.raises(LLMError, match="GEMINI_API_KEY"):
        GeminiLLM.from_env({})


def test_from_env_puts_the_preferred_model_first() -> None:
    llm = GeminiLLM.from_env({"GEMINI_API_KEY": "k", "GEMINI_MODEL": "my-model"})
    assert llm._models[0] == "my-model"
    assert llm._models[1:] == list(DEFAULT_MODELS)
