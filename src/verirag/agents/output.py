from __future__ import annotations

import json
from typing import Any


class BadModelOutput(ValueError):
    """The model replied, but not in the format we asked for."""


def extract_json(reply: str) -> dict[str, Any]:
    """Find the JSON object in a model reply.

    Models like to add a sentence before the JSON or wrap it in a Markdown code
    fence. We take everything from the first "{" to the last "}" and parse it.
    If that fails, we raise BadModelOutput with a short, clear message, because
    the caller (the graph) must decide what to do, not crash with a stack trace.
    """
    start, end = reply.find("{"), reply.rfind("}")
    if start == -1 or end < start:
        raise BadModelOutput("the reply contains no JSON object")
    try:
        data = json.loads(reply[start : end + 1])
    except json.JSONDecodeError as error:
        raise BadModelOutput(f"the reply is not valid JSON: {error.msg}") from error
    if not isinstance(data, dict):
        raise BadModelOutput("the JSON is not an object")
    return data
