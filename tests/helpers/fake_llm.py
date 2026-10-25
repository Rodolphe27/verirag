"""A scripted language model for tests: fast, free and fully predictable."""

from __future__ import annotations

from collections.abc import Callable

Reply = str | Callable[[str], str]


class ScriptedLLM:
    """Returns prepared replies in order and remembers every prompt it received.

    A reply is either a fixed text or a function of the prompt. If the code
    under test calls the model more often than the test expected, the fake
    fails loudly instead of inventing an answer.
    """

    def __init__(self, *replies: Reply) -> None:
        self._replies = list(replies)
        self.prompts: list[str] = []

    def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if not self._replies:
            raise AssertionError(f"unexpected extra model call #{len(self.prompts)}")
        reply = self._replies.pop(0)
        return reply(prompt) if callable(reply) else reply
