from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Claim:
    """One statement of an answer, with the sources the writer says support it.

    `chunk_numbers` are 1-based positions in the list of chunks the writer saw.
    """

    text: str
    chunk_numbers: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError("a claim needs some text")
        if any(n < 1 for n in self.chunk_numbers):
            raise ValueError("chunk numbers start at 1")


@dataclass(frozen=True)
class ClaimVerdict:
    """The verifier's decision for one claim."""

    claim: Claim
    supported: bool
    reason: str = ""


@dataclass(frozen=True)
class Verdict:
    """The verifier's decisions for a whole draft."""

    items: tuple[ClaimVerdict, ...]

    @property
    def all_supported(self) -> bool:
        """True only if there is at least one claim and every claim is supported.

        An empty verdict is NOT a pass. When the verifier says nothing, we
        cannot call the answer verified (fail safe).
        """
        return bool(self.items) and all(item.supported for item in self.items)

    @property
    def unsupported(self) -> tuple[ClaimVerdict, ...]:
        return tuple(item for item in self.items if not item.supported)

    def feedback(self) -> str:
        """Plain-text note for the writer: which claims failed and why."""
        return "\n".join(
            f"- {v.claim.text} ({v.reason or 'not supported'})" for v in self.unsupported
        )
