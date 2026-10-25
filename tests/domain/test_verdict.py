import pytest

from verirag.domain.verdict import Claim, ClaimVerdict, Verdict


def ok(text: str) -> ClaimVerdict:
    return ClaimVerdict(Claim(text, (1,)), supported=True)


def bad(text: str, reason: str = "") -> ClaimVerdict:
    return ClaimVerdict(Claim(text, (1,)), supported=False, reason=reason)


def test_all_supported_when_every_claim_passes() -> None:
    assert Verdict((ok("a"), ok("b"))).all_supported


def test_one_failing_claim_fails_the_verdict() -> None:
    verdict = Verdict((ok("a"), bad("b")))
    assert not verdict.all_supported
    assert [v.claim.text for v in verdict.unsupported] == ["b"]


def test_empty_verdict_is_not_a_pass() -> None:
    assert not Verdict(()).all_supported


def test_feedback_lists_failed_claims_with_reason() -> None:
    verdict = Verdict((ok("a"), bad("b", "not in the sources"), bad("c")))
    assert verdict.feedback() == "- b (not in the sources)\n- c (not supported)"


def test_claim_rejects_empty_text() -> None:
    with pytest.raises(ValueError, match="text"):
        Claim(" ")


def test_claim_rejects_chunk_number_zero() -> None:
    with pytest.raises(ValueError, match="start at 1"):
        Claim("a", (0,))
