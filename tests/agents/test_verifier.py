import json
from typing import Any

from tests.helpers.fake_llm import ScriptedLLM
from verirag.agents.verifier import Verifier, locate_evidence
from verirag.domain.answer import Draft
from verirag.domain.chunk import Chunk
from verirag.domain.verdict import Claim

CHUNKS = [
    Chunk("Fines can reach 35 million euros\nor 7 % of turnover.", "article-99.md"),
    Chunk("Providers must keep logs.", "article-12.md"),
]
DRAFT = Draft(
    "Fines reach 35 million euros. Logs are kept for ten years.",
    (Claim("Fines reach 35 million euros.", (1,)), Claim("Logs are kept for ten years.", (2,))),
)


def verdict(claim: int, supported: bool, evidence: str = "", reason: str = "") -> dict[str, Any]:
    return {"claim": claim, "supported": supported, "evidence": evidence, "reason": reason}


def reply(*items: dict[str, Any]) -> str:
    return json.dumps({"verdicts": list(items)})


def check(model_reply: str, draft: Draft = DRAFT) -> list[tuple[bool, str]]:
    result = Verifier(ScriptedLLM(model_reply)).verify(draft, CHUNKS)
    return [(v.supported, v.reason) for v in result.items]


def test_a_claim_with_real_evidence_is_supported() -> None:
    model = reply(
        verdict(1, True, "fines can reach 35 million euros"),
        verdict(2, False, reason="the sources do not give a duration"),
    )
    assert check(model) == [
        (True, "fines can reach 35 million euros"),
        (False, "the sources do not give a duration"),
    ]


def test_evidence_matches_despite_line_breaks_and_capitals() -> None:
    model = reply(
        verdict(1, True, "Fines can reach 35 million euros or 7 % of turnover"), verdict(2, False)
    )
    assert check(model)[0][0] is True


def test_made_up_evidence_is_rejected_even_if_the_model_says_supported() -> None:
    model = reply(verdict(1, True, "fines can reach 99 billion"), verdict(2, False))
    assert check(model)[0] == (False, "the quoted evidence is not in the sources")


def test_supported_without_any_evidence_is_rejected() -> None:
    model = reply(verdict(1, True, ""), verdict(2, False))
    assert check(model)[0][0] is False


def test_a_missing_verdict_means_not_supported() -> None:
    model = reply(verdict(1, True, "fines can reach 35 million euros"))
    assert check(model)[1] == (False, "the verifier gave no verdict for this claim")


def test_an_unreadable_reply_fails_every_claim() -> None:
    assert check("sorry, I cannot do that") == [
        (False, "the verifier reply could not be read"),
        (False, "the verifier reply could not be read"),
    ]


def test_a_malformed_entry_is_ignored_and_its_claim_fails() -> None:
    model = json.dumps({"verdicts": [{"claim": "x"}, verdict(2, False, reason="no")]})
    assert [s for s, _ in check(model)] == [False, False]


def test_the_first_verdict_wins_when_a_claim_appears_twice() -> None:
    model = reply(
        verdict(1, False, reason="first"), verdict(1, True, "fines can reach"), verdict(2, False)
    )
    assert check(model)[0] == (False, "first")


def test_a_draft_without_claims_gets_an_empty_verdict_and_no_model_call() -> None:
    llm = ScriptedLLM()
    result = Verifier(llm).verify(Draft("The sources do not say.", ()), CHUNKS)
    assert not result.all_supported
    assert llm.prompts == []


def test_the_prompt_lists_all_sources_and_numbered_claims() -> None:
    llm = ScriptedLLM(reply(verdict(1, False), verdict(2, False)))
    Verifier(llm).verify(DRAFT, CHUNKS)
    prompt = llm.prompts[0]
    assert "[2] Providers must keep logs." in prompt
    assert "2. Logs are kept for ten years." in prompt


def test_locate_evidence_returns_the_chunk_that_holds_the_quote() -> None:
    assert locate_evidence("PROVIDERS MUST keep logs", CHUNKS) is CHUNKS[1]


def test_locate_evidence_finds_nothing_for_empty_or_unknown_quotes() -> None:
    assert locate_evidence("", CHUNKS) is None
    assert locate_evidence("something else", CHUNKS) is None


def test_a_quote_spread_over_two_chunks_is_not_evidence() -> None:
    assert locate_evidence("or 7 % of turnover. Providers must keep logs", CHUNKS) is None
