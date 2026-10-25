# ADR 2: Refuse instead of guessing

Status: accepted

## Context

Language models produce fluent text whether or not the sources support it. In a document
question-answering system, a confident wrong answer is worse than no answer: the reader has no
easy way to tell the difference. The project exists to give answers a reader can trust.

## Decision

The system releases an answer only when **every claim** of it has been verified against the
retrieved sources. Otherwise it returns a refusal that states the reason and lists the closest
passages. Concretely:

- If no retrieved chunk reaches the minimum score (0.30), the question is refused before any
  model is called.
- If the writer finds no answer in the sources, the system refuses at once.
- If claims are still unsupported after the retry limit, the system refuses.
- There is **no "answer directly" route** for questions that skip the documents. An answer that
  skips the verifier cannot be verified, so small talk and general knowledge are out of scope.
- A refusal is a normal result (exit code 2, `status: "refused"` in the MCP tool), not an error.

## Consequences

- Some questions that a model could have answered correctly from its own knowledge are refused.
  This is intended: the promise is "everything I tell you is backed by your documents".
- The refusal rate matters as much as the answer rate. The evaluation therefore measures both,
  including real questions refused by mistake (false refusals).
- The cost of a verified answer is at least two model calls, and up to six with retries.
- The user is never left guessing: a refusal says why, and points to what the documents do contain.

## Alternatives considered

- **Answer with a warning** ("this may be wrong"). Readers skip warnings.
- **A confidence number from the model.** Models are poorly calibrated about their own claims,
  and a number gives a false sense of precision.
- **Answer from general knowledge when the documents are silent.** Mixes verified and unverified
  content in one text, which is the failure this project tries to remove.
