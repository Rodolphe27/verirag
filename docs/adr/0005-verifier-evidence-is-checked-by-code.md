# ADR 5: The verifier's evidence is checked by code

Status: accepted

## Context

A second model that judges the first model's answer is a good idea, but it is still a model. It
can say "supported" about a claim that is not, and it can invent a reason. If the system simply
trusted its yes or no, the verification would be only as reliable as the thing it checks.

## Decision

A claim counts as supported only if **both** conditions hold:

1. the verifier says it is supported, **and**
2. the verifier quotes the exact words from the sources that prove it, and **the code finds that
   quote inside one retrieved chunk**.

Everything else fails: no verdict, a malformed entry, an unreadable reply, a quote that is empty,
shorter than 15 characters, or not present. "Doubt means not supported". An empty verdict (a draft
with no claims) is never a pass.

The matching is tolerant of two harmless habits of models and strict about the rest. It accepts
`...` (or `[...]`, or `…`) for skipped words, as long as every part appears in the chunk in the
same order, and it treats curly quotes and long dashes like plain ones. The whole quote must still
come from a single chunk.

The same quote then decides the **citation**: the final answer cites the chunk that contains the
verified evidence, not the source numbers the writer wrote down.

## Consequences

- A made-up justification cannot pass. The system's trust rests on a string search, not on the
  model's honesty.
- Some true claims are rejected because the model paraphrased its quote. This costs a retry or a
  refusal, and it is the safe direction for this project. The first real run showed this failure
  once, which is why the ellipsis and typography tolerance exists.
- Citations are exact. A wrong source number from the writer cannot reach the reader (a test covers
  this case).
- The check proves that the quote is in the sources. It does not prove that the quote logically
  supports the claim; that judgement still belongs to the verifier model. The evaluation notes
  this limit.

## Alternatives considered

- **Trust the verifier's yes or no.** Simple, but a single hallucination passes unnoticed.
- **A different, stronger model as verifier.** Helps, costs more, and does not remove the need for
  a check that does not depend on any model.
- **Natural-language inference classifiers.** A possible later step, which would add a dependency
  and a model to maintain.
