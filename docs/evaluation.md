# Evaluation

Numbers in this file come from commands anyone can run. Nothing is estimated or rounded in
VeriRAG's favour. The question set is small (35 questions, one corpus), so read the results as
a smoke test with honest numbers, not as a benchmark.

## How to reproduce

```bash
verirag index docs/sample      # the 14 EU AI Act articles
verirag evaluate               # prints the table below
```

Questions and expected sources: [`eval/questions.json`](../eval/questions.json).
Embedding model: `paraphrase-multilingual-MiniLM-L12-v2` (local, German and English).
Chunks: about 450 characters with 150 overlap. Run date: 2026-10-25.

## 1. Retrieval: does the search find the right article?

27 answerable questions (22 English, 5 German) and 8 questions the corpus cannot answer.
A search *hits* when a correct article is among the 4 best chunks.

| Search mode | Hit rate@4 | MRR | False refusals | Correct refusals |
|---|---|---|---|---|
| dense (meaning only) | 93% | 0.84 | 0% | 75% |
| keyword (BM25 only) | 74% | 0.57 | 7% | 75% |
| **hybrid (both, default)** | **96%** | **0.87** | **0%** | **75%** |

- **Hit rate**: share of questions where a correct article is in the top 4.
- **MRR** (mean reciprocal rank): 1.0 if the right article is always first, 0.5 if always second.
- **False refusals**: real questions the score gate would refuse by mistake.
- **Correct refusals**: off-topic questions the score gate refuses.

What the numbers say:

1. **Hybrid beats both parts.** Keyword search alone is weak on everyday wording (74%), meaning
   search alone misses exact terms; fusing them gives the best hit rate and the best ranking.
   With 27 questions, a difference of one question is 3.7 points, so the gap between dense and
   hybrid (93% vs 96%, one question) is suggestive, not proof.
2. **German questions work on English text.** The multilingual model finds the right article
   for the five German questions.
3. **The score gate is a first filter, not a guarantee.** A question is refused when no chunk
   reaches a cosine score of 0.30. Six of eight off-topic questions are stopped here. The two
   that pass are close to the corpus in topic:

   | Question | Best score |
   |---|---|
   | How many employees does the European Commission have? | 0.54 |
   | What does the GDPR say about the right to data portability? | 0.48 |

   Raising the threshold does not fix this: the weakest real question scores 0.39, and from
   0.40 on real questions start to be refused (false refusals 4% at 0.40, 7% at 0.45).
   A similarity score measures *topic*, not *answerability*. That is why VeriRAG has a second
   layer: the writer and the verifier. The end-to-end evaluation (below) checks whether they
   catch what the gate lets through.

## 2. End to end: answers and refusals

The whole system (search, writer, verifier, refusal) on the same 35 questions, with Gemini
(`gemini-3.5-flash`, falling back to `gemini-3.1-flash-lite`). Run date: 2026-10-25.

```bash
verirag evaluate --end-to-end --output eval/results/2026-10-25-gemini.json
verirag evaluate --end-to-end --resume eval/results/2026-10-25-gemini.json \
                 --output eval/results/2026-10-25-gemini.json     # second pass
```

| Measure | Result |
|---|---|
| Answerable: answered, cites a correct article | 24/27 (89%) |
| Answerable: answered, cites another article | 0/27 (0%) |
| Answerable: refused | 3/27 (11%) |
| Unanswerable: refused | 8/8 (100%) |
| Unanswerable: answered anyway | 0/8 (0%) |
| Runs that failed with a model error (left out above) | 0 |

Every answer, with its status and citations, is saved in
[`eval/results/2026-10-25-gemini.json`](../eval/results/2026-10-25-gemini.json).

**The run was done in two passes.** The free Gemini tier answered "model overloaded" (HTTP 503)
for 14 of the 27 answerable questions in the first pass (29 minutes). Those runs say nothing about
quality, so they were not counted as refusals. The second pass (`--resume`) kept the 13 finished
results unchanged and asked only the 14 failed questions again; all finished.

What the numbers say:

1. **No wrong answer was released.** None of the 8 off-topic questions was answered, and none of
   the answered questions cited an unrelated article.
2. **The second layer works as designed.** The two off-topic questions that the score gate let
   through (GDPR, European Commission staff; see section 1) were stopped by the writer and the
   verifier: the answer "could not be verified against the sources".
3. **The system errs towards refusing.** Three real questions were refused:
   - *How must deepfakes be labelled?* The search did not return Article 50 in its top 4. This is
     the one retrieval miss of section 1. With no supporting text the system said "I am not sure"
     and did not invent an answer. This is the intended behaviour, and it also shows that search
     quality limits what the system can answer.
   - *Is it allowed to rank people by their social behaviour...?* and *What is human oversight
     supposed to achieve?* were refused after the retries because the verifier did not accept the
     quoted proof. The articles do contain the answers, so these are **false refusals** caused by
     strictness. It is the price of the rule "doubt means not supported" (ADR 2 and ADR 5).
4. **German questions work.** The three German questions that finished were answered from the
   English articles, in German.

What this evaluation does **not** show: whether the wording of each answer is fully correct. The
code guarantees that each claim has a quote that exists in a retrieved passage; it does not grade
the final text. The author read a sample of the answers against the articles and found them
consistent (for example the fine of up to EUR 35 000 000 or 7 % of turnover for prohibited
practices), but this was a sample read, not a full review.

## Limits

- One corpus (EU legal text), 35 questions written by the author, who also chose the expected
  articles. One model provider was used live (Gemini); the Anthropic adapter is tested with mocked
  HTTP only. A second person labelling the questions would be a better test.
- The hit rate uses files, not exact passages: an article counts as correct if it is the right
  article, even when the best passage inside it is another one.
- Embedding and keyword settings were not tuned on these questions; the only value inspected
  against them is the 0.30 threshold, and it was left unchanged.
