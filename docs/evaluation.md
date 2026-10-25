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

Not measured yet. This section is filled in when the end-to-end run exists.

## Limits

- One corpus (EU legal text), 35 questions written by the author, who also chose the expected
  articles. A second person labelling the questions would be a better test.
- The hit rate uses files, not exact passages: an article counts as correct if it is the right
  article, even when the best passage inside it is another one.
- Embedding and keyword settings were not tuned on these questions; the only value inspected
  against them is the 0.30 threshold, and it was left unchanged.
