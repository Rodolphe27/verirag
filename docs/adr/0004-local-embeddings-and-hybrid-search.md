# ADR 4: Local embeddings and hybrid search

Status: accepted

## Context

The documents are in German and English, may be private, and the project should run for free on a
laptop. Meaning search (embeddings) handles rephrased questions well but can miss exact terms such
as an article number; keyword search does the opposite.

## Decision

- Embed locally with `fastembed` and the multilingual model
  `paraphrase-multilingual-MiniLM-L12-v2`. No document leaves the machine for embedding, and there
  is no embedding cost.
- Search with **both** methods and fuse the rankings with Reciprocal Rank Fusion (RRF): meaning
  search (cosine similarity, NumPy over a JSON index) and keyword search (BM25).
- RRF adds `1 / (60 + rank)` from each list. Ranks are used instead of raw scores because cosine
  (0 to 1) and BM25 (0 to 20 or more) are on different scales.
- The score returned on a chunk is always the **cosine similarity**, whatever the ranking, because
  its 0-to-1 range is easy to interpret and the score gate uses it.

## Consequences

- Measured on 27 questions ([evaluation](../evaluation.md)): a correct article is in the top 4 for
  96% (hybrid), 93% (meaning only) and 74% (keywords only).
- The index is a JSON file read into memory, with a hash per file so unchanged files are not
  embedded again. That is fine for hundreds of documents; thousands would call for a vector
  database behind the same `Retriever` port.
- Words are cut to six letters as a simple stemmer that also works for German. It causes some false
  matches between long words with the same start; the evaluation did not show a problem, but it is a
  known trade-off.
- The first run downloads the model (about 220 MB).

## Alternatives considered

- **A hosted embedding API.** Better models exist, but documents would leave the machine, with a
  cost and a key to manage.
- **A vector database now.** Premature for this corpus size; the `Retriever` port keeps the door open.
- **Meaning search only.** The evaluation shows keywords add a little (3.7 points on this set) at
  almost no cost. The gain is small on a small set, and larger on collections full of exact terms.
