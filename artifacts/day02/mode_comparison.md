# Day 2 mode comparison

Same corpus, same chunker, same offsets as Day 1. Labels were not edited.

RRF k = **60**. Raising k flattens the gap between rank 1 and rank 10 (gentler fusion). Lowering k makes the top of each list dominate.

Hybrid fuses **ranks**, never raw scores. BM25 is unbounded and corpus-dependent; cosine sits in [-1, 1]. Averaging them would let the larger unit win.

## Score floors

| Mode | Floor | Why |
|---|---:|---|
| bm25 | 2.0000 | Unchanged from Day 1. Exact-term hits score well above 4; below 2 is weak leftover overlap. |
| vector | 0.3500 | Cosine is a ranking signal, not confidence. A vector index always returns a neighbour; this floor is where we refuse it. |
| hybrid | 0.0311 | `1/(k+1) + 1/(k+8)`. Rank-1 in one list only is not enough — that is what a no_match nearest neighbour looks like. |

## Three-way metrics (scored queries Q01–Q08, Q11–Q15)

| Mode | Hit@1 | Hit@5 | MRR | Floor |
|---|---:|---:|---:|---:|
| bm25 | 0.6154 | 0.6923 | 0.6538 | 2.0000 |
| vector | 0.7692 | 0.9231 | 0.8269 | 0.3500 |
| hybrid | 0.6154 | 0.8462 | 0.7308 | 0.0311 |

### Per category

| Category | n | bm25 Hit@5 / MRR | vector Hit@5 / MRR | hybrid Hit@5 / MRR |
|---|---:|---|---|---|
| exact_term | 4 | 1.0000 / 0.8750 | 1.0000 / 1.0000 | 1.0000 / 0.8750 |
| synonym_poor | 2 | 0.5000 / 0.5000 | 1.0000 / 0.6250 | 0.5000 / 0.5000 |
| multi_chunk | 2 | 1.0000 / 1.0000 | 1.0000 / 0.7500 | 1.0000 / 1.0000 |
| semantic_only | 5 | 0.4000 / 0.4000 | 0.8000 / 0.8000 | 0.8000 / 0.6000 |

## no_match (Q09, Q10, Q16) — scored separately

Correct = zero hits with score above that mode's floor. Not folded into Hit@1 / Hit@5 / MRR.

| Mode | Query | max score | above floor | correct |
|---|---|---:|---:|---|
| bm25 | Q09 | 8.3264 | 5 | False |
| bm25 | Q10 | 5.9396 | 5 | False |
| bm25 | Q16 | 5.9987 | 3 | False |
| vector | Q09 | 0.3807 | 1 | False |
| vector | Q10 | 0.5180 | 5 | False |
| vector | Q16 | 0.3898 | 5 | False |
| hybrid | Q09 | 0.0323 | 4 | False |
| hybrid | Q10 | 0.0328 | 3 | False |
| hybrid | Q16 | 0.0323 | 2 | False |

## Where vector beat BM25

**Q05** (`synonym_poor`): Can a vendor hand the agreement over to another company?

Vector first-match rank=4; BM25 first-match rank=None.

The query wording and the document wording do not overlap enough for BM25. That is the `semantic_only` point: lexical retrieval had nothing distinctive to score.

## Where BM25 beat vector

**Q07** (`multi_chunk`): How much notice is required before a supplier is removed, and does the contract override it?

BM25 first-match rank=1; vector first-match rank=2.

Shared terms already put the right chunk at the top for BM25. Cosine ranked a nearby-meaning passage above the exact span.

## Hybrid vs both singles

- vs bm25: hybrid MRR 0.7308 > 0.6538
- vs vector: hybrid MRR 0.7308 <= 0.8269
- hybrid lost on `exact_term` (MRR 0.8750 vs vector 1.0000)
- hybrid lost on `synonym_poor` (MRR 0.5000 vs vector 0.6250)
- hybrid lost on `semantic_only` (MRR 0.6000 vs vector 0.8000)

## Cache evidence

Run twice from the project root:

```
python -m aico.retrieval.embed --index data/index --out data/vectors
```

First run embeds uncached chunks. Second run over unchanged chunks must print `provider_calls=0`. Edit one chunk's text, re-run: that chunk and only that chunk is re-embedded. Invalidation key is the SHA-256 of the chunk **text**, plus model alias.

