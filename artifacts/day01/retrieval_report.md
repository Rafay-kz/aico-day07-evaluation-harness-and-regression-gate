# Day 1 retrieval report

Lexical BM25 baseline. No embeddings. Labels were not edited.

## Configurations

200/40 and 400/80 are the assignment's suggested pair. They sit on either side of a typical policy section (~150–250 words). 200 is small enough that an anchor can fall across a cut; 400 is large enough that a fact and its exception can share a chunk. Overlap is 20% of the window in both cases, so the comparison is about size, not a different overlap policy.

Token counting: `whitespace_separated_words` — a token is a maximal run of non-whitespace characters. Punctuation stays attached until search strips it. This is not a model tokenizer and not a character count: `thirty five` is two tokens here and might be one or two subword pieces in an LLM tokenizer.

Search tokenisation (query and documents): the same whitespace split, then lowercase, strip edge punctuation, drop a small English stopword list. Stopwords are removed because `the` / `of` / `what` appear in almost every chunk; IDF would already down-weight them, and dropping them keeps scores on content words. `must` and `not` are kept.

BM25 parameters: K1=1.5, B=0.75 (named constants).

Score floor for no_match: **2.0**. Exact-term rank-1 hits on this corpus score well above 4. Scores below 2.0 are single weak-term leftovers. The floor is *not* set high enough to pretend Q09/Q10 returned nothing — those queries retrieve real bait passages (late payment / late delivery) and are reported as such.

## Metrics (scored queries Q01–Q08)

| Config | Chunks | Hit@1 | Hit@5 | MRR |
|---|---:|---:|---:|---:|
| 200/40 | 35 | 0.7500 | 0.8750 | 0.8125 |
| 400/80 | 16 | 0.7500 | 0.8750 | 0.8125 |

### Per category

**200/40**

| Category | n | Hit@1 | Hit@5 | MRR | Full hit |
|---|---:|---:|---:|---:|---|
| exact_term | 4 | 0.7500 | 1.0000 | 0.8750 | — |
| synonym_poor | 2 | 0.5000 | 0.5000 | 0.5000 | — |
| multi_chunk | 2 | 1.0000 | 1.0000 | 1.0000 | 0.5000 |

**400/80**

| Category | n | Hit@1 | Hit@5 | MRR | Full hit |
|---|---:|---:|---:|---:|---|
| exact_term | 4 | 0.7500 | 1.0000 | 0.8750 | — |
| synonym_poor | 2 | 0.5000 | 0.5000 | 0.5000 | — |
| multi_chunk | 2 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

## no_match (Q09–Q10) — scored separately

Not folded into Hit@1 / Hit@5 / MRR. Correct = zero hits with score > 2.0.

| Config | Query | max score | above floor | correct |
|---|---|---:|---:|---|
| 200/40 | Q09 | 10.1370 | 5 | False |
| 200/40 | Q10 | 7.0596 | 5 | False |
| 400/80 | Q09 | 9.6883 | 3 | False |
| 400/80 | Q10 | 5.8683 | 3 | False |

- 200/40: 0/2 correctly returned nothing above the floor.
- 400/80: 0/2 correctly returned nothing above the floor.

## Winner: 400/80

Config 400/80 wins because 400-token chunks leave fewer competing slices of the word 'notice', so both Q07 anchors appear in the top 5: ninety days written notice of withdrawal (DOC-001) and the contract clause that supersedes any notice period in the sourcing policy (DOC-002). At 200/40, one of those anchors is crowded out of the top 5 even though Hit@1 still fires on the other.

## Query BM25 handled badly

**Q05** (synonym_poor): Can a vendor hand the agreement over to another company?

The query never uses the document's words. It says `vendor` and `hand the agreement over`; DOC-002 says `may not assign or novate the agreement`. After stopword removal the query is essentially `vendor hand agreement another company` — none of those distinctive terms are `assign` or `novate`, which are also rare (high IDF) in the corpus. BM25 therefore ranks other chunks that share generic words like `agreement`. Fix without embeddings: expand the query with `assign` and `novate` at search time. (Q06 happened to Hit@1 because `order` occurs in the same partial-delivery passage as the anchor — co-occurrence, not synonym understanding.)

## Per-query detail (winning config)

| Query | Category | Hit@1 | Hit@5 | Rank | Full hit |
|---|---|---|---|---:|---|
| Q01 | exact_term | True | True | 1 | True |
| Q02 | exact_term | True | True | 1 | True |
| Q03 | exact_term | True | True | 1 | True |
| Q04 | exact_term | False | True | 2 | True |
| Q05 | synonym_poor | False | False | — | False |
| Q06 | synonym_poor | True | True | 1 | True |
| Q07 | multi_chunk | True | True | 1 | True |
| Q08 | multi_chunk | True | True | 1 | True |

