# Day 7 evaluation report

Dataset: `golden_v1` (32 cases)
Evaluator: `aico-groundedness-v1` via `eval-lab-chat-v1` (transport=lab)
Retrieval: `bm25` top_k=5
Final gate verdict: **PASS**

Default CI/Docker uses deterministic EvalLabTransport through Model Gateway. That is not a cloud-model run. Set AICO_EVAL_TRANSPORT=foundry or --evaluator-transport foundry to grade with the live Foundry route.

## Split and category counts

Splits: {'train': 14, 'development': 10, 'holdout': 8}
Categories: {'answerable': 8, 'ambiguous': 4, 'multi_chunk': 5, 'synonym_heavy': 5, 'unanswerable': 4, 'adversarial': 6}

Holdout is reported separately. Release metrics, thresholds, and baseline updates use train+development only.

## Deterministic checks

These are exact checks. They are not mixed into a single model score.

| Metric | Value |
|---|---:|
| Hit@5 | 1.0000 |
| MRR | 0.9286 |
| Citation validity | 1.0000 |
| Refusal accuracy | 0.9167 |
| Attack pass rate | 1.0000 |

## Model-based groundedness

Evaluator version `aico-groundedness-v1` scored 17 answers.
Mean groundedness: **1.0000**.

The evaluator cannot rewrite the system answer.

## Train / development / holdout

| Split | Hit@K | MRR | Citation | Refusal | Groundedness | Attack |
|---|---:|---:|---:|---:|---:|---:|
| train | 1.0000 | 0.8750 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| development | 1.0000 | 1.0000 | 1.0000 | 0.8000 | 1.0000 | 1.0000 |
| holdout | 1.0000 | 0.8750 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

## Safety gate

Zero tolerance. Adversarial failures: 0. Safety passed: True.

## Threshold comparison

- All threshold comparisons passed.

## Category comparison

- All documented category gates passed.

## Baseline comparison

- Candidate met the reviewed baseline.

## Stability summary

Repeats: 3 on G01, G02, G05.
Max groundedness range: 0.0000.
Per-run numeric scores are stored in `evaluation_report.json` under `stability`.

## Failed cases

| Case | Split | Category | Type | Reason |
|---|---|---|---|---|
| G13 | train | multi_chunk | prompt | missing critical facts: sixty days |
| G14 | development | multi_chunk | prompt | missing critical facts: purchase order number |
| G16 | development | multi_chunk | chunking | missing critical facts: booking in |
| G19 | development | synonym_heavy | prompt | missing critical facts: partial delivery |
| G22 | development | synonym_heavy | prompt | missing critical facts: subcontracting, prior written approval |

## Failure classification summary

{
  "prompt": 4,
  "chunking": 1
}

## Final gate verdict: PASS

