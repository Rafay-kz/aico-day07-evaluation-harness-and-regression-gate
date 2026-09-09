# Day 7 evaluation resources

These files are the reviewed Day 7 quality baseline. They are not generated at
runtime, except `evals/baseline_v1.json` which may be rewritten only through the
explicit `--update-baseline` path.

| File | Role |
|---|---|
| `golden_v1.json` | Versioned labelled cases. Labels were set from the synthetic corpus before candidate-run inspection. |
| `thresholds_v1.json` | Machine-readable overall and per-category release gates. Applied to train+development; never derived from holdout. |
| `baseline_v1.json` | Reviewed expected performance snapshot. Normal evaluation must not overwrite it. |

## Field mapping

`golden_v1.json` uses these case fields:

- `case_id` — stable ID
- `category` — `answerable` / `ambiguous` / `multi_chunk` / `synonym_heavy` / `unanswerable` / `adversarial`
- `split` — `train` / `development` / `holdout`
- `question` — the input (assignment “question/input”)
- `expected_sources` — document IDs such as `DOC-004`, matched only against retrieved hits
- `answerability` — `answerable` / `unanswerable` / `ambiguous`
- `critical_facts` — strings that a supported answer must include; also used as Day 1 relevance anchors for Hit@K/MRR
- `prohibited_claims` — strings that must not appear in the system answer
- `expected_outcome` — `answer` / `refuse` / `clarify` / `block`

## Splits

Holdout is measured and reported separately. It is not used to choose retrieval
settings, prompts, routing, thresholds, labels, expected sources, answerability,
critical facts, or prohibited claims, and it is excluded from baseline metrics.

Tuning helpers (`iter_tuning_cases`) yield only `train` and `development`.

## Evaluator

Model-based groundedness uses the existing Model Gateway with evaluator prompt
version `aico-groundedness-v1`.

- Default / CI: `EvalLabTransport` (deterministic lab). Reports set
  `cloud_model_executed: false` and `ci_evidence_note`.
- Optional live route: `--evaluator-transport foundry` or `AICO_EVAL_TRANSPORT=foundry`.

Per-run stability scores are persisted in `evaluation_report.json` under `stability`.

## Baseline update

```bash
uv run python -m aico.evals.day07 --update-baseline --reviewed-by "Your Name"
```

Without `--reviewed-by` the written file has `reviewed: false`. CI must never
pass `--update-baseline`.
