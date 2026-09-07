# Day 7 evaluation resources

These files are the reviewed Day 7 quality baseline. They are not generated at
runtime, except `evals/baseline_v1.json` which may be rewritten only through the
explicit `--update-baseline` path.

| File | Role |
|---|---|
| `golden_v1.json` | Versioned labelled cases. Labels were set from the synthetic corpus before candidate-run inspection. |
| `thresholds_v1.json` | Machine-readable release gates. Applied as-is; never derived from holdout. |
| `baseline_v1.json` | Reviewed expected performance snapshot. Normal evaluation must not overwrite it. |

## Field mapping

`golden_v1.json` uses these case fields:

- `case_id` — stable ID
- `category` — `answerable` / `ambiguous` / `multi_chunk` / `synonym_heavy` / `unanswerable` / `adversarial`
- `split` — `train` / `development` / `holdout`
- `question` — the input (assignment “question/input”)
- `expected_sources` — document IDs such as `DOC-004`, matched only against retrieved `source_file` values
- `answerability` — `answerable` / `unanswerable` / `ambiguous`
- `critical_facts` — strings that a supported answer must include
- `prohibited_claims` — strings that must not appear in the system answer
- `expected_outcome` — `answer` / `refuse` / `clarify` / `block`

## Splits

Holdout is measured and reported separately. It is not used to choose retrieval
settings, prompts, routing, thresholds, labels, expected sources, answerability,
critical facts, or prohibited claims.

Tuning helpers (`iter_tuning_cases`) yield only `train` and `development`.

## Evaluator

Model-based groundedness uses the existing Model Gateway with evaluator prompt
version `aico-groundedness-v1`. The default lab transport is deterministic so
CI can run without Foundry credentials. Observed stability variation is therefore
expected to be zero; that is reported, not hidden.

## Baseline update

```bash
uv run python -m aico.evals.day07 --update-baseline
```

CI must never pass that flag.
