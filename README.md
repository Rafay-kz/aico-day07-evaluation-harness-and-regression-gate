# AICO Day 7 — Evaluation Harness and Regression Gate

Day 7 adds a versioned golden dataset, deterministic and model-based evaluation, a reviewed baseline, and a CI regression gate on top of the Days 1–6 RAG service.

```text
uv run python -m aico.evals.day07
```

That command loads `evals/golden_v1.json`, evaluates required categories, writes JSON and Markdown reports, compares the candidate with thresholds and the reviewed baseline, applies safety zero tolerance, classifies failures, and exits non-zero when the gate fails.

## Setup

`uv` is required. Do not use `pip install` or `requirements.txt`.

```bash
uv sync --frozen
uv run ruff check .
uv run pytest -q
uv run python -m aico.evals.day07
```

Never commit, zip, or share `.venv` or `.env`.

Trusted identity for a local API run:

```bash
export AICO_TRUSTED_TENANT_ID=TENANT-SYN-001
export AICO_TRUSTED_USER_ID=USER-SYN-001
uv run uvicorn aico.api.app:app --reload
```

## Day 7 commands

```bash
uv run python -m aico.evals.day07
uv run python -m aico.evals.day07 --update-baseline --reviewed-by "Your Name"
uv run python -m aico.evals.day07 --weaken-retrieval
uv run python -m aico.evals.day07 --prove-controlled-regression
uv run python -m aico.evals.day07 --evaluator-transport foundry
```

`--update-baseline --reviewed-by NAME` is the only path that may rewrite `evals/baseline_v1.json`. Without `--reviewed-by`, the file is written with `reviewed: false` so a candidate cannot auto-approve itself. CI never passes those flags. Holdout is measured and reported; release metrics, thresholds, and baseline updates use train+development only.

## Docker

```bash
docker build -t aico-day07 .
docker run --rm aico-day07
```

The default evaluator transport is deterministic lab traffic through Model Gateway. CI/Docker use that path and must be labelled as fake-only evidence, not a cloud-model run. Optional live grading:

```bash
export AICO_EVAL_TRANSPORT=foundry
uv run python -m aico.evals.day07 --evaluator-transport foundry
```

## Layout

```text
src/aico/
  retrieval/           Day 1–2 remain working
  platform/            Day 3 remains working (accepted extras: fake_transport.py, retry.py, routing.py)
  contracts/           Day 4 remains working
  rag/                 Day 5 remains working
  security/            Day 5 remains working
  api/                 Day 6 remains working
  observability/       Day 6 remains working
  evals/
    day01.py
    day02.py
    day07.py
    dataset.py
    metrics.py
    groundedness.py
    regression.py
    failure_classifier.py
evals/
  golden_v1.json
  thresholds_v1.json
  baseline_v1.json
  README.md
artifacts/day07/
tests/test_day07_*.py
Dockerfile
.github/workflows/day07-quality-gate.yml
```

## Filename mapping (accepted earlier-day names)

| Path | Why it remains |
|---|---|
| `src/aico/retrieval/embed.py` | Day 2 embedding CLI |
| `src/aico/platform/fake_transport.py` | Day 3 in-process transport |
| `src/aico/platform/retry.py` | Day 3 retry policy |
| `src/aico/platform/routing.py` | Day 3 fallback checks |
| `fixtures/gateway_cases.json` | Day 3 gateway fixtures |
| `tests/day05_helpers.py` | Day 5 test helpers |
| `tests/day06_helpers.py` | Day 6 test helpers |
| `pytest.ini` | Kept beside `pyproject.toml` pytest config |

Day 1–6 deterministic tests remain part of `uv run pytest -q`.
