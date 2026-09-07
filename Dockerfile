# syntax=docker/dockerfile:1
# Multi-stage image. Locked uv install. No .venv copy. No credentials.

FROM python:3.12-slim-bookworm AS builder
COPY --from=ghcr.io/astral-sh/uv:0.8.15 /uv /uvx /bin/
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev --no-editable

FROM python:3.12-slim-bookworm AS runtime
WORKDIR /app
RUN useradd --create-home --uid 10001 appuser
COPY --from=builder /app/.venv /app/.venv
COPY src ./src
COPY pyproject.toml uv.lock README.md ./
COPY config ./config
COPY contracts ./contracts
COPY data ./data
COPY evals ./evals
RUN mkdir -p artifacts/day07 && chown -R appuser:appuser /app
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app/src" \
    PYTHONDONTWRITEBYTECODE=1
USER appuser
CMD ["python", "-m", "aico.evals.day07"]
