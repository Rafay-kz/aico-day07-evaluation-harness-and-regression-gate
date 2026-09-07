# ADR-003 — Model Routing and Fallback

## Status

Accepted

## Context

Day 2 retrieval called Microsoft Foundry for embeddings from application code. Chat will use the same models. Application code must not import the model SDK, must not hardcode deployment names, and must not leak prompts or secrets. Fallback to another route is a policy decision, not an automatic retry.

## Decision

All chat and embed calls go through `src/aico/platform/model_gateway.py`. Retrieval keeps the Day 2 `EmbeddingProvider` interface as `GatewayEmbeddingProvider`. The OpenAI SDK and `DefaultAzureCredential` live only in `src/aico/platform/foundry_adapter.py`. Tests use `FakeTransport` behind the same contract.

### Deployment aliases

Chat and embedding aliases come from `config/model-routing.yaml`. Callers request an operation, not a deployment name. Required aliases missing from YAML raise `ConfigurationError`. They are not defaulted.

### Authentication

Live calls use `DefaultAzureCredential` and `get_bearer_token_provider` with scope `https://cognitiveservices.azure.com/.default`. The endpoint is read from the environment variable named in YAML (`AICO_FOUNDRY_ENDPOINT`). No API key, bearer token or client secret is stored in source or YAML.

### Timeout and cancellation

Every request may set `timeout_seconds` and a `CancellationToken`. The gateway applies the configured timeout when the request omits one, checks cancellation before each attempt and before backoff sleep, and the fake transport can block until cancelled. The Foundry client is constructed with the same timeout.

### Retry policy

Retryable: timeout, rate limit, server error.
Non-retryable: authentication, bad request, cancellation, fallback blocked.

Ceiling: `resilience.retry.max_attempts` (3). Backoff: `base_delay_ms * 2^attempt`, capped at `max_delay_ms`. Full jitter is applied when `jitter: true` (delay is `random() * capped`). Retry stops at the ceiling. There is no infinite loop.

### Error normalization

The Foundry adapter maps SDK/HTTP failures to `GatewayTimeout`, `RateLimitError`, `AuthenticationError`, `BadRequestError` and `ServerError`. Application code catches these types.

### Routing and fallback

Fallback runs only after a retryable primary failure is exhausted, and only when `routing.fallback.enabled` is true. Before the fallback transport is called, the gateway checks provider, region, data boundary, risk class and budget. Any failed check raises `FallbackBlockedError` with `failed_checks`. Silent cross-provider or cross-boundary switching is prohibited.

### Metadata

Successful calls return `GatewayMetadata`: `model_alias`, `token_usage` (nulls if the provider omitted them), `latency_ms`, `retry_count` (failed attempts before success), `budget_status`. Prompt and completion text are not stored in metadata.

### Logging

Logger `aico.platform.model_gateway` records operation, model alias, latency, retry count, outcome, route and budget status. It does not log prompts, completions, authorization headers or secrets.

## Consequences

Retrieval no longer depends on the model SDK. Day 2 fake embeddings still produce the same deterministic vectors, so evaluation scores stay unchanged. Live Foundry access requires an approved identity (`az login` or managed identity), not a committed key. Fallback is explicit and testable with two fake transports.

## Day 2 regression evidence

`pytest -q` keeps the existing Day 1/Day 2 suite green. See `artifacts/day03/gateway_demo.md`.
