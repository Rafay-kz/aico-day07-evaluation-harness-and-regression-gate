# Day 3 gateway demonstration

Sanitized evidence only. No prompts, completions, credentials or authorization headers.

## Typed boundary

- Chat and embed: `src/aico/platform/model_gateway.py`
- Foundry adapter (SDK + `DefaultAzureCredential`): `src/aico/platform/foundry_adapter.py`
- Day 2 retrieval uses `GatewayEmbeddingProvider` / `FakeEmbeddingProvider` over the gateway

## SDK isolation

Repository search: `openai` and `azure.identity` appear only in `src/aico/platform/foundry_adapter.py`.

Covered by `tests/test_model_gateway.py`.

## Deterministic test run

```text
PYTHONPATH=src python -m pytest -q
117 passed
```

## Embed through the gateway

Fake-transport embed returns vectors plus sanitized metadata (`model_alias`, token usage as nulls when the transport has none, `latency_ms`, `retry_count=0`, `budget_status=within_budget`).

Day 2 `FakeEmbeddingProvider` still embeds via `ModelGateway`, so cache and ranking tests stay deterministic.

## Chat through the gateway

Fake-transport chat returns `gateway-chat-ok` as the result payload. That text is not copied into metadata or logs.

## Retryable failure then success

Fixture G08: transport sequence `rate_limit` then `success`. Result metadata `retry_count=1`. Backoff is bounded exponential; tests record delays without sleeping in real time.

## Retry ceiling

Fixture G09: three `server_error` outcomes, `max_attempts=3`. The call fails with normalized `server_error`. Transport is invoked three times, never more.

## Timeout

Fixture G03: transport outcome `timeout` becomes `GatewayTimeout` (`category=timeout`, retryable).

## Non-retryable failure

Fixtures G05/G06: authentication and bad_request fail on the first attempt (`chat_calls=1`).

## Blocked fallback

Fixtures G11–G14: primary `server_error`, then policy check. Region, data boundary, risk or budget mismatch raises `FallbackBlockedError` with `failed_checks`. The fallback transport is not called.

Fixture G10: all five checks pass, fallback is used, caller receives a successful chat result.

## Day 2 regression

Existing Day 1/Day 2 tests remain in the same `pytest -q` run. Fake embedding vectors are unchanged, so Day 2 evaluation scores are unchanged.
