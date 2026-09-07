# Day 6 Sanitized Trace Summary

## Request

- Request ID: `req-trace-lab-001`
- Correlation ID: `corr-trace-lab-001`
- Result category: `answered`
- Total latency: recorded on the `api` span (synthetic FakeTransport; sub-millisecond to low tens of ms)

## Trace Stages

| Stage | Span / Operation | Parent | Duration | Sanitized Attributes |
|---|---|---|---:|---|
| API | `api` | root | request latency | `aico.request_id`, `aico.correlation_id` |
| Policy/Input | `policy` | `api` | policy latency | same correlation id; outcome is an allow/clarify/block category |
| Retrieval | `retrieval` | `api` | retrieval latency | same correlation id; cache miss for in-memory `FixedRetriever` |
| Model Gateway | `model_gateway` | `api` | gateway latency | same correlation id; model alias; retry count; token totals when present |
| Validation | `validation` | `api` | validation latency | same correlation id; contract + citation membership |
| Response Composition | `response_composition` | `api` | mapping latency | same correlation id; public `AskResponse` only |

## Operational Metrics Observed

- Retrieval latency: recorded (`retrieval_latency_ms`)
- Model latency: recorded (`gateway_latency_ms`)
- Token usage: recorded from gateway metadata on the metered fake transport (input/output/total)
- Retry count: `0` on the successful fake path
- Cache hit/miss: miss (`FixedRetriever` has no persistent vector cache)

## Redaction Check

Confirm the exported trace does **not** contain:

- [x] raw user question/prompt
- [x] retrieved evidence text
- [x] raw model completion
- [x] authorization claims/header
- [x] secrets/tokens

## Notes

This artifact is from a deterministic fake `/ask` (no live Foundry). Persistent Day 2 embedding-cache hits are not part of this in-memory retriever path; cache is reported as a miss. Token counts come from the fake transport's sanitized `TokenUsage`, not from a provider response body.
