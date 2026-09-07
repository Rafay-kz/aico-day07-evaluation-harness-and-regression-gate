# Day 4 validation report

Contract/schema version: **1.0** (`schema_version` on both `CitedAnswer` and `ResponseEnvelope`).

Generated schema paths (from Pydantic source models, not hand-written):

- `contracts/schema/cited_answer.v1.schema.json`
- `contracts/schema/response_envelope.v1.schema.json`

Markdown-wrapped JSON uses **one bounded unwrap**: if the entire payload is a single fenced block, the fence is stripped once, then parsed. Arbitrary prose around JSON is rejected.

Out-of-range criterion: fixture D04-08 uses an empty `chunk_id`, which violates `minLength: 1`.

## Fixture summary

| ID | Name | Stage | Result |
|---|---|---|---|
| D04-01 | valid_first_pass | valid | typed `CitedAnswer` |
| D04-02 | malformed_json | parse | typed parse failure; non-repairable (no gateway call) |
| D04-03 | markdown_wrapped_json | documented unwrap | typed `CitedAnswer` |
| D04-04 | missing_required_field | contract | typed contract failure (`answer`) |
| D04-05 | extra_field | contract | typed contract failure (`unexpected`) |
| D04-06 | wrong_type | contract | typed contract failure (`answer`) |
| D04-07 | invalid_enum | contract | typed contract failure (`status`) |
| D04-08 | out_of_range_value | contract | typed contract failure (`citations.0.chunk_id`) |
| D04-09 | semantic_answered_without_citation | semantic | schema pass, semantic fail (S1) |
| D04-10 | semantic_insufficient_with_high_confidence | semantic | schema pass, semantic fail (S2) |
| D04-11 | repairable_invalid_response | repair then valid | one gateway repair → typed success |
| D04-12 | repair_still_invalid | repair then failure | one gateway repair → `repair_exhausted` |

## Valid first-pass cases

- D04-01
- D04-03 after bounded markdown unwrap

## Contract/schema failures

- D04-02 parse
- D04-04 missing required field
- D04-05 extra field
- D04-06 wrong type
- D04-07 invalid enum
- D04-08 out-of-range / empty constrained string

## Semantic failures

- D04-09 S1 (answered with no citations)
- D04-10 S2 (insufficient_evidence with high confidence)
- Additional unit coverage: S3 duplicate `chunk_id`, S4 insufficient_evidence with citations, S5 answer/status prefix mismatch

## Repair attempts

- Allowed for contract and semantic failures only
- Cap = 1 (`RepairBudget`; a second `consume()` always fails)
- Repair calls `ModelGateway.chat` only (Day 3 boundary)
- Repaired text is run through parse → contract → semantic again

## Repair successes

- D04-11: missing `answer` repaired to a valid cited answer (fake gateway)

## Final failures

- D04-12: wrong-type repair still wrong-type → typed `repair_exhausted`
- D04-02: parse failure with a gateway present still makes **zero** repair calls

## Compatibility test result

**Pass.** `tests/fixtures/day04/existing_caller_v1.json` has no `warning` field and still validates as `ResponseEnvelope` v1 after `warning` was added as optional.

## Schema-valid but semantically invalid example (safe)

D04-09 parses as JSON and matches the cited-answer schema: `status=answered`, non-empty `answer`, `citations=[]`, `confidence_label=medium`, `schema_version=1.0`. Semantic rule S1 rejects it because an answered result must include at least one citation. The object is not rewritten into a passing value.

## Breaking changes that require a schema-version decision

These would break the existing v1 caller (demonstrated in `test_day04_compatibility.py`):

1. Removing a required field (for example `request_id`)
2. Changing a field type incompatibly (for example `request_id` from string to number)
3. Making an incompatible enum change (for example `status=partial`)
4. Making an optional field required (for example requiring `warning`)
5. Adding a new required field that old payloads omit

Adding optional `warning` does **not** require a version bump.

## Tests

Deterministic suite: `pytest -q` — Day 1–4 tests green. Failure paths use `FakeTransport`; they do not call Foundry.

Live chat, when used, goes through the Model Gateway to `gpt-4.1-mini` on the Foundry Responses API. Secrets and full invalid model payloads are not written here.
