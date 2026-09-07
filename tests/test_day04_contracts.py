"""Day 4 contract/schema validation against supplied fixtures."""

from __future__ import annotations

import ast
import json
import logging
from pathlib import Path

from aico.contracts.errors import FailureCategory, TypedFailure
from aico.contracts.models import SCHEMA_DIR, SCHEMA_VERSION_V1, CitedAnswer, export_json_schemas
from aico.contracts.semantic import validate_semantics
from aico.contracts.service import ingest_cited_answer
from aico.contracts.validator import parse_and_validate_cited_answer

FIXTURE_PATH = Path("tests/fixtures/day04/structured_output_cases.json")
SRC_CONTRACTS = Path("src/aico/contracts")
UNIQUE_RAW = "UNIQUE_INVALID_MODEL_OUTPUT_DO_NOT_LOG"


def _cases() -> dict[str, dict]:
    payload = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    return {item["id"]: item for item in payload["cases"]}


def test_valid_fixture_becomes_typed_pydantic_object() -> None:
    case = _cases()["D04-01"]
    result = parse_and_validate_cited_answer(case["raw"])
    assert isinstance(result, CitedAnswer)
    assert result.schema_version == SCHEMA_VERSION_V1
    assert result.status.value == "answered"
    assert result.answer == "Supplier insurance is required."


def test_malformed_json_is_typed_parse_failure() -> None:
    case = _cases()["D04-02"]
    result = parse_and_validate_cited_answer(case["raw"])
    assert isinstance(result, TypedFailure)
    assert result.category == FailureCategory.PARSE
    assert result.is_schema_failure()


def test_markdown_wrapped_json_uses_bounded_unwrap() -> None:
    case = _cases()["D04-03"]
    result = ingest_cited_answer(case["raw"])
    assert isinstance(result, CitedAnswer)
    assert result.answer == "Policy applies."


def test_missing_required_field_is_contract_failure() -> None:
    result = parse_and_validate_cited_answer(_cases()["D04-04"]["raw"])
    assert isinstance(result, TypedFailure)
    assert result.category == FailureCategory.CONTRACT
    assert result.field_path == "answer"


def test_extra_field_is_rejected() -> None:
    result = parse_and_validate_cited_answer(_cases()["D04-05"]["raw"])
    assert isinstance(result, TypedFailure)
    assert result.category == FailureCategory.CONTRACT
    assert result.field_path == "unexpected"


def test_wrong_type_is_rejected() -> None:
    result = parse_and_validate_cited_answer(_cases()["D04-06"]["raw"])
    assert isinstance(result, TypedFailure)
    assert result.category == FailureCategory.CONTRACT
    assert result.field_path == "answer"


def test_invalid_enum_is_rejected() -> None:
    result = parse_and_validate_cited_answer(_cases()["D04-07"]["raw"])
    assert isinstance(result, TypedFailure)
    assert result.category == FailureCategory.CONTRACT
    assert result.field_path == "status"


def test_out_of_range_constraint_is_enforced() -> None:
    result = parse_and_validate_cited_answer(_cases()["D04-08"]["raw"])
    assert isinstance(result, TypedFailure)
    assert result.category == FailureCategory.CONTRACT
    assert result.field_path == "citations.0.chunk_id"


def test_committed_schema_is_generated_from_source_model() -> None:
    generated = export_json_schemas(SCHEMA_DIR)
    for name, path in generated.items():
        committed = (SCHEMA_DIR / name).read_text(encoding="utf-8")
        assert path.read_text(encoding="utf-8") == committed
        schema = json.loads(committed)
        assert "properties" in schema
        assert "schema_version" in schema["properties"]


def test_schema_version_appears_in_output_metadata() -> None:
    result = parse_and_validate_cited_answer(_cases()["D04-01"]["raw"])
    assert isinstance(result, CitedAnswer)
    dumped = result.model_dump()
    assert dumped["schema_version"] == "1.0"


def test_contract_layer_does_not_import_provider_sdks() -> None:
    offenders: list[str] = []
    for path in SRC_CONTRACTS.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                names.append(node.module)
            for name in names:
                if name == "openai" or name.startswith("openai.") or name.startswith("azure."):
                    offenders.append(f"{path} imports {name}")
    assert offenders == []


def test_invalid_output_is_not_logged_in_full(caplog) -> None:
    raw = (
        '{"schema_version":"1.0","status":"answered","answer":"'
        + UNIQUE_RAW
        + '","citations":[],"confidence_label":"medium","unexpected":true}'
    )
    with caplog.at_level(logging.INFO):
        result = ingest_cited_answer(raw)
    assert isinstance(result, TypedFailure)
    assert UNIQUE_RAW not in caplog.text


def test_schema_valid_object_is_not_a_dict() -> None:
    result = parse_and_validate_cited_answer(_cases()["D04-01"]["raw"])
    assert not isinstance(result, dict)


def test_semantic_is_not_run_inside_contract_validator() -> None:
    """D04-09 is schema-valid; contract validator must not reject it."""
    contract = parse_and_validate_cited_answer(_cases()["D04-09"]["raw"])
    assert isinstance(contract, CitedAnswer)
    semantic = validate_semantics(contract)
    assert isinstance(semantic, TypedFailure)
    assert semantic.category == FailureCategory.SEMANTIC
