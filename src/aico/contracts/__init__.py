"""Day 4 contract boundary: untrusted model output → typed contract or typed failure."""

from aico.contracts.errors import FailureCategory, TypedFailure
from aico.contracts.models import (
    SCHEMA_VERSION_V1,
    CitedAnswer,
    Citation,
    ConfidenceLabel,
    ResponseEnvelope,
    Status,
    export_json_schemas,
)
from aico.contracts.semantic import validate_semantics
from aico.contracts.service import ingest_cited_answer, wrap_envelope
from aico.contracts.validator import parse_json, validate_cited_answer, validate_envelope

__all__ = [
    "SCHEMA_VERSION_V1",
    "CitedAnswer",
    "Citation",
    "ConfidenceLabel",
    "FailureCategory",
    "ResponseEnvelope",
    "Status",
    "TypedFailure",
    "export_json_schemas",
    "ingest_cited_answer",
    "parse_json",
    "validate_cited_answer",
    "validate_envelope",
    "validate_semantics",
    "wrap_envelope",
]
