from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION_V1: Literal["1.0"] = "1.0"
SCHEMA_DIR = Path(__file__).resolve().parents[3] / "contracts" / "schema"


class Status(str, Enum):
    answered = "answered"
    insufficient_evidence = "insufficient_evidence"


class ConfidenceLabel(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"


class StrictModel(BaseModel):
    """Reject unknown fields. Model output is untrusted."""

    model_config = ConfigDict(extra="forbid")


class Citation(StrictModel):
    chunk_id: str = Field(min_length=1)
    source_file: str = Field(min_length=1)


class CitedAnswer(StrictModel):
    schema_version: Literal["1.0"]
    status: Status
    answer: str = Field(min_length=1)
    citations: list[Citation]
    confidence_label: ConfidenceLabel


class ResponseEnvelope(StrictModel):
    schema_version: Literal["1.0"]
    request_id: str = Field(min_length=1)
    result: CitedAnswer
    model_alias: str = Field(min_length=1)
    trace_id: str | None = Field(default=None, min_length=1)
    warning: str | None = Field(default=None, min_length=1)


def cited_answer_json_schema() -> dict:
    return CitedAnswer.model_json_schema()


def response_envelope_json_schema() -> dict:
    return ResponseEnvelope.model_json_schema()


def export_json_schemas(directory: Path | None = None) -> dict[str, Path]:
    target = directory if directory is not None else SCHEMA_DIR
    target.mkdir(parents=True, exist_ok=True)
    mapping = {
        "cited_answer.v1.schema.json": cited_answer_json_schema(),
        "response_envelope.v1.schema.json": response_envelope_json_schema(),
    }
    written: dict[str, Path] = {}
    for name, schema in mapping.items():
        path = target / name
        path.write_text(
            json.dumps(schema, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        written[name] = path
    return written
