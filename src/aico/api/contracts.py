"""Public HTTP contracts. These models are not the internal RAG/domain objects."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

AskOutcome = Literal[
    "answered",
    "insufficient_evidence",
    "clarify",
    "blocked",
]


class AskRequest(BaseModel):
    """Public /ask input. Extra body fields (including identity) are ignored."""

    model_config = ConfigDict(extra="ignore")

    question: str = Field(min_length=1)


class PublicCitation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    source_file: str


class AskResponse(BaseModel):
    """Public /ask success body. Separate from internal RAG result types."""

    model_config = ConfigDict(extra="forbid")

    request_id: str
    correlation_id: str
    outcome: AskOutcome
    answer: str | None = None
    citations: list[PublicCitation] = Field(default_factory=list)
    confidence_label: str | None = None


class ApiErrorBody(BaseModel):
    """Single public error envelope for 4xx/5xx responses."""

    model_config = ConfigDict(extra="forbid")

    code: str
    message: str
    request_id: str
    correlation_id: str
