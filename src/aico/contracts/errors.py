"""Typed failures for the Day 4 contract boundary. Never return an unchecked dict."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class FailureCategory(str, Enum):
    PARSE = "parse"
    CONTRACT = "contract"
    SEMANTIC = "semantic"
    REPAIR_EXHAUSTED = "repair_exhausted"
    NON_REPAIRABLE = "non_repairable"
    CITATION = "citation"


@dataclass(frozen=True)
class TypedFailure:
    """Safe, structured rejection of untrusted model output."""

    category: FailureCategory
    message: str
    field_path: str | None = None
    repairable: bool = False
    repair_attempted: bool = False

    def is_schema_failure(self) -> bool:
        return self.category in {FailureCategory.PARSE, FailureCategory.CONTRACT}

    def is_semantic_failure(self) -> bool:
        return self.category == FailureCategory.SEMANTIC

    def is_citation_failure(self) -> bool:
        return self.category == FailureCategory.CITATION
