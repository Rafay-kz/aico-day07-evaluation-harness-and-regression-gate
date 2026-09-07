"""Typed FastAPI boundary for the Day 5 grounded RAG pipeline."""

from aico.api.app import app, create_app
from aico.api.contracts import ApiErrorBody, AskRequest, AskResponse
from aico.api.dependencies import AppContainer, TrustedClaims

__all__ = [
    "ApiErrorBody",
    "AppContainer",
    "AskRequest",
    "AskResponse",
    "TrustedClaims",
    "app",
    "create_app",
]
