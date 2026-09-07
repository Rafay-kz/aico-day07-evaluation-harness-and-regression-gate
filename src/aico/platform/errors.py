"""Normalized gateway errors. Application code must not catch provider SDK exceptions."""

from __future__ import annotations


class GatewayError(Exception):
    """Base type for every failure that crosses the Model Gateway boundary."""

    category = "gateway"

    def __init__(self, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.retryable = retryable


class ConfigurationError(GatewayError):
    category = "configuration"


class GatewayTimeout(GatewayError):
    category = "timeout"

    def __init__(self, message: str = "model call timed out") -> None:
        super().__init__(message, retryable=True)


class RateLimitError(GatewayError):
    category = "rate_limit"

    def __init__(self, message: str = "model provider rate-limited the call") -> None:
        super().__init__(message, retryable=True)


class AuthenticationError(GatewayError):
    category = "authentication"

    def __init__(self, message: str = "model provider rejected authentication") -> None:
        super().__init__(message, retryable=False)


class BadRequestError(GatewayError):
    category = "bad_request"

    def __init__(self, message: str = "model provider rejected the request") -> None:
        super().__init__(message, retryable=False)


class ServerError(GatewayError):
    category = "server_error"

    def __init__(self, message: str = "model provider returned a server failure") -> None:
        super().__init__(message, retryable=True)


class OperationCancelled(GatewayError):
    category = "cancelled"

    def __init__(self, message: str = "model operation was cancelled") -> None:
        super().__init__(message, retryable=False)


class FallbackBlockedError(GatewayError):
    category = "fallback_blocked"

    def __init__(
        self,
        message: str | None = None,
        *,
        failed_checks: list[str],
    ) -> None:
        self.failed_checks = list(failed_checks)
        super().__init__(
            message or f"fallback is blocked by routing policy ({', '.join(self.failed_checks)})",
            retryable=False,
        )


ERROR_BY_CATEGORY: dict[str, type[GatewayError]] = {
    "timeout": GatewayTimeout,
    "rate_limit": RateLimitError,
    "authentication": AuthenticationError,
    "bad_request": BadRequestError,
    "server_error": ServerError,
    "cancelled": OperationCancelled,
    "fallback_blocked": FallbackBlockedError,
}


def error_for_category(category: str) -> GatewayError:
    if category == "fallback_blocked":
        return FallbackBlockedError(failed_checks=["policy"])
    cls = ERROR_BY_CATEGORY.get(category)
    if cls is None:
        return ServerError(f"unknown failure category {category!r}")
    return cls()
