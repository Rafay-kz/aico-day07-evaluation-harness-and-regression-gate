from __future__ import annotations

from fastapi import HTTPException
from fastapi.responses import JSONResponse

from aico.api.contracts import ApiErrorBody


class ApiError(HTTPException):
    def __init__(
        self,
        status_code: int,
        code: str,
        message: str,
        *,
        request_id: str = "",
        correlation_id: str = "",
    ) -> None:
        self.code = code
        self.safe_message = message
        self.request_id = request_id
        self.correlation_id = correlation_id
        super().__init__(status_code=status_code, detail=message)


def error_body(
    *,
    code: str,
    message: str,
    request_id: str,
    correlation_id: str,
) -> dict:
    return ApiErrorBody(
        code=code,
        message=message,
        request_id=request_id,
        correlation_id=correlation_id,
    ).model_dump()


def error_response(
    status_code: int,
    *,
    code: str,
    message: str,
    request_id: str,
    correlation_id: str,
    extra_headers: dict[str, str] | None = None,
) -> JSONResponse:
    headers = {
        "X-Request-ID": request_id,
        "X-Correlation-ID": correlation_id,
    }
    if extra_headers:
        headers.update(extra_headers)
    return JSONResponse(
        status_code=status_code,
        content=error_body(
            code=code,
            message=message,
            request_id=request_id,
            correlation_id=correlation_id,
        ),
        headers=headers,
    )
