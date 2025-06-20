from datetime import datetime, timezone
from http import HTTPStatus
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class ErrorDetail(BaseModel):
    field: Optional[str] = None
    message: str


class ErrorResponse(BaseModel):
    status_code: int
    message: str
    details: Optional[List[ErrorDetail]] = None
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    request_id: Optional[str] = None
    path: Optional[str] = None
    method: Optional[str] = None


class AppException(Exception):
    def __init__(
        self,
        message: str,
        details: Optional[List[ErrorDetail]] = None,
        status_code: int = HTTPStatus.INTERNAL_SERVER_ERROR,
        headers: Optional[Dict[str, str]] = None,
    ):
        super().__init__(message)
        self.message = message
        self.details = details or []
        self.status_code = status_code
        self.headers = headers or {}

    def to_response(
        self,
        request_id: Optional[str] = None,
        path: Optional[str] = None,
        method: Optional[str] = None,
    ) -> ErrorResponse:
        return ErrorResponse(
            status_code=self.status_code,
            message=self.message,
            details=self.details,
            request_id=request_id,
            path=path,
            method=method,
        )
