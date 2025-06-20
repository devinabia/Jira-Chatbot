from http import HTTPStatus
from typing import Dict, List, Optional

from .base import AppException, ErrorDetail


class HTTPException(AppException):
    def __init__(
        self,
        status_code: int,
        message: str,
        details: Optional[List[ErrorDetail]] = None,
        headers: Optional[Dict[str, str]] = None,
    ):
        super().__init__(message=message, details=details, status_code=status_code, headers=headers)


class BadRequestException(HTTPException):
    def __init__(
        self,
        message: str = "Invalid request",
        details: Optional[List[ErrorDetail]] = None,
        headers: Optional[Dict[str, str]] = None,
    ):
        super().__init__(
            status_code=HTTPStatus.BAD_REQUEST,
            message=message,
            details=details,
            headers=headers,
        )


class UnauthorizedException(HTTPException):
    def __init__(
        self,
        message: str = "Unauthorized",
        details: Optional[List[ErrorDetail]] = None,
        headers: Optional[Dict[str, str]] = None,
    ):
        if headers is None:
            headers = {"WWW-Authenticate": "Bearer"}
        super().__init__(
            status_code=HTTPStatus.UNAUTHORIZED,
            message=message,
            details=details,
            headers=headers,
        )


class ForbiddenException(HTTPException):
    def __init__(
        self,
        message: str = "Forbidden",
        details: Optional[List[ErrorDetail]] = None,
        headers: Optional[Dict[str, str]] = None,
    ):
        super().__init__(
            status_code=HTTPStatus.FORBIDDEN,
            message=message,
            details=details,
            headers=headers,
        )


class NotFoundException(HTTPException):
    def __init__(
        self,
        message: str = "Resource not found",
        details: Optional[List[ErrorDetail]] = None,
        headers: Optional[Dict[str, str]] = None,
    ):
        super().__init__(
            status_code=HTTPStatus.NOT_FOUND,
            message=message,
            details=details,
            headers=headers,
        )


class ConflictException(HTTPException):
    def __init__(
        self,
        message: str = "Resource conflict",
        details: Optional[List[ErrorDetail]] = None,
        headers: Optional[Dict[str, str]] = None,
    ):
        super().__init__(
            status_code=HTTPStatus.CONFLICT,
            message=message,
            details=details,
            headers=headers,
        )


class ValidationException(HTTPException):
    def __init__(
        self,
        message: str = "Validation error",
        details: Optional[List[ErrorDetail]] = None,
        headers: Optional[Dict[str, str]] = None,
    ):
        super().__init__(
            status_code=HTTPStatus.UNPROCESSABLE_ENTITY,
            message=message,
            details=details,
            headers=headers,
        )


class RateLimitException(HTTPException):
    def __init__(
        self,
        message: str = "Rate limit exceeded",
        details: Optional[List[ErrorDetail]] = None,
        headers: Optional[Dict[str, str]] = None,
    ):
        super().__init__(
            status_code=HTTPStatus.TOO_MANY_REQUESTS,
            message=message,
            details=details,
            headers=headers,
        )
