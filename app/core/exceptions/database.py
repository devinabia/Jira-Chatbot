from http import HTTPStatus
from typing import Dict, List, Optional

from .base import AppException, ErrorDetail


class DatabaseException(AppException):
    def __init__(
        self,
        message: str = "Database error occurred",
        details: Optional[List[ErrorDetail]] = None,
        status_code: int = 500,
        headers: Optional[Dict[str, str]] = None,
    ):
        super().__init__(message=message, details=details, status_code=status_code, headers=headers)


class ConnectionException(DatabaseException):
    def __init__(
        self,
        message: str = "Database connection error",
        details: Optional[List[ErrorDetail]] = None,
    ):
        super().__init__(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            message=message,
            details=details,
        )


class QueryException(DatabaseException):
    def __init__(
        self,
        message: str = "Database query error",
        details: Optional[List[ErrorDetail]] = None,
    ):
        super().__init__(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            message=message,
            details=details,
        )
