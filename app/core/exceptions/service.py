from http import HTTPStatus
from typing import Dict, List, Optional

from .base import AppException, ErrorDetail


class ServiceException(AppException):
    def __init__(
        self,
        message: str = "Service error occurred",
        details: Optional[List[ErrorDetail]] = None,
        status_code: int = 503,
        headers: Optional[Dict[str, str]] = None,
    ):
        super().__init__(message=message, details=details, status_code=status_code, headers=headers)


class ExternalServiceException(ServiceException):
    def __init__(
        self,
        message: str = "External service error",
        details: Optional[List[ErrorDetail]] = None,
        service_name: Optional[str] = None,
    ):
        if service_name:
            message = f"{service_name}: {message}"
        super().__init__(
            status_code=HTTPStatus.SERVICE_UNAVAILABLE, message=message, details=details
        )
