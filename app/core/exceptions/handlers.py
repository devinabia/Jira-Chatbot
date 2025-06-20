import logging
import sys
import traceback
from http import HTTPStatus

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from .base import AppException, ErrorDetail, ErrorResponse

logger = logging.getLogger(__name__)


async def app_exception_handler(request: Request, exc: AppException) -> JSONResponse:
    error_response = exc.to_response(
        request_id=request.state.request_id,
        path=request.url.path,
        method=request.method,
    )

    logger.error(
        "Application error occurred",
        extra={
            "request_id": request.state.request_id,
            "error_code": exc.status_code,
            "path": request.url.path,
            "method": request.method,
            "error_response": error_response.model_dump(),
        },
        exc_info=True,
    )

    return JSONResponse(
        status_code=exc.status_code,
        content=error_response.model_dump(),
        headers=exc.headers,
    )


async def validation_exception_handler(request: Request, exc: ValidationError) -> JSONResponse:
    details = [
        ErrorDetail(
            field=".".join(str(loc) for loc in error["loc"] if loc != "body"),
            message=error["msg"],
        )
        for error in exc.errors()
    ]

    error_response = ErrorResponse(
        status_code=HTTPStatus.UNPROCESSABLE_ENTITY,
        message="Request validation failed",
        details=details,
        request_id=request.state.request_id,
        path=request.url.path,
        method=request.method,
    )

    return JSONResponse(
        status_code=HTTPStatus.UNPROCESSABLE_ENTITY,
        content=error_response.dict(),
        headers=getattr(exc, "headers", None),
    )


async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    error_response = ErrorResponse(
        status_code=exc.status_code,
        message=str(exc.detail),
        request_id=getattr(request.state, "request_id", "unknown"),
        path=request.url.path,
        method=request.method,
    )

    return JSONResponse(
        status_code=exc.status_code,
        content=error_response.dict(),
        headers=getattr(exc, "headers", None),
    )


async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    request_id = getattr(request.state, "request_id", "unknown")

    error_response = ErrorResponse(
        status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
        message="An unexpected error occurred",
        request_id=request_id,
        path=request.url.path,
        method=request.method,
    )

    logger.error(
        "Unhandled exception occurred",
        extra={
            "request_id": request_id,
            "path": request.url.path,
            "method": request.method,
            "traceback": "".join(traceback.format_exception(*sys.exc_info())),
        },
    )

    return JSONResponse(
        status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
        content=error_response.dict(),
        headers=getattr(exc, "headers", None),
    )


def setup_exception_handlers(app: FastAPI) -> None:
    app.add_exception_handler(AppException, app_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    app.add_exception_handler(Exception, unhandled_exception_handler)
