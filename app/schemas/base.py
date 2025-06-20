from http import HTTPStatus
from typing import Generic, List, Optional, TypeVar

from pydantic import BaseModel

ResponseData = TypeVar("ResponseData")


class BaseResponse(BaseModel, Generic[ResponseData]):
    message: str
    status_code: HTTPStatus = HTTPStatus.OK
    data: Optional[ResponseData] = None

