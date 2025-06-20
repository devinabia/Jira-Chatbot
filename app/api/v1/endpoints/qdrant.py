from fastapi import APIRouter

from app.controllers import QdrantController
from app.schemas import BaseResponse
from app.services import QdrantService


class QdrantRouter:
    def __init__(self, bot_service: QdrantService):
        self.controller = QdrantController(bot_service)
        self.router = APIRouter()
        self.setup_routes()

    def setup_routes(self):
        self.router.add_api_route(
            "/dump-data",
            self.controller.dump_data_to_qdrant,
            methods=["POST"],
            response_model=BaseResponse,
        )

        self.router.add_api_route(
            "/get-data",
            self.controller.get_qdrant_stats,
            methods=["GET"],
            response_model=BaseResponse,
        )