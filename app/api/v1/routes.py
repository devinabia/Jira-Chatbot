from fastapi import APIRouter

from app.core.config.database import mongodb
from app.repositories import (
    BotRepository,
)
from app.services import (
    BotService,
    QdrantService
)

from .endpoints import (
    BotRouter,
    QdrantRouter
)


def create_api_router() -> APIRouter:
    api_router = APIRouter()

    bot_service = BotService()

    profile_router = BotRouter(bot_service)


    qdrant_service = QdrantService()
    qdrant_router = QdrantRouter(qdrant_service)



    api_router.include_router(profile_router.router, prefix="/bot", tags=["Bot"])
    api_router.include_router(qdrant_router.router, prefix="/qdrant", tags=["Qdrant"])


    return api_router
