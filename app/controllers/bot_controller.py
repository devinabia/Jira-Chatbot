from fastapi import Query
from app.schemas import BaseResponse, UserQuery
from app.services import BotService

class BotController:
    def __init__(self, bot_service: BotService):
        self.bot_service = bot_service

    async def ask_user(self, _schema: UserQuery) -> BaseResponse:
        result = await self.bot_service.query_confluence(_schema.query)
        
        return BaseResponse(
            message="success",
            data=result
        )