from fastapi import Query, HTTPException
from app.schemas import BaseResponse, UserQuery
from app.services import BotService


class BotController:
    def __init__(self, bot_service: BotService):
        self.bot_service = bot_service

    async def ask_user(self, _schema: UserQuery) -> BaseResponse:
        try:
            result = await self.bot_service.query_confluence(_schema.query)

            return BaseResponse(message="success", data=result)
        except Exception as e:
            print(e)
            return BaseResponse(
                message="error",
                data=None,
                error=str(e),
            )
