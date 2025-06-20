from bson import ObjectId
from motor.motor_asyncio import (
    AsyncIOMotorClient,
    AsyncIOMotorCollection,
    AsyncIOMotorDatabase,
)

from app.core.exceptions import BadRequestException, NotFoundException
from app.core.config import Collections
from dotenv import load_dotenv
from app.core.secrets import config_secrets
load_dotenv()
import os


class BotRepository:
    def __init__(self, client: "AsyncIOMotorClient"):
        self.client: "AsyncIOMotorClient" = client
        self.database: "AsyncIOMotorDatabase" = client[config_secrets.DB_MONGODB_DB_NAME]
        self.collection: "AsyncIOMotorCollection" = self.database[
            Collections.PROFILES_COLLECTION
        ]

    # @monitor_transaction(op="db.profile.create_user_profile")
    # async def any_db_query(self, profile, user_id):