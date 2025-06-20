# app/core/database.py
import logging
from motor.motor_asyncio import AsyncIOMotorClient
from typing import Optional
from app.core.config import settings

logger = logging.getLogger(__name__)


class MongoDB:
    client: Optional[AsyncIOMotorClient] = None


mongodb = MongoDB()


async def get_database() -> AsyncIOMotorClient:
    return mongodb.client


async def connect_to_mongodb(mongodb_url: str, **kwargs):
    """Create database connection"""
    logger.info("Connecting to MongoDB...")
    mongodb.client = AsyncIOMotorClient(mongodb_url, **kwargs)
    
    try:
        await mongodb.client.admin.command('ping')
        logger.info("Successfully connected to MongoDB")
    except Exception as e:
        logger.error(f"Failed to connect to MongoDB: {e}")
        raise


async def close_mongodb_connection():
    """Close database connection"""
    logger.info("Closing connection to MongoDB...")
    if mongodb.client:
        mongodb.client.close()
        logger.info("Successfully closed MongoDB connection")