# app/core/config.py
import os
from typing import List
from pydantic_settings import BaseSettings
from app.core.secrets import config_secrets

class AppSettings(BaseSettings):
    PROJECT_NAME: str = "Jira Confluence Bot"
    DESCRIPTION: str = "Jira Confluence Bot"
    VERSION: str = "1.0.0"
    ENVIRONMENT: str = "development"
    DEBUG: bool = True
    DOCS_URL: str = "/docs"
    REDOC_URL: str = "/redoc"
    OPENAPI_URL: str = "/openapi.json"
    RELOAD: bool = True
    WORKERS_COUNT: int = 1
    BACKEND_CORS_ORIGINS: List[str] = ["*"]
    ALLOWED_HOSTS: List[str] = ["*"]
    

# class DatabaseSettings(BaseSettings):
#     DB_MONGODB_URL: str = config_secrets.DB_MONGODB_URL
#     DB_MONGODB_DB_NAME: str = config_secrets.DB_MONGODB_DB_NAME
    
#     @property
#     def mongodb_connection_params(self) -> dict:
#         return {
#             "maxPoolSize": 10,
#             "minPoolSize": 1,
#             "maxIdleTimeMS": 45000,
#             "waitQueueTimeoutMS": 5000,
#             "serverSelectionTimeoutMS": 5000,
#         }


class SecuritySettings(BaseSettings):
    JWT_SECRET_KEY: str = "12345"
    JWT_ALGORITHM: str = "HS256"
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    

class Settings:
    def __init__(self):
        self.app = AppSettings()
        # self.db = DatabaseSettings()
        self.security = SecuritySettings()


settings = Settings()