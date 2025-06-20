import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Dict
import re

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
import requests
from app.core.secrets import config_secrets

from slack_bolt import App
from slack_bolt.adapter.fastapi import SlackRequestHandler

from app.api.v1.routes import create_api_router
from app.core.config import settings
from app.core.config.database import (
    mongodb,
    connect_to_mongodb,
    close_mongodb_connection,
)
from app.core.exceptions import setup_exception_handlers
from app.core.middlewares import (
    RequestIDMiddleware,
    RequestLoggingMiddleware,
    ResponseTimeMiddleware,
)
from app.utilities import Cronjob

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

load_dotenv()

slack_app = App(
    token=os.environ.get("SLACK_BOT_TOKEN"),
    signing_secret=os.environ.get("SLACK_SIGNING_SECRET"),
)


def call_api(query):
    """Helper function to call your API"""
    try:
        api_endpoint = f"""{config_secrets.APP_BACKEND_URL}api/v1/bot/ask-query"""
        response = requests.post(
            api_endpoint,
            json={"query": query},
            headers={"Content-Type": "application/json"},
            timeout=30,
        )

        if response.status_code == 200:
            result = response.json()
            if result.get("status_code") == 200:
                return result.get("data", "No answer found")
            else:
                return f"❌ Error: {result.get('message', 'Unknown error occurred')}"
        else:
            return f"❌ API Error: {response.status_code}"
    except Exception as e:
        logger.error(f"Error calling API: {e}")
        return "❌ Something went wrong. Please try again."


@slack_app.event("app_mention")
def handle_app_mention(event, say):
    """Handle when bot is mentioned in channels"""
    try:
        logger.info(f"App mention received: {event}")

        text = event.get("text", "")
        user = event.get("user", "Unknown")

        user_query = re.sub(r"<@[A-Z0-9]+>", "", text).strip()

        if not user_query:
            say(
                "Please provide a question after mentioning me. Example: `@bot what is the deployment process?`"
            )
            return

        say("🔍 Searching Confluence and Jira...")

        answer = call_api(user_query)

        say(
            {
                "text": f"💡 Answer:\n\n{answer}",
                "unfurl_links": True,
                "unfurl_media": True,
            }
        )

    except Exception as e:
        logger.error(f"Error handling app mention: {e}")
        say("❌ Something went wrong. Please try again.")


@slack_app.message("")
def handle_dm_message(message, say):
    """Handle any direct message to the bot"""
    try:
        if message.get("channel_type") != "im":
            return

        logger.info(f"DM received: {message}")

        user_query = message["text"].strip()

        if not user_query:
            say("Please provide a question. Example: `what is the deployment process?`")
            return

        if user_query.lower().startswith("ask"):
            return

        say("🔍 Searching Confluence and Jira...")

        answer = call_api(user_query)

        say(f"💡 Answer:\n\n{answer}")

    except Exception as e:
        logger.error(f"Error handling DM message: {e}")
        say("❌ Something went wrong. Please try again.")


@slack_app.message("ask")
def handle_ask_message(message, say):
    """Handle messages that start with 'ask' in channels and DMs"""
    try:
        logger.info(f"Ask message received: {message}")

        user_query = message["text"].replace("ask", "", 1).strip()
        user_name = message.get("user", "Unknown")

        if not user_query:
            say(
                "Please provide a question after 'ask'. Example: `ask what is the deployment process?`"
            )
            return

        say("🔍 Searching Confluence and Jira...")

        answer = call_api(user_query)

        say(
            {
                "text": f"💡 Answer:\n\n{answer}",
                "unfurl_links": True,
                "unfurl_media": True,
            }
        )

    except Exception as e:
        logging.error(f"Error handling ask message: {e}")
        say("❌ Something went wrong. Please try again.")


@slack_app.command("/ask")
def handle_slash_command(ack, respond, command):
    """Handle /ask slash command"""
    ack()

    try:
        logger.info(f"Slash command received: {command}")

        query = command["text"].strip()
        user_name = command["user_name"]

        if not query:
            respond(
                {
                    "response_type": "ephemeral",
                    "text": "Please provide a question. Example: `/ask what is the deployment process?`",
                }
            )
            return

        answer = call_api(query)

        respond(
            {
                "response_type": "in_channel",
                "text": f"💡 *{user_name} asked:* {query}\n\n{answer}",
                "unfurl_links": True,
                "unfurl_media": True,
            }
        )

    except Exception as e:
        logging.error(f"Error handling slash command: {e}")
        respond(
            {
                "response_type": "ephemeral",
                "text": "❌ Something went wrong. Please try again.",
            }
        )


slack_handler = SlackRequestHandler(slack_app)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    logger.info("Application is starting up...")

    try:
        logger.info("Attempting to connect to MongoDB...")
        await connect_to_mongodb(
            settings.db.DB_MONGODB_URL, **settings.db.mongodb_connection_params
        )
        app.mongodb_client = mongodb.client
        app.mongodb = app.mongodb_client[settings.db.DB_MONGODB_DB_NAME]
        logger.info("Successfully connected to MongoDB")

    except Exception as e:
        logger.error(f"Failed to connect to MongoDB: {str(e)}", exc_info=True)
        raise

    try:
        await startup_tasks(app)
    except Exception as e:
        logger.error(f"Error during startup tasks: {str(e)}", exc_info=True)
        raise

    await Cronjob.start_scheduler()
    api_router = create_api_router()
    app.include_router(api_router, prefix="/api/v1")

    yield

    logger.info("Shutting down application...")

    try:
        await close_mongodb_connection()
        logger.info("MongoDB connection closed")
    except Exception as e:
        logger.error(f"Error closing MongoDB connection: {str(e)}", exc_info=True)

    try:
        await cleanup_tasks(app)
    except Exception as e:
        logger.error(f"Error during cleanup tasks: {str(e)}", exc_info=True)

    Cronjob.scheduler.shutdown()
    logger.info("Scheduler shutdown.")

    logger.info("Application shutdown complete.")


async def startup_tasks(app: FastAPI) -> None:
    """Additional startup tasks"""
    logger.info("Startup tasks completed")


async def cleanup_tasks(app: FastAPI) -> None:
    """Additional cleanup tasks"""
    logger.info("Cleanup tasks completed")


def create_application() -> FastAPI:
    """Create FastAPI application"""
    app = FastAPI(
        title=settings.app.PROJECT_NAME,
        description=settings.app.DESCRIPTION,
        version=settings.app.VERSION,
        docs_url=settings.app.DOCS_URL,
        redoc_url=settings.app.REDOC_URL,
        openapi_url=settings.app.OPENAPI_URL,
        lifespan=lifespan,
        debug=settings.app.DEBUG,
    )

    setup_middlewares(app)
    setup_exception_handlers(app)
    setup_base_routes(app)
    setup_slack_routes(app)

    return app


def setup_middlewares(app: FastAPI) -> None:
    """Setup application middlewares"""
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[str(origin) for origin in settings.app.BACKEND_CORS_ORIGINS],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.add_middleware(TrustedHostMiddleware, allowed_hosts=settings.app.ALLOWED_HOSTS)

    app.add_middleware(RequestIDMiddleware)
    app.add_middleware(RequestLoggingMiddleware)
    app.add_middleware(ResponseTimeMiddleware)


def setup_base_routes(app: FastAPI) -> None:
    """Setup base application routes"""

    @app.get("/health", tags=["Health"])
    async def health_check() -> Dict[str, Any]:
        """Health check endpoint"""
        return {
            "status": "healthy",
            "version": settings.app.VERSION,
            "environment": settings.app.ENVIRONMENT,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    @app.get("/")
    async def root() -> Dict[str, str]:
        """Root endpoint"""
        return {"message": "FastAPI server is up and running"}


def setup_slack_routes(app: FastAPI) -> None:
    """Setup Slack integration routes"""

    @app.post("/slack/events", tags=["Slack"])
    async def slack_events(request: Request):
        """Handle Slack events"""
        try:
            logger.info(f"Slack event received: {request.method} {request.url}")
            return await slack_handler.handle(request)
        except Exception as e:
            logger.error(f"Error handling Slack event: {e}")
            return JSONResponse(
                status_code=500, content={"error": "Internal server error"}
            )

    @app.get("/slack/health", tags=["Slack"])
    async def slack_health_check() -> Dict[str, str]:
        """Slack bot health check"""
        return {
            "status": "healthy",
            "bot_name": "Confluence Bot",
            "message": "Slack bot is running",
        }


app = create_application()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.app.RELOAD,
        workers=settings.app.WORKERS_COUNT,
    )

application = app
