from fastapi import APIRouter, Request, HTTPException, BackgroundTasks
from app.controllers import BotController
from app.schemas import BaseResponse, UserQuery
from app.services import BotService
import logging
import requests
import asyncio

logger = logging.getLogger(__name__)


class BotRouter:
    def __init__(self, bot_service: BotService):
        self.controller = BotController(bot_service)
        self.router = APIRouter()
        self.setup_routes()

    def setup_routes(self):
        self.router.add_api_route(
            "/ask-query",
            self.handle_ask_query,
            methods=["POST"],
        )

        self.router.add_api_route(
            "/test-query",
            self.controller.ask_user,
            methods=["POST"],
        )


    async def handle_ask_query(
        self, request: Request, background_tasks: BackgroundTasks
    ):
        """
        Unified endpoint that handles both JSON (API calls) and form data (Slack slash commands)
        """
        try:
            content_type = request.headers.get("content-type", "").lower()
            logger.info(f"Received request with content-type: {content_type}")

            if "application/json" in content_type:
                logger.info("Processing JSON request")
                try:
                    body = await request.json()
                    if "query" not in body:
                        raise HTTPException(
                            status_code=422, detail="Missing 'query' field"
                        )

                    user_query = UserQuery(query=body["query"])
                    result = await self.controller.ask_user(user_query)

                    return result

                except Exception as e:
                    logger.error(f"JSON processing error: {e}")
                    raise HTTPException(status_code=400, detail=str(e))

            elif "application/x-www-form-urlencoded" in content_type:
                logger.info("Processing Slack form data request")

                try:
                    form_data = await request.form()

                    command = form_data.get("command", "")
                    text = form_data.get("text", "").strip()
                    user_name = form_data.get("user_name", "Unknown")
                    response_url = form_data.get("response_url", "")

                    logger.info(
                        f"Slack - Command: {command}, Text: {text}, User: {user_name}"
                    )
                    logger.info(f"Response URL: {response_url}")

                    if command != "/ask":
                        return {
                            "response_type": "ephemeral",
                            "text": f"Unknown command: {command}",
                        }

                    if not text:
                        return {
                            "response_type": "ephemeral",
                            "text": "Please provide a question. Example: `/ask what is the deployment process?`",
                        }

                    if not response_url:
                        logger.error("No response_url provided by Slack")
                        return {
                            "response_type": "ephemeral",
                            "text": "❌ Invalid request from Slack",
                        }

                    background_tasks.add_task(
                        self.process_slack_query_async, text, user_name, response_url
                    )

                    logger.info("Returning immediate acknowledgment to Slack")
                    return {
                        "response_type": "in_channel",
                        "text": f"🔍 *{user_name} asked:* {text}",
                    }

                except Exception as e:
                    logger.error(f"Slack processing error: {e}")
                    return {
                        "response_type": "ephemeral",
                        "text": "❌ Something went wrong. Please try again.",
                    }

            else:
                raise HTTPException(
                    status_code=415, detail=f"Unsupported content type: {content_type}"
                )

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"General error in handle_ask_query: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Internal server error")

    async def process_slack_query_async(
        self, query: str, user_name: str, response_url: str
    ):
        """
        Background task to process the actual query and send response to Slack
        """
        try:
            logger.info(f"Starting background processing for query: {query}")

            await asyncio.sleep(1)

            user_query = UserQuery(query=query)
            result = await self.controller.ask_user(user_query)

            logger.info(f"Query processed successfully, sending response to Slack")

            final_response = {
                "response_type": "in_channel",
                "replace_original": True,
                "text": f"💡 Answer:\n\n{result.data}",
            }

            response = requests.post(
                response_url,
                json=final_response,
                headers={"Content-Type": "application/json"},
                timeout=30,
            )

            if response.status_code == 200:
                logger.info(
                    f"Successfully sent final response to Slack for query: {query}"
                )
            else:
                logger.error(
                    f"Failed to send response to Slack. Status: {response.status_code}, Response: {response.text}"
                )

        except Exception as e:
            logger.error(f"Error in background task: {e}", exc_info=True)

            error_response = {
                "response_type": "in_channel",
                "replace_original": True,
                "text": f"💡 Answer:\n\n❌ Sorry, something went wrong while processing your question. Please try again.",
            }

            try:
                error_result = requests.post(
                    response_url,
                    json=error_response,
                    headers={"Content-Type": "application/json"},
                    timeout=30,
                )
                if error_result.status_code == 200:
                    logger.info("Successfully sent error response to Slack")
                else:
                    logger.error(
                        f"Failed to send error response to Slack: {error_result.status_code}"
                    )
            except Exception as send_error:
                logger.error(f"Failed to send error response to Slack: {send_error}")
