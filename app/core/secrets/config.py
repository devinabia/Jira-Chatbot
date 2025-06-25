from typing import ClassVar
import os
from dotenv import load_dotenv

load_dotenv()


class Secrets:
    OPENAI_LLM = "gpt-4o-mini"
    GEMINI_LLM = "gemini-2.5-flash-preview-04-17"
    QDRANT_COLLECTION = "jira_board"

    OPENAI_KEY = os.getenv("OPENAI_KEY")
    GEMINI_KEY = os.getenv("GEMINI_KEY")
    QDRANT_URL = os.getenv("QDRANT_URL")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
    APP_BACKEND_URL = os.getenv("APP_BACKEND_URL")

    SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")
    SLACK_SIGNING_SECRET = os.getenv("SLACK_SIGNING_SECRET")

    JIRA_URL = os.getenv("JIRA_URL")
    JIRA_CONFLUENCE_URL = os.getenv("JIRA_CONFLUENCE_URL")

    CONFLUENCE_USER = os.getenv("CONFLUENCE_USER")
    CONFLUENCE_TOKEN = os.getenv("CONFLUENCE_TOKEN")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")