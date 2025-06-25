from app.repositories import BotRepository
from dotenv import load_dotenv
from typing import Dict, Any
from app.core.secrets import config_secrets
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import asyncio
from openai import OpenAI
from app.utilities import Utils

load_dotenv()


class BotService:
    def __init__(self):

        genai.configure(api_key=config_secrets.GEMINI_KEY)
        self.llm = genai.GenerativeModel(
            model_name=config_secrets.GEMINI_LLM,
            generation_config={
                "temperature": 0.1,
                "top_p": 0.1,
                "max_output_tokens": 8192,
            },
        )

        self.qdrant_client = QdrantClient(
            url=config_secrets.QDRANT_URL,
            api_key=config_secrets.QDRANT_API_KEY,
            prefer_grpc=False,
            check_compatibility=False,
        )
        self.openai_client = OpenAI(api_key=config_secrets.OPENAI_API_KEY)
        # self.encoder = SentenceTransformer("all-MiniLM-L6-v2")

    async def _get_embedding(self, text: str) -> list:
        """Get embedding using OpenAI API"""
        response = await asyncio.to_thread(
            self.openai_client.embeddings.create,
            input=text,
            model="text-embedding-3-small",
            dimensions=1024,
        )
        return response.data[0].embedding

    async def search_confluence_knowledge(self, user_query: str) -> Dict[str, Any]:
        """Search knowledge base and return retrieved content"""
        try:
            if not user_query.strip():
                return {"status": "error", "message": "Query cannot be empty"}

            query_vector = await self._get_embedding(user_query)
            search_results = await asyncio.to_thread(
                self.qdrant_client.search,
                # collection_name=config_secrets.QDRANT_COLLECTION,
                collection_name=config_secrets.QDRANT_COLLECTION,
                query_vector=query_vector,
                limit=10,
                with_payload=True,
            )
            if not search_results:
                return {
                    "status": "success",
                    "retrieved_content": [],  # Return empty list instead of string
                    "sources": [],
                    "total_pages_searched": 0,
                }

            retrieved_content = []
            sources = []
            seen_urls = set()

            for result in search_results:
                payload = result.payload
                content = payload.get("text", "")

                retrieved_content.append(
                    {
                        "title": payload.get("title", "Unknown"),
                        "space": payload.get(
                            "space", payload.get("project", "Unknown")
                        ),
                        "content": content,
                        "url": payload.get("url", ""),
                        "score": result.score,
                    }
                )

                if payload.get("url") and payload["url"] not in seen_urls:
                    sources.append(
                        {
                            "title": payload.get("title", "Unknown"),
                            "url": payload["url"],
                            "space": payload.get(
                                "space", payload.get("project", "Unknown")
                            ),
                        }
                    )
                    seen_urls.add(payload["url"])

            return {
                "status": "success",
                "query": user_query,
                "retrieved_content": retrieved_content,
                "sources": sources,
                "total_pages_searched": len(retrieved_content),
            }

        except Exception as e:
            return {"status": "error", "message": f"Unexpected error: {str(e)}"}

    async def query_confluence(self, user_query: str) -> str:
        """Main method to query knowledge base and generate response"""
        try:
            search_result = await self.search_confluence_knowledge(user_query)

            if search_result["status"] == "error":
                return f"❌ Error: {search_result['message']}"

            context = self._prepare_context(search_result["retrieved_content"])

            response = await self._generate_response(
                user_query, context, search_result["sources"]
            )

            return response

        except Exception as e:
            return f"❌ Error: {str(e)}"

    def _prepare_context(self, retrieved_content) -> str:
        """Prepare context from retrieved documents"""
        # Handle case where retrieved_content is a string (no results found)
        if isinstance(retrieved_content, str) or not retrieved_content:
            return "No relevant information found in the knowledge base."

        context_parts = []
        for i, doc in enumerate(retrieved_content, 1):
            # Ensure doc is a dictionary before calling .get()
            if isinstance(doc, dict):
                score_info = (
                    f" (Relevance: {doc.get('score', 0):.3f})"
                    if doc.get("score")
                    else ""
                )
                context_part = f"""
                Document {i}: {doc['title']}{score_info} (Space: {doc['space']})
                Content: {doc['content']}
                URL: {doc['url']}
                ---"""
                context_parts.append(context_part)

        return (
            "\n".join(context_parts)
            if context_parts
            else "No relevant information found in the knowledge base."
        )

    async def _generate_response(self, user_query: str, context: str, sources) -> str:
        prompt = f"""You are a helpful assistant for the Inabia team, answering questions using internal documentation from Jira and Confluence.

        Your role:
        - Answer questions using ONLY the provided context from Jira and Confluence
        - Be accurate — if the answer isn't in the context, say so clearly and naturally
        - Keep responses to 3-4 lines maximum
        - Format for Slack using only supported markdown

        Slack Formatting Guidelines:
        - Use *bold* for emphasis (single asterisks)
        - Use _italic_ for secondary emphasis
        - Use `code` for technical terms
        - Use • for bullet points if needed
        - For links, just use the plain URL (no markdown link syntax)
        - DO NOT use headers (# ## ###) - Slack doesn't support them
        - DO NOT use [text](url) link format - Slack doesn't support it
        - Keep formatting minimal and clean

        Response Guidelines:
        - DO NOT sound like a bot — avoid phrases like "Based on the provided context"
        - Keep answers concise and direct
        - If you can't find relevant information, say:
        "I couldn't find any information about *[topic]* in your Jira or Confluence."
        - Use single asterisks (*text*) for bold, NOT double asterisks (**text**)
        - If no relevant information is found, DO NOT include any sources in your response
        - NEVER add sources when you say information wasn't found
        - Maintain a conversational and professional tone

        *Question:* {user_query}

        *Context:*
        {context}

        Provide a brief, helpful answer (max 3-4 lines) formatted for Slack. Do NOT include sources if no relevant information is found."""

        try:
            response = await asyncio.to_thread(
                self.openai_client.chat.completions.create,
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant for the Inabia team. Follow the instructions precisely and keep responses concise.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=4000,
                top_p=0.9,
            )

            response_text = response.choices[0].message.content.strip()

            no_info_phrases = [
                "couldn't find any information",
                "couldn't find information",
                "no information",
                "not found",
                "don't have information",
                "no relevant information",
            ]

            response_lower = response_text.lower()
            no_info_found = any(phrase in response_lower for phrase in no_info_phrases)

            # Only add sources if information was actually found and used
            if sources and not no_info_found:
                sources_text = "\n\n📚 *Sources:*\n"
                for source in sources[:3]:
                    try:
                        sources_text += f"• <{source['url']}|{source.get('title', 'View Document')}>\n"
                    except Exception as e:
                        sources_text += f"• {source['url']}\n"
                response_text += sources_text

            return response_text

        except Exception:
            return f"Sorry, I encountered an error while generating the response. Please try again."
