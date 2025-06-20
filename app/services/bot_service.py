from app.repositories import BotRepository
from dotenv import load_dotenv
from typing import Dict, Any
from app.core.secrets import config_secrets
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import asyncio

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

        self.encoder = SentenceTransformer("all-MiniLM-L6-v2")

    async def search_confluence_knowledge(self, user_query: str) -> Dict[str, Any]:
        """Search knowledge base and return retrieved content"""
        try:
            if not user_query.strip():
                return {"status": "error", "message": "Query cannot be empty"}

            query_vector = await asyncio.to_thread(self.encoder.encode, user_query)
            query_vector = query_vector.tolist()

            search_results = await asyncio.to_thread(
                self.qdrant_client.search,
                collection_name="jira_board",
                query_vector=query_vector,
                limit=5,
                with_payload=True,
            )

            if not search_results:
                return {
                    "status": "success",
                    "retrieved_content": "No relevant information found.",
                    "sources": [],
                    "total_pages_searched": 0,
                }

            retrieved_content = []
            sources = []
            seen_urls = set()

            for result in search_results:
                payload = result.payload
                content_with_images = payload.get("text", "")

                if payload.get("images") and len(payload["images"]) > 0:
                    image_context = "\n\n🖼️ VISUAL CONTENT AVAILABLE:\n"
                    for i, img in enumerate(payload["images"], 1):
                        image_context += f"{i}. "

                        if img.get("alt"):
                            image_context += f"Image: {img['alt']}"
                        elif img.get("title"):
                            image_context += f"Image: {img['title']}"
                        else:
                            image_context += "Image (no description available)"

                        if img.get("src"):
                            img_url = img["src"]
                            if img_url.startswith("/"):
                                img_url = f"{config_secrets.JIRA_URL}{img_url}"
                            elif img_url.startswith("http"):
                                pass
                            else:
                                img_url = (
                                    f"{config_secrets.JIRA_CONFLUENCE_URL}{img_url}"
                                )

                            image_context += f"\n   📎 Image URL: {img_url}"

                        if img.get("width") or img.get("height"):
                            image_context += f"\n   📐 Size: {img.get('width', '?')}x{img.get('height', '?')}"

                        image_context += "\n\n"

                    image_context += "💡 TIP: Click the image URLs above to view the actual diagrams/images.\n"
                    content_with_images += image_context

                retrieved_content.append(
                    {
                        "title": payload.get("title", "Unknown"),
                        "space": payload.get(
                            "space", payload.get("project", "Unknown")
                        ),
                        "content": content_with_images,
                        "url": payload.get("url", ""),
                        "images": payload.get("images", []),
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
        if not retrieved_content:
            return "No relevant information found in the knowledge base."

        context_parts = []
        for i, doc in enumerate(retrieved_content, 1):
            score_info = (
                f" (Relevance: {doc.get('score', 0):.3f})" if doc.get("score") else ""
            )
            context_part = f"""
                Document {i}: {doc['title']}{score_info} (Space: {doc['space']})
                Content: {doc['content']}
                URL: {doc['url']}
                ---"""
            context_parts.append(context_part)

        return "\n".join(context_parts)

    async def _generate_response(self, user_query: str, context: str, sources) -> str:

        prompt = f"""You are a helpful assistant for the Inabia team, answering questions using internal documentation from **Jira** and **Confluence**.

        Your role:
        - Answer questions using ONLY the provided context from Jira and Confluence
        - Be accurate — if the answer isn't in the context, say so clearly and naturally  
        - Include source references when available
        - Mention visual content and include image URLs if applicable
        - Use markdown formatting for better readability
        - Keep responses concise but complete

        Guidelines:
        - DO NOT sound like a bot — avoid phrases like “Based on the provided context”
        - If you can't find relevant information, say:
        “I couldn’t find any information about **[topic]** in your Jira or Confluence.”
        - Reference document names or ticket IDs when possible
        - Use bullet points and formatting for clear, scannable responses
        - Maintain a conversational and professional tone

        Use the following context from Jira and Confluence to answer this question:

        **Question:** {user_query}

        **Context:**
        {context}

        Please provide a helpful answer based on the above context. If images are mentioned, include their URLs. Format your response with markdown."""

        try:
            response = await asyncio.to_thread(self.llm.generate_content, prompt)

            response_text = response.text

            if sources:
                sources_text = "\n\n📚 **Sources:**\n"
                for source in sources[:3]:
                    sources_text += f"• [{source['title']}]({source['url']}) (Space: {source['space']})\n"
                response_text += sources_text

            return response_text

        except Exception as e:
            return f"Error generating response: {str(e)}"
