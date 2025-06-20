# services/qdrant_service.py
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from atlassian import Confluence, Jira
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import uuid
import time
from typing import List, Dict, Any
import logging
from app.core.secrets import config_secrets


class QdrantService:
    def __init__(self):
        self.qdrant_client = QdrantClient(
            url=config_secrets.QDRANT_URL, api_key=config_secrets.QDRANT_API_KEY
        )
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2")
        self.collection_name = config_secrets.QDRANT_COLLECTION

        self.confluence_url = config_secrets.JIRA_CONFLUENCE_URL
        self.confluence_username = config_secrets.CONFLUENCE_USER
        self.confluence_token = config_secrets.CONFLUENCE_TOKEN

        self.jira_url = config_secrets.JIRA_URL
        self.jira_username = self.confluence_username
        self.jira_token = self.confluence_token

    def create_collection(self):
        """Create Qdrant collection if it doesn't exist"""
        try:
            collections = self.qdrant_client.get_collections()
            collection_names = [col.name for col in collections.collections]

            if self.collection_name not in collection_names:
                self.qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
                )
                logging.info(f"Created collection: {self.collection_name}")
            else:
                logging.info(f"Collection {self.collection_name} already exists")

        except Exception as e:
            logging.error(f"Error creating collection: {e}")
            raise

    def extract_confluence_data(self) -> List[Dict[str, Any]]:
        """Extract all data from Confluence"""
        confluence = Confluence(
            url=self.confluence_url,
            username=self.confluence_username,
            password=self.confluence_token,
            cloud=True,
        )

        documents = []
        spaces = confluence.get_all_spaces(start=0, limit=100)

        for space in spaces["results"]:
            space_key = space["key"]
            space_name = space["name"]

            try:
                pages = confluence.get_all_pages_from_space(
                    space=space_key, start=0, limit=500, expand="body.storage,version"
                )

                for page in pages:
                    try:
                        page_content = confluence.get_page_by_id(
                            page["id"], expand="body.storage"
                        )

                        if "body" in page_content and "storage" in page_content["body"]:
                            html_content = page_content["body"]["storage"]["value"]
                            soup = BeautifulSoup(html_content, "html.parser")

                            images_info = []

                            for img in soup.find_all("img"):
                                img_data = {
                                    "src": img.get("src", ""),
                                    "alt": img.get("alt", ""),
                                    "title": img.get("title", ""),
                                    "width": img.get("width", ""),
                                    "height": img.get("height", ""),
                                    "type": "embedded",
                                }
                                if img_data["src"]:
                                    images_info.append(img_data)

                            for ac_img in soup.find_all("ac:image"):
                                attachment = ac_img.find("ri:attachment")
                                if attachment:
                                    filename = attachment.get("ri:filename", "")
                                    img_data = {
                                        "src": f"/wiki/download/attachments/{page['id']}/{filename}",
                                        "alt": filename,
                                        "title": filename,
                                        "width": ac_img.get("ac:width", ""),
                                        "height": ac_img.get("ac:height", ""),
                                        "type": "attachment",
                                    }
                                    images_info.append(img_data)

                            if images_info:
                                logging.info(
                                    f"Found {len(images_info)} images in page: {page_content['title']}"
                                )

                            for script in soup(["script", "style"]):
                                script.decompose()

                            clean_text = soup.get_text()
                            lines = (line.strip() for line in clean_text.splitlines())
                            clean_text = "\n".join(line for line in lines if line)

                            if clean_text.strip():
                                page_url = (
                                    self.confluence_url
                                    + page_content["_links"]["webui"]
                                )

                                documents.append(
                                    {
                                        "id": str(uuid.uuid4()),
                                        "text": clean_text,
                                        "source": "confluence",
                                        "title": page_content["title"],
                                        "space": space_name,
                                        "space_key": space_key,
                                        "url": page_url,
                                        "page_id": page["id"],
                                        "created_date": page_content.get(
                                            "version", {}
                                        ).get("when", ""),
                                        "type": "page",
                                        "images": images_info,
                                    }
                                )

                    except Exception as e:
                        logging.warning(
                            f"Error processing page {page.get('id', 'unknown')}: {e}"
                        )
                        continue

                    time.sleep(0.1)

            except Exception as e:
                logging.warning(f"Error processing space {space_key}: {e}")
                continue

        return documents

    def extract_jira_data(self) -> List[Dict[str, Any]]:
        """Extract all data from Jira"""
        jira = Jira(
            url=self.jira_url,
            username=self.jira_username,
            password=self.jira_token,
            cloud=True,
        )

        documents = []

        try:
            projects = jira.projects()

            for project in projects:
                project_key = project["key"]
                project_name = project["name"]

                try:
                    start_at = 0
                    max_results = 100

                    while True:
                        issues = jira.jql(
                            f"project = {project_key}",
                            start=start_at,
                            limit=max_results,
                            expand="changelog",
                        )

                        if not issues["issues"]:
                            break

                        for issue in issues["issues"]:
                            try:
                                fields = issue["fields"]

                                text_content = f"Summary: {fields.get('summary', '')}\n"

                                if fields.get("description"):
                                    text_content += (
                                        f"Description: {fields['description']}\n"
                                    )

                                comments = jira.comments(issue["key"])
                                if comments:
                                    text_content += "Comments:\n"
                                    for comment in comments:
                                        text_content += f"- {comment.get('body', '')}\n"

                                if text_content.strip():
                                    issue_url = f"{self.jira_url}/browse/{issue['key']}"

                                    documents.append(
                                        {
                                            "id": str(uuid.uuid4()),
                                            "text": text_content,
                                            "source": "jira",
                                            "title": fields.get("summary", ""),
                                            "project": project_name,
                                            "project_key": project_key,
                                            "issue_key": issue["key"],
                                            "issue_type": fields.get(
                                                "issuetype", {}
                                            ).get("name", ""),
                                            "status": fields.get("status", {}).get(
                                                "name", ""
                                            ),
                                            "priority": fields.get("priority", {}).get(
                                                "name", ""
                                            ),
                                            "assignee": (
                                                fields.get("assignee", {}).get(
                                                    "displayName", ""
                                                )
                                                if fields.get("assignee")
                                                else ""
                                            ),
                                            "url": issue_url,
                                            "created_date": fields.get("created", ""),
                                            "updated_date": fields.get("updated", ""),
                                            "type": "issue",
                                        }
                                    )

                            except Exception as e:
                                logging.warning(
                                    f"Error processing issue {issue.get('key', 'unknown')}: {e}"
                                )
                                continue

                        start_at += max_results
                        time.sleep(0.1)

                except Exception as e:
                    logging.warning(f"Error processing project {project_key}: {e}")
                    continue

        except Exception as e:
            logging.error(f"Error fetching Jira projects: {e}")

        return documents

    def chunk_text(
        self, text: str, chunk_size: int = 500, overlap: int = 100
    ) -> List[str]:
        """Split text into overlapping chunks"""
        if len(text) <= chunk_size:
            return [text]

        chunks = []
        start = 0

        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]

            if end < len(text):
                last_space = chunk.rfind(" ")
                if last_space > start + chunk_size // 2:
                    end = start + last_space
                    chunk = text[start:end]

            chunks.append(chunk.strip())
            start = end - overlap

        return chunks

    def clear_collection(self):
        """Delete and recreate the collection"""
        try:
            collections = self.qdrant_client.get_collections()
            collection_names = [col.name for col in collections.collections]

            if self.collection_name in collection_names:
                self.qdrant_client.delete_collection(self.collection_name)
                logging.info(f"Deleted existing collection: {self.collection_name}")

        except Exception as e:
            logging.warning(f"Error clearing collection: {e}")

    def dump_all_data_to_qdrant(self) -> Dict[str, Any]:
        """Dump all Confluence and Jira data to Qdrant (deletes and recreates collection)"""
        try:
            logging.info("Deleting existing collection...")
            self.clear_collection()

            self.create_collection()

            logging.info("Extracting Confluence data...")
            confluence_docs = self.extract_confluence_data()

            logging.info("Extracting Jira data...")
            jira_docs = self.extract_jira_data()

            all_documents = confluence_docs + jira_docs

            if not all_documents:
                return {
                    "status": "warning",
                    "message": "No documents found to index",
                    "total_documents": 0,
                }

            points = []

            for doc in all_documents:
                chunks = self.chunk_text(doc["text"])

                for i, chunk in enumerate(chunks):
                    if chunk.strip():
                        embedding = self.encoder.encode(chunk).tolist()

                        point = PointStruct(
                            id=str(uuid.uuid4()),
                            vector=embedding,
                            payload={
                                **doc,
                                "text": chunk,
                                "chunk_index": i,
                                "total_chunks": len(chunks),
                            },
                        )
                        points.append(point)

            batch_size = 100
            total_points = len(points)

            for i in range(0, total_points, batch_size):
                batch = points[i : i + batch_size]
                self.qdrant_client.upsert(
                    collection_name=self.collection_name, points=batch
                )
                logging.info(
                    f"Uploaded batch {i//batch_size + 1}/{(total_points + batch_size - 1)//batch_size}"
                )

            return {
                "status": "success",
                "message": "Data successfully dumped to Qdrant",
                "total_documents": len(all_documents),
                "total_chunks": total_points,
                "confluence_documents": len(confluence_docs),
                "jira_documents": len(jira_docs),
                "collection_name": self.collection_name,
            }

        except Exception as e:
            logging.error(f"Error dumping data to Qdrant: {e}")
            return {"status": "error", "message": f"Failed to dump data: {str(e)}"}
