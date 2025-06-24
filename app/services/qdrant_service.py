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

        # Use the working credentials format
        self.confluence_base_url = f"{config_secrets.JIRA_CONFLUENCE_URL}/wiki"
        self.jira_base_url = config_secrets.JIRA_CONFLUENCE_URL
        self.auth = (
            config_secrets.CONFLUENCE_USER,
            config_secrets.CONFLUENCE_TOKEN,
        )

    def create_basic_auth_header(self, username: str, token: str) -> str:
        """Create basic auth header for manual API calls"""
        import base64

        credentials = f"{username}:{token}"
        encoded_credentials = base64.b64encode(credentials.encode()).decode()
        return f"Basic {encoded_credentials}"

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

    def test_confluence_connection(self):
        """Test Confluence connection using the working method"""
        try:
            import requests

            logging.info(
                f"Testing Confluence connection to: {self.confluence_base_url}"
            )
            logging.info(f"Using auth: {self.auth[0]}")

            # Test with the same method that works
            url = f"{self.confluence_base_url}/rest/api/space"
            response = requests.get(url, auth=self.auth, timeout=30)

            if response.status_code == 200:
                data = response.json()
                spaces_count = len(data.get("results", []))
                logging.info(
                    f"✅ Confluence connection successful! Found {spaces_count} spaces"
                )
                return True
            else:
                logging.error(
                    f"❌ Confluence connection failed: {response.status_code} - {response.text}"
                )
                return False

        except Exception as e:
            logging.error(f"❌ Confluence connection error: {type(e).__name__}: {e}")
            return False

    def test_jira_connection(self):
        """Test Jira connection using the working method"""
        try:
            import requests

            logging.info(f"Testing Jira connection to: {self.jira_base_url}")
            logging.info(f"Using auth: {self.auth[0]}")

            # Test with direct API call
            url = f"{self.jira_base_url}/rest/api/2/myself"
            response = requests.get(url, auth=self.auth, timeout=30)

            if response.status_code == 200:
                data = response.json()
                display_name = data.get("displayName", "Unknown")
                logging.info(
                    f"✅ Jira connection successful! Connected as: {display_name}"
                )
                return True
            else:
                logging.error(
                    f"❌ Jira connection failed: {response.status_code} - {response.text}"
                )
                return False

        except Exception as e:
            logging.error(f"❌ Jira connection error: {type(e).__name__}: {e}")
            return False

    def extract_confluence_data(self) -> List[Dict[str, Any]]:
        """Extract data from specific Confluence spaces: INABIA and AM"""
        try:
            import requests

            logging.info("Starting Confluence data extraction...")

            # Test connection first
            connection_test_passed = self.test_confluence_connection()
            if not connection_test_passed:
                logging.warning(
                    "Connection test failed, but proceeding with data extraction attempt..."
                )

            documents = []

            # Only process these specific spaces
            target_spaces = ["INABIA", "AM"]

            for space_key in target_spaces:
                logging.info(f"Processing space: {space_key}")

                try:
                    # Get space info first
                    space_url = f"{self.confluence_base_url}/rest/api/space/{space_key}"
                    space_response = requests.get(space_url, auth=self.auth, timeout=30)

                    if space_response.status_code != 200:
                        logging.warning(
                            f"Could not access space {space_key}: {space_response.status_code}"
                        )
                        continue

                    space_data = space_response.json()
                    space_name = space_data.get("name", space_key)
                    logging.info(f"Found space: {space_name} ({space_key})")

                    # Get all pages from space
                    url = f"{self.confluence_base_url}/rest/api/space/{space_key}/content/page?limit=1000"
                    response = requests.get(url, auth=self.auth, timeout=30)
                    response.raise_for_status()

                    pages_data = response.json()
                    pages = pages_data.get("results", [])
                    logging.info(f"Found {len(pages)} pages in space {space_name}")

                    for page in pages:
                        try:
                            page_id = page["id"]

                            # Get page content with body
                            content_url = f"{self.confluence_base_url}/rest/api/content/{page_id}?expand=body.storage,version"
                            content_response = requests.get(
                                content_url, auth=self.auth, timeout=30
                            )
                            content_response.raise_for_status()

                            page_content = content_response.json()

                            if (
                                "body" in page_content
                                and "storage" in page_content["body"]
                            ):
                                html_content = page_content["body"]["storage"]["value"]
                                soup = BeautifulSoup(html_content, "html.parser")

                                # Clean HTML content
                                for script in soup(["script", "style"]):
                                    script.decompose()

                                clean_text = soup.get_text()
                                lines = (
                                    line.strip() for line in clean_text.splitlines()
                                )
                                clean_text = "\n".join(line for line in lines if line)

                                if clean_text.strip():
                                    page_url = f"{self.confluence_base_url}/spaces/{space_key}/pages/{page_id}"

                                    documents.append(
                                        {
                                            "id": str(uuid.uuid4()),
                                            "text": clean_text,
                                            "source": "confluence",
                                            "title": page_content["title"],
                                            "space": space_name,
                                            "space_key": space_key,
                                            "url": page_url,
                                            "page_id": page_id,
                                            "created_date": page_content.get(
                                                "version", {}
                                            ).get("when", ""),
                                            "type": "page",
                                        }
                                    )

                        except Exception as e:
                            logging.warning(
                                f"Error processing page {page.get('id', 'unknown')} in space {space_name}: {e}"
                            )
                            continue

                        # Rate limiting
                        time.sleep(0.1)

                except Exception as e:
                    logging.warning(f"Error processing space {space_key}: {e}")
                    continue

            logging.info(
                f"Successfully extracted {len(documents)} documents from Confluence spaces: {', '.join(target_spaces)}"
            )
            return documents

        except Exception as e:
            logging.error(f"Error in extract_confluence_data: {type(e).__name__}: {e}")
            raise

    def extract_jira_data(self) -> List[Dict[str, Any]]:
        """Extract all data from Jira using direct API calls"""
        try:
            import requests

            logging.info("Starting Jira data extraction...")

            # Test connection first
            connection_test_passed = self.test_jira_connection()
            if not connection_test_passed:
                logging.warning(
                    "Connection test failed, but proceeding with data extraction attempt..."
                )

            documents = []

            try:
                # Get all projects
                url = f"{self.jira_base_url}/rest/api/2/project"
                response = requests.get(url, auth=self.auth, timeout=30)
                response.raise_for_status()

                projects = response.json()
                logging.info(f"Found {len(projects)} Jira projects")

                for project in projects:
                    project_key = project["key"]
                    project_name = project["name"]
                    logging.info(f"Processing project: {project_name} ({project_key})")

                    try:
                        start_at = 0
                        max_results = 100
                        project_issues_count = 0

                        while True:
                            try:
                                # Search for issues in project
                                search_url = f"{self.jira_base_url}/rest/api/2/search"
                                params = {
                                    "jql": f"project = {project_key}",
                                    "startAt": start_at,
                                    "maxResults": max_results,
                                    "expand": "changelog",
                                }

                                response = requests.get(
                                    search_url,
                                    auth=self.auth,
                                    params=params,
                                    timeout=30,
                                )
                                response.raise_for_status()

                                search_results = response.json()
                                issues = search_results.get("issues", [])

                                if not issues:
                                    break

                                for issue in issues:
                                    try:
                                        fields = issue["fields"]

                                        text_content = (
                                            f"Summary: {fields.get('summary', '')}\n"
                                        )

                                        if fields.get("description"):
                                            text_content += f"Description: {fields['description']}\n"

                                        # Get comments
                                        try:
                                            comments_url = f"{self.jira_base_url}/rest/api/2/issue/{issue['key']}/comment"
                                            comments_response = requests.get(
                                                comments_url, auth=self.auth, timeout=30
                                            )

                                            if comments_response.status_code == 200:
                                                comments_data = comments_response.json()
                                                comments = comments_data.get(
                                                    "comments", []
                                                )

                                                if comments:
                                                    text_content += "Comments:\n"
                                                    for comment in comments:
                                                        text_content += f"- {comment.get('body', '')}\n"
                                        except Exception as e:
                                            logging.debug(
                                                f"Error fetching comments for {issue['key']}: {e}"
                                            )

                                        if text_content.strip():
                                            issue_url = f"{self.jira_base_url}/browse/{issue['key']}"

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
                                                    "status": fields.get(
                                                        "status", {}
                                                    ).get("name", ""),
                                                    "priority": fields.get(
                                                        "priority", {}
                                                    ).get("name", ""),
                                                    "assignee": (
                                                        fields.get("assignee", {}).get(
                                                            "displayName", ""
                                                        )
                                                        if fields.get("assignee")
                                                        else ""
                                                    ),
                                                    "url": issue_url,
                                                    "created_date": fields.get(
                                                        "created", ""
                                                    ),
                                                    "updated_date": fields.get(
                                                        "updated", ""
                                                    ),
                                                    "type": "issue",
                                                }
                                            )
                                            project_issues_count += 1

                                    except Exception as e:
                                        logging.warning(
                                            f"Error processing issue {issue.get('key', 'unknown')}: {e}"
                                        )
                                        continue

                                start_at += max_results
                                time.sleep(0.1)  # Rate limiting

                            except Exception as e:
                                logging.warning(
                                    f"Error fetching issues for project {project_key} at offset {start_at}: {e}"
                                )
                                break

                        logging.info(
                            f"Processed {project_issues_count} issues from project {project_name}"
                        )

                    except Exception as e:
                        logging.warning(f"Error processing project {project_key}: {e}")
                        continue

            except Exception as e:
                logging.error(f"Error fetching Jira projects: {e}")
                raise

            logging.info(f"Successfully extracted {len(documents)} documents from Jira")
            return documents

        except Exception as e:
            logging.error(f"Error in extract_jira_data: {type(e).__name__}: {e}")
            raise

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

            # Try to break at word boundaries
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
            raise

    def dump_all_data_to_qdrant(self) -> Dict[str, Any]:
        """Dump only Confluence data to Qdrant (deletes and recreates collection)"""
        try:
            logging.info("Starting data dump to Qdrant...")

            # Clear and recreate collection
            logging.info("Clearing existing collection...")
            self.clear_collection()
            self.create_collection()

            # Extract data only from Confluence
            confluence_docs = []
            jira_docs = []  # Keep empty for now

            try:
                logging.info("Extracting Confluence data...")
                confluence_docs = self.extract_confluence_data()
            except Exception as e:
                logging.error(f"Failed to extract Confluence data: {e}")
                raise

            # Skip Jira extraction completely
            logging.info("Skipping Jira extraction - only processing Confluence data")

            all_documents = confluence_docs + jira_docs

            if not all_documents:
                return {
                    "status": "warning",
                    "message": "No documents found to index",
                    "total_documents": 0,
                    "confluence_documents": 0,
                    "jira_documents": 0,
                }

            # Process documents into chunks and create embeddings
            logging.info(f"Processing {len(all_documents)} documents into chunks...")
            points = []

            for doc_idx, doc in enumerate(all_documents):
                try:
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

                    if (
                        doc_idx + 1
                    ) % 10 == 0:  # More frequent logging since we have fewer docs
                        logging.info(
                            f"Processed {doc_idx + 1}/{len(all_documents)} documents"
                        )

                except Exception as e:
                    logging.warning(
                        f"Error processing document {doc.get('title', 'unknown')}: {e}"
                    )
                    continue

            # Upload to Qdrant in batches
            logging.info(f"Uploading {len(points)} chunks to Qdrant...")
            batch_size = 100
            total_points = len(points)

            for i in range(0, total_points, batch_size):
                try:
                    batch = points[i : i + batch_size]
                    self.qdrant_client.upsert(
                        collection_name=self.collection_name, points=batch
                    )
                    logging.info(
                        f"Uploaded batch {i//batch_size + 1}/{(total_points + batch_size - 1)//batch_size}"
                    )
                except Exception as e:
                    logging.error(f"Error uploading batch {i//batch_size + 1}: {e}")
                    raise

            logging.info("Data dump completed successfully!")
            return {
                "status": "success",
                "message": "Confluence data successfully dumped to Qdrant",
                "total_documents": len(all_documents),
                "total_chunks": total_points,
                "confluence_documents": len(confluence_docs),
                "jira_documents": len(jira_docs),
                "collection_name": self.collection_name,
            }

        except Exception as e:
            logging.error(f"Error dumping data to Qdrant: {type(e).__name__}: {e}")
            return {
                "status": "error",
                "message": f"Failed to dump data: {str(e)}",
                "total_documents": 0,
                "total_chunks": 0,
                "confluence_documents": 0,
                "jira_documents": 0,
            }
