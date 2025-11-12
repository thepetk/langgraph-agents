#!/usr/bin/env python3
"""
Local RAG Ingestion Service
Processes documents from various sources (GitHub, S3, URLs) and stores them in vector databases.
"""

import os
import sys
import time
import uuid
import yaml
import tempfile
import subprocess
from typing import List, Dict, Any
import logging

from llama_stack_client import LlamaStackClient, Agent, AgentEventLogger
from llama_stack_client.types import Document as LlamaStackDocument
from llama_stack_client.types.vector_stores.vector_store_file import VectorStoreFile
from llama_stack_client.types.file import File

# Import docling for document processing
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from docling_core.types.doc.labels import DocItemLabel

from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


class IngestionService:
    """Service for ingesting documents into vector databases."""

    def __init__(self, config_path: str):
        """Initialize the ingestion service with configuration."""
        self.vector_db_ids = None
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        # Initialize Llama Stack client
        self.llama_stack_url = self.config["llamastack"]["base_url"]
        self.client = None
        self.vector_store_ids = []

        # Vector DB configuration
        self.vector_db_config = self.config["vector_db"]

        # Document converter setup
        pipeline_options = PdfPipelineOptions()
        pipeline_options.generate_picture_images = True
        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )
        self.chunker = HybridChunker()

    def wait_for_llamastack(self, max_retries: int = 2, retry_delay: int = 5):
        """Wait for Llama Stack to be ready."""
        logger.info(f"Waiting for Llama Stack at {self.llama_stack_url}...")

        for attempt in range(max_retries):
            try:
                self.client = LlamaStackClient(base_url=self.llama_stack_url)
                # Try to list models as a health check
                self.client.models.list()
                logger.info("Llama Stack is ready!")
                return True
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.info(
                        f"Attempt {attempt + 1}/{max_retries}: Llama Stack not ready yet. Retrying in {retry_delay}s..."
                    )
                    time.sleep(retry_delay)
                else:
                    logger.error(
                        f"Failed to connect to Llama Stack after {max_retries} attempts: {e}"
                    )
                    return False

        return False

    def fetch_from_github(self, config: Dict[str, Any], temp_dir: str) -> List[str]:
        """Fetch documents from a GitHub repository."""
        url = config["url"]
        path = config.get("path", "")
        branch = config.get("branch", "main")
        token = config.get("token", "")

        logger.info(f"Cloning from GitHub: {url} (branch: {branch}, path: {path})")

        clone_dir = os.path.join(temp_dir, "repo")

        # Prepare git clone command
        if token:
            # Insert token into URL for private repos
            auth_url = url.replace("https://", f"https://{token}@")
            cmd = [
                "git",
                "clone",
                "--depth",
                "1",
                "--branch",
                branch,
                auth_url,
                clone_dir,
            ]
        else:
            cmd = ["git", "clone", "--depth", "1", "--branch", branch, url, clone_dir]

        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"Git clone failed: {e.stderr}")
            return []

        # Get the target directory
        target_dir = os.path.join(clone_dir, path) if path else clone_dir

        if not os.path.exists(target_dir):
            logger.error(f"Path {path} not found in repository")
            return []

        # Find all PDF files
        pdf_files = []
        for root, _, files in os.walk(target_dir):
            for file in files:
                if file.lower().endswith(".pdf"):
                    pdf_files.append(os.path.join(root, file))

        logger.info(f"Found {len(pdf_files)} PDF files in {target_dir}")
        return pdf_files

    def fetch_from_urls(self, config: Dict[str, Any], temp_dir: str) -> List[str]:
        """Fetch documents from direct URLs."""
        import requests

        urls = config.get("urls", [])
        logger.info(f"Fetching {len(urls)} documents from URLs")

        download_dir = os.path.join(temp_dir, "url_files")
        os.makedirs(download_dir, exist_ok=True)

        pdf_files = []
        for url in urls:
            try:
                filename = os.path.basename(url.split("?")[0])  # Remove query params
                if not filename.lower().endswith(".pdf"):
                    filename += ".pdf"

                file_path = os.path.join(download_dir, filename)

                logger.info(f"Downloading: {url}")
                response = requests.get(url, timeout=60)
                response.raise_for_status()

                with open(file_path, "wb") as f:
                    f.write(response.content)

                pdf_files.append(file_path)

            except Exception as e:
                logger.error(f"Error downloading {url}: {e}")

        logger.info(f"Downloaded {len(pdf_files)} PDF files from URLs")
        return pdf_files

    def process_documents(self, pdf_files: List[str]) -> List[File]:
        """Process PDF files into chunks using docling."""
        logger.info(f"Processing {len(pdf_files)} documents with docling...")

        llama_documents = []
        doc_id = 0

        for file_path in pdf_files:
            try:
                logger.info(f"Processing: {os.path.basename(file_path)}")

                file_create_response = self.client.files.create(
                    file=Path(file_path), purpose="assistants"
                )
                llama_documents.append(file_create_response)
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")

        logger.info(f"Total chunks created: {len(llama_documents)}")
        return llama_documents

    def get_provider_id(self) -> str:
        """Get the provider ID for the vector database."""
        providers = self.client.providers.list()
        for provider in providers:
            if provider.api == "vector_io":
                return provider.provider_id
        return None

    def create_vector_db(self, vector_store_name: str, documents: List[File]) -> bool:
        """Create vector database and insert documents."""
        if not documents:
            logger.warning(f"No documents to insert for {vector_store_name}")
            return False

        logger.info(f"Creating vector database: {vector_store_name}")

        # Register vector database
        resp_vector_db_id = ""
        try:
            vector_store = self.client.vector_stores.create(name=vector_store_name)

            self.vector_store_ids.append(vector_store.id)

        except Exception as e:
            error_msg = str(e)
            if "already exists" in error_msg.lower():
                logger.info(
                    f"Vector DB '{vector_store_name}' already exists, continuing..."
                )
            else:
                logger.error(f"Failed to register vector DB '{vector_store_name}': {e}")
                return False

        # Insert documents
        try:
            logger.info(f"Inserting {len(documents)}  into vector store...")
            for doc in documents:
                file_ingest_response = self.client.vector_stores.files.create(
                    vector_store_id=vector_store.id,
                    file_id=doc.id,
                )
                logger.info(
                    f"✓ Successfully inserted documents into '{vector_store_name}' with resp '{file_ingest_response}'"
                )
            return True

        except Exception as e:
            logger.error(f"Error inserting documents into '{vector_store_name}': {e}")
            return False

    def process_pipeline(
        self, pipeline_name: str, pipeline_config: Dict[str, Any]
    ) -> bool:
        """Process a single pipeline."""
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Processing pipeline: {pipeline_name}")
        logger.info(f"{'=' * 60}")

        if not pipeline_config.get("enabled", False):
            logger.info(f"Pipeline '{pipeline_name}' is disabled, skipping")
            return True

        vector_store_name = pipeline_config["vector_store_name"]
        source = pipeline_config["source"]
        source_config = pipeline_config["config"]

        # Create temporary directory for this pipeline
        with tempfile.TemporaryDirectory() as temp_dir:
            # Fetch documents based on source type
            if source == "GITHUB":
                pdf_files = self.fetch_from_github(source_config, temp_dir)
            elif source == "S3":
                pdf_files = self.fetch_from_s3(source_config, temp_dir)
            elif source == "URL":
                pdf_files = self.fetch_from_urls(source_config, temp_dir)
            else:
                logger.error(f"Unknown source type: {source}")
                return False

            if not pdf_files:
                logger.warning(f"No PDF files found for pipeline '{pipeline_name}'")
                return False

            # Process documents
            documents = self.process_documents(pdf_files)

            if not documents:
                logger.warning(f"No documents processed for pipeline '{pipeline_name}'")
                return False

            # Create vector database and insert documents
            return self.create_vector_db(vector_store_name, documents)

    def run(self):
        """Run the ingestion service."""
        logger.info("Starting RAG Ingestion Service")
        logger.info(f"Configuration: {os.path.abspath('ingestion-config.yaml')}")

        # Wait for Llama Stack to be ready
        if not self.wait_for_llamastack():
            logger.error("Failed to connect to Llama Stack. Exiting.")
            sys.exit(1)

        # Process all enabled pipelines
        pipelines = self.config.get("pipelines", {})
        total = len(pipelines)
        successful = 0
        failed = 0
        skipped = 0

        for pipeline_name, pipeline_config in pipelines.items():
            if not pipeline_config.get("enabled", False):
                skipped += 1
                continue

            try:
                if self.process_pipeline(pipeline_name, pipeline_config):
                    successful += 1
                else:
                    failed += 1
            except Exception as e:
                logger.error(
                    f"Unexpected error processing pipeline '{pipeline_name}': {e}"
                )
                failed += 1

        # Summary
        logger.info(f"\n{'=' * 60}")
        logger.info("Ingestion Summary")
        logger.info(f"{'=' * 60}")
        logger.info(f"Total pipelines: {total}")
        logger.info(f"Successful: {successful}")
        logger.info(f"Failed: {failed}")
        logger.info(f"Skipped: {skipped}")
        logger.info(f"{'=' * 60}\n")

        if failed > 0:
            logger.warning(f"{failed} pipeline(s) failed. Check logs for details.")
            # sys.exit(1)
        else:
            logger.info("All pipelines completed successfully!")
            # sys.exit(0)


def clean_text(text):
    """clean text to handle encoding issues."""
    replacements = {
        "\u2013": "-",
        "\u2014": "--",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2026": "...",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    return text.encode("ascii", "ignore").decode("ascii")


if __name__ == "__main__":
    config_file = os.getenv("INGESTION_CONFIG", "/config/ingestion-config.yaml")

    if not os.path.exists(config_file):
        logger.error(f"Configuration file not found: {config_file}")
        sys.exit(1)

    service = IngestionService(config_file)
    service.run()

    model = os.getenv("INFERENCE_MODEL", "ollama/llama3.2:3b")
    from datetime import datetime

    # Query vector stores directly
    user_query = "describe the workspaces at FantaCo"
    logger.info(f"Querying vector stores with: {user_query}")

    # Get vector_db_ids from vector_stores.list()
    vector_stores = service.client.vector_stores.list() or []
    if not vector_stores:
        logger.error("No vector stores found")
        sys.exit(1)

    vector_db_ids = [vs.id for vs in vector_stores]
    logger.info(f"Found vector stores: {vector_db_ids}")

    # Query the vector stores (query each one separately)
    all_chunks = []
    for vector_db_id in vector_db_ids:
        logger.info(f"Querying vector store: {vector_db_id}")
        query_results = service.client.vector_io.query(
            vector_db_id=vector_db_id, query=user_query, params={"max_chunks": 5}
        )
        all_chunks.extend(query_results.chunks)

    logger.info(f"Retrieved {len(all_chunks)} total chunks from all vector stores")

    # Query the vector stores (query each one separately)
    chunks_text = [clean_text(chunk.content) for chunk in all_chunks]
    context = "\n\n".join(chunks_text)

    # Build context from query results
    context = "\n\n".join([chunk.content for chunk in all_chunks])

    # Create prompt with context
    prompt = f"""Context from documents:
{context}

Question: {user_query}

Please answer the question based on the context provided above."""

    current = datetime.now()
    formatted_datetime_string = current.strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"GGM making response call {formatted_datetime_string}")

    rag_response = service.client.responses.create(
        model=model,
        input=prompt,
    )

    current = datetime.now()
    formatted_datetime_string = current.strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"GGM returned from response call {formatted_datetime_string}")

    for i, output_item in enumerate(rag_response.output):
        if len(rag_response.output) > 1:
            print(f"\n--- Output Item {i + 1} ---")
        print(f"Output type: {output_item.type}")

        if hasattr(output_item, "content"):
            if isinstance(output_item.content, list):
                for content in output_item.content:
                    if hasattr(content, "text"):
                        print(f"Response: {content.text}")
            else:
                print(f"Response: {output_item.content}")
        elif hasattr(output_item, "text"):
            print(f"Response: {output_item.text}")
