#!/usr/bin/env python3
"""
Local RAG Ingestion Service
Processes documents from various sources (GitHub, S3, URLs) and stores them in vector databases.
"""

import os
import sys
import time
import yaml
import tempfile
import subprocess
import logging
import requests
import json
from typing import Any

from llama_stack_client import LlamaStackClient
from llama_stack_client.types.file import File

from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document as LangChainDocument

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker

from pathlib import Path

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


class IngestionService:
    """
    Service for ingesting documents into vector databases.
    """

    def __init__(self, config_path: "str") -> "None":
        self.vector_db_ids = None
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        # Initialize Llama Stack client
        self.llama_stack_url = self.config["llamastack"]["base_url"]
        self.client = None
        self.vector_store_ids = []

        # Vector DB configuration
        self.vector_db_config = self.config["vector_db"]

        # File metadata mapping: file_id -> {original_filename, github_url, category}
        self.file_metadata = {}

        # Document converter setup
        pipeline_options = PdfPipelineOptions()
        pipeline_options.generate_picture_images = True
        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )
        self.chunker = HybridChunker()

    def wait_for_llamastack(
        self, max_retries: "int" = 2, retry_delay: "int" = 5
    ) -> "bool":
        logger.info(f"Waiting for Llama Stack at {self.llama_stack_url}...")

        for attempt in range(max_retries):
            try:
                self.client = LlamaStackClient(base_url=self.llama_stack_url)
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

    # TODO: Replace with PyGithub
    def fetch_from_github(
        self, config: "dict[str, Any]", temp_dir: "str"
    ) -> "list[str]":
        """
        fetches documents from a GitHub repository.
        """
        url = str(config["url"])
        path = str(config.get("path", ""))
        branch = str(config.get("branch", "main"))
        token = str(config.get("token", ""))

        logger.info(f"Cloning from GitHub: {url} (branch: {branch}, path: {path})")

        clone_dir = os.path.join(temp_dir, "repo")

        if token:
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

    def fetch_from_urls(self, config: "dict[str, Any]", temp_dir: "str") -> "list[str]":
        """
        fetches documents from direct URLs.
        """

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

    def process_documents(
        self, pdf_files: "list[str]", github_base_url="", category=""
    ) -> "list[File]":
        """
        processes PDF files into chunks using docling.
        """
        logger.info(f"Processing {len(pdf_files)} documents with docling...")

        llama_documents = []

        for file_path in pdf_files:
            try:
                original_filename = os.path.basename(file_path)
                logger.info(f"Processing: {original_filename}")

                # Convert PDF using docling to extract text
                result = self.converter.convert(file_path)

                # Export to markdown and clean the text
                markdown_text = result.document.export_to_markdown()
                cleaned_text = clean_text(markdown_text)

                # Create a temporary text file with cleaned content
                with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as tmp_file:
                    tmp_file.write(cleaned_text)
                    tmp_file_path = tmp_file.name

                try:
                    # Upload the cleaned text file
                    file_create_response = self.client.files.create(
                        file=Path(tmp_file_path), purpose="assistants"
                    )
                    llama_documents.append(file_create_response)

                    # Store metadata mapping file_id to original filename and GitHub URL
                    file_id = file_create_response.id
                    github_url = (
                        f"{github_base_url}/{original_filename}" if github_base_url else ""
                    )

                    self.file_metadata[file_id] = {
                        "original_filename": original_filename,
                        "github_url": github_url,
                        "category": category,
                    }
                    logger.info(f"Mapped file_id '{file_id}' -> '{original_filename}'")
                finally:
                    # Clean up temporary file
                    if os.path.exists(tmp_file_path):
                        os.unlink(tmp_file_path)

            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")

        logger.info(f"Total files processed: {len(llama_documents)}")
        return llama_documents

    def create_vector_db(
        self, vector_store_name: "str", documents: "list[File]"
    ) -> "bool":
        """
        creates vector database and inserts documents.
        """
        if not documents:
            logger.warning(f"No documents to insert for {vector_store_name}")
            return False

        logger.info(f"Creating vector database: {vector_store_name}")

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
        self, pipeline_name: "str", pipeline_config: "dict[str, Any]"
    ) -> "bool":
        """
        processes a single pipeline.
        """
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Processing pipeline: {pipeline_name}")
        logger.info(f"{'=' * 60}")

        if not pipeline_config.get("enabled", False):
            logger.info(f"Pipeline '{pipeline_name}' is disabled, skipping")
            return True

        vector_store_name = pipeline_config["vector_store_name"]
        source = pipeline_config["source"]
        source_config = pipeline_config["config"]

        category = (
            vector_store_name.split("-")[0] if vector_store_name else pipeline_name
        )

        github_base_url = ""
        if source == "GITHUB":
            github_url = source_config.get("url", "").rstrip(".git").rstrip("/")
            branch = source_config.get("branch", "main")
            path = source_config.get("path", "")
            github_base_url = f"{github_url}/blob/{branch}/{path}".rstrip("/")

        with tempfile.TemporaryDirectory() as temp_dir:
            if source == "GITHUB":
                pdf_files = self.fetch_from_github(source_config, temp_dir)
            elif source == "URL":
                pdf_files = self.fetch_from_urls(source_config, temp_dir)
            else:
                logger.error(f"Unknown source type: {source}")
                return False

            if not pdf_files:
                logger.warning(f"No PDF files found for pipeline '{pipeline_name}'")
                return False

            documents = self.process_documents(pdf_files, github_base_url, category)

            if not documents:
                logger.warning(f"No documents processed for pipeline '{pipeline_name}'")
                return False

            return self.create_vector_db(vector_store_name, documents)

    def run(self) -> "None":
        """
        runs the ingestion service.
        """
        logger.info("Starting RAG Ingestion Service")
        logger.info(f"Configuration: {os.path.abspath('ingestion-config.yaml')}")

        if not self.wait_for_llamastack():
            logger.error("Failed to connect to Llama Stack. Exiting.")
            sys.exit(1)

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

        logger.info(f"\n{'=' * 60}")
        logger.info("Ingestion Summary")
        logger.info(f"{'=' * 60}")
        logger.info(f"Total pipelines: {total}")
        logger.info(f"Successful: {successful}")
        logger.info(f"Failed: {failed}")
        logger.info(f"Skipped: {skipped}")
        logger.info(f"{'=' * 60}\n")

        if successful == 0:
            logger.warning(f"all pipeline(s) failed. Check logs for details.")
            sys.exit(1)
        elif failed > 0:
            logger.warning(f"{failed} pipeline(s) failed. Check logs for details.")
        else:
            logger.info("All pipelines completed successfully!")

        self.save_file_metadata()

    def save_file_metadata(self, output_path="rag_file_metadata.json") -> "None":
        """
        saves file metadata mapping to JSON for use by RAG service.
        """

        if not self.file_metadata:
            logger.warning("No file metadata to save")
            return

        try:
            with open(output_path, "w") as f:
                json.dump(self.file_metadata, f, indent=2)
            logger.info(
                f"Saved file metadata for {len(self.file_metadata)} files to {output_path}"
            )
        except Exception as e:
            logger.error(f"Failed to save file metadata: {e}")


class LlamaStackRetriever(BaseRetriever):
    """
    LangChain retriever that queries llama-stack vector stores.
    """

    client: "LlamaStackClient"
    vector_store_ids: "list[str]"
    max_chunks: "int" = 5

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(
        self,
        query: "str",
        *,
        run_manager=None,
    ) -> "list[LangChainDocument]":
        """
        retrieves documents from llama-stack vector stores.
        """
        logger.info(f"Retrieving documents for query: {query}")

        vector_stores = self.client.vector_stores.list() or []
        if not vector_stores:
            logger.error("No vector stores found")
            return []

        vector_db_ids = [vs.id for vs in vector_stores]
        logger.info(f"Found {len(vector_db_ids)} vector stores...")

        all_chunks = []
        for vector_db_id in vector_db_ids:
            logger.info(f"Querying vector store: {vector_db_id}")
            try:
                query_results = self.client.vector_io.query(
                    vector_db_id=vector_db_id,
                    query=query,
                    params={"max_chunks": self.max_chunks},
                )
                all_chunks.extend(query_results.chunks)
            except Exception as e:
                logger.error(f"Error querying vector store {vector_db_id}: {e}")

        logger.info(f"Retrieved {len(all_chunks)} total chunks from all vector stores")

        documents = []
        for chunk in all_chunks:
            content = clean_text(chunk.content)
            documents.append(
                LangChainDocument(
                    page_content=content,
                    metadata={"source": getattr(chunk, "document_id", "unknown")},
                )
            )

        return documents


def clean_text(text: "str") -> "str":
    """
    cleans text to handle encoding issues.
    """
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

    model = os.getenv("INFERENCE_MODEL", "vllm/qwen3-8b-fp8")

    llm = ChatOpenAI(
        model=model,
        temperature=0,
        base_url=f"{service.llama_stack_url}/v1",
        api_key="not-needed",
    )

    retriever = LlamaStackRetriever(
        client=service.client, vector_store_ids=service.vector_store_ids, max_chunks=5
    )

    rag_template = """Answer the question based only on the following context:

{context}

Question: {question}

Answer:"""

    rag_prompt = ChatPromptTemplate.from_template(rag_template)

    def format_docs(docs: "list[LangChainDocument]") -> "str":
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | rag_prompt
        | llm
        | StrOutputParser()
    )

    user_query = "describe the workspaces at FantaCo"

    current = datetime.now()
    formatted_datetime_string = current.strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"GGM making RAG chain call at {formatted_datetime_string}")

    try:
        response = rag_chain.invoke(user_query)
        current = datetime.now()
        formatted_datetime_string = current.strftime("%Y-%m-%d %H:%M:%S")
        logger.info(f"GGM returned from RAG chain call {formatted_datetime_string}")

        print(f"\n{'=' * 60}")
        print(f"Query: {user_query}")
        print(f"{'=' * 60}")
        print(f"Response: {response}")
        print(f"{'=' * 60}\n")
    except Exception as e:
        logger.error(f"Error during RAG chain execution: {e}")

    try:
        retrieved_docs = retriever._get_relevant_documents(user_query)
        context = format_docs(retrieved_docs)
        prompt = rag_prompt.format(context=context, question=user_query)
        response = llm.invoke(prompt)

        current = datetime.now()
        formatted_datetime_string = current.strftime("%Y-%m-%d %H:%M:%S")
        logger.info(
            f"GGM returned from manual context call {formatted_datetime_string}"
        )

        print(f"\n{'=' * 60}")
        print(f"Manual Context Query: {user_query}")
        print(f"{'=' * 60}")
        print(f"Response: {response.content}")
        print(f"{'=' * 60}\n")
    except Exception as e:
        logger.error(f"Error during manual context execution: {e}")
