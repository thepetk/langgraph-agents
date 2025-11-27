#!/usr/bin/env python3
"""
RAG Service Module
Manages vector stores and provides RAG capabilities for the workflow.
Uses LlamaStackClient to access vector stores created by ingest_openai.py
"""

import os
import json
import yaml
import logging
from typing import Dict, List, Optional, Any
from llama_stack_client import LlamaStackClient

logger = logging.getLogger(__name__)


class RAGService:
    """Service for managing RAG vector stores and providing retrieval capabilities."""

    def __init__(self, llama_stack_url: str = "http://localhost:8321", 
                 ingestion_config_path: str = None,
                 file_metadata_path: str = None):
        """Initialize the RAG service."""
        self.llama_stack_url = llama_stack_url
        self.client: Optional[LlamaStackClient] = None
        self.vector_store_map: Dict[str, List[str]] = {}
        self.all_vector_store_ids: List[str] = []
        
        # Source URL mapping: category -> GitHub base URL for PDFs
        self.source_url_map: Dict[str, str] = {}
        self.ingestion_config: Dict[str, Any] = {}
        
        # File metadata: file_id -> {original_filename, github_url, category}
        self.file_metadata: Dict[str, Dict[str, str]] = {}
        
        # Load ingestion config if available
        if ingestion_config_path is None:
            # Try default locations
            for path in ['ingestion-config.yaml', '/config/ingestion-config.yaml']:
                if os.path.exists(path):
                    ingestion_config_path = path
                    break
        
        if ingestion_config_path and os.path.exists(ingestion_config_path):
            self._load_ingestion_config(ingestion_config_path)
        
        # Load file metadata if available
        if file_metadata_path is None:
            # Try default locations
            for path in ['rag_file_metadata.json', '/config/rag_file_metadata.json']:
                if os.path.exists(path):
                    file_metadata_path = path
                    break
        
        if file_metadata_path and os.path.exists(file_metadata_path):
            self._load_file_metadata(file_metadata_path)
    
    def _load_file_metadata(self, metadata_path: str):
        """Load file metadata mapping from JSON."""
        try:
            with open(metadata_path, 'r') as f:
                self.file_metadata = json.load(f)
            logger.info(f"RAG Service: Loaded metadata for {len(self.file_metadata)} files from {metadata_path}")
        except Exception as e:
            logger.warning(f"RAG Service: Could not load file metadata: {e}")
    
    def _load_ingestion_config(self, config_path: str):
        """Load ingestion config to map vector stores to source URLs."""
        try:
            with open(config_path, 'r') as f:
                self.ingestion_config = yaml.safe_load(f)
            
            # Build source URL mapping from pipelines
            pipelines = self.ingestion_config.get('pipelines', {})
            for pipeline_name, pipeline_config in pipelines.items():
                if not pipeline_config.get('enabled', False):
                    continue
                
                vector_store_name = pipeline_config.get('vector_store_name', '')
                source = pipeline_config.get('source', '')
                config = pipeline_config.get('config', {})
                
                if source == 'GITHUB':
                    # Construct GitHub blob URL for viewing files
                    github_url = config.get('url', '').rstrip('.git').rstrip('/')
                    branch = config.get('branch', 'main')
                    path = config.get('path', '')
                    
                    # GitHub blob URL format: https://github.com/{org}/{repo}/blob/{branch}/{path}
                    base_url = f"{github_url}/blob/{branch}/{path}".rstrip('/')
                    
                    # Map category to source URL
                    # Extract category from vector store name (e.g., "legal-vector-db-v1-0" -> "legal")
                    category = vector_store_name.split('-')[0] if vector_store_name else ''
                    if category:
                        self.source_url_map[category] = base_url
                        logger.info(f"RAG Service: Mapped category '{category}' to source URL: {base_url}")
            
            logger.info(f"RAG Service: Loaded source mappings for {len(self.source_url_map)} categories")
        except Exception as e:
            logger.warning(f"RAG Service: Could not load ingestion config: {e}")

    def initialize(self) -> bool:
        """Initialize the Llama Stack client and load vector stores."""
        try:
            self.client = LlamaStackClient(base_url=self.llama_stack_url)
            # Health check
            self.client.models.list()
            logger.info("RAG Service: Llama Stack client initialized successfully")
            
            # Load vector stores
            return self.load_vector_stores()
        except Exception as e:
            logger.error(f"RAG Service: Failed to initialize: {e}")
            return False

    def load_vector_stores(self) -> bool:
        """Load vector stores from Llama Stack and map them by category."""
        try:
            vector_stores = self.client.vector_stores.list() or []
            
            if not vector_stores:
                logger.warning("RAG Service: No vector stores found")
                return False

            # Map vector stores by name/category
            self.all_vector_store_ids = [vs.id for vs in vector_stores]
            
            # Create category mappings based on vector store names from ingestion-config.yaml
            for vs in vector_stores:
                vs_name_lower = vs.name.lower() if vs.name else vs.id.lower()
                vs_id = vs.id
                
                # Map to categories based on naming conventions from ingestion-config.yaml
                if "legal" in vs_name_lower:
                    if "legal" not in self.vector_store_map:
                        self.vector_store_map["legal"] = []
                    self.vector_store_map["legal"].append(vs_id)
                    logger.info(f"RAG Service: Mapped '{vs.name}' -> 'legal'")
                
                if "techsupport" in vs_name_lower or "support" in vs_name_lower:
                    if "support" not in self.vector_store_map:
                        self.vector_store_map["support"] = []
                    self.vector_store_map["support"].append(vs_id)
                    logger.info(f"RAG Service: Mapped '{vs.name}' -> 'support'")
                
                if "hr" in vs_name_lower:
                    if "hr" not in self.vector_store_map:
                        self.vector_store_map["hr"] = []
                    self.vector_store_map["hr"].append(vs_id)
                    logger.info(f"RAG Service: Mapped '{vs.name}' -> 'hr'")
                
                if "sales" in vs_name_lower:
                    if "sales" not in self.vector_store_map:
                        self.vector_store_map["sales"] = []
                    self.vector_store_map["sales"].append(vs_id)
                    logger.info(f"RAG Service: Mapped '{vs.name}' -> 'sales'")
                
                if "procurement" in vs_name_lower:
                    if "procurement" not in self.vector_store_map:
                        self.vector_store_map["procurement"] = []
                    self.vector_store_map["procurement"].append(vs_id)
                    logger.info(f"RAG Service: Mapped '{vs.name}' -> 'procurement'")

            logger.info(f"RAG Service: Loaded {len(self.all_vector_store_ids)} vector stores")
            logger.info(f"RAG Service: Categories available: {list(self.vector_store_map.keys())}")
            
            return len(self.all_vector_store_ids) > 0
        except Exception as e:
            logger.error(f"RAG Service: Failed to load vector stores: {e}")
            return False

    def get_vector_store_ids(self, category: Optional[str] = None) -> List[str]:
        """Get vector store IDs for a specific category or all stores."""
        if category:
            return self.vector_store_map.get(category, [])
        return self.all_vector_store_ids

    def get_file_search_tool(self, category: Optional[str] = None) -> Optional[Dict]:
        """
        Get file_search tool configuration for OpenAI responses API.
        This tool format works with both OpenAI SDK and Llama Stack's OpenAI-compatible endpoint.
        """
        vector_store_ids = self.get_vector_store_ids(category)
        
        if not vector_store_ids:
            logger.warning(f"RAG Service: No vector stores for category '{category}'")
            return None
        
        # This is the same format used in ingest_openai.py lines 377-382
        return {
            "type": "file_search",
            "vector_store_ids": vector_store_ids
        }

    def get_source_url(self, category: str, filename: str) -> str:
        """Construct the GitHub URL for a source file."""
        base_url = self.source_url_map.get(category, '')
        if base_url and filename:
            return f"{base_url}/{filename}"
        return ""

    def extract_sources_from_response(self, rag_response, category: str) -> List[Dict[str, str]]:
        """
        Extract source document references from a RAG response.
        Uses the file_metadata mapping to get original filenames and GitHub URLs.
        Returns a list of dicts with 'filename', 'url', and optionally 'snippet'.
        """
        sources = []
        seen_files = set()
        
        for output_item in rag_response.output:
            if hasattr(output_item, 'type') and output_item.type == 'file_search_call':
                # Extract results from file_search_call
                results = getattr(output_item, 'results', None)
                if results:
                    for result in results:
                        file_id = None
                        snippet = None
                        
                        # Try to get file_id from result
                        if hasattr(result, 'file_id'):
                            file_id = result.file_id
                        elif hasattr(result, 'id'):
                            file_id = result.id
                        elif hasattr(result, 'filename'):
                            # filename might be the file_id in Llama Stack
                            file_id = result.filename
                        elif hasattr(result, 'name'):
                            file_id = result.name
                        elif isinstance(result, dict):
                            file_id = result.get('file_id') or result.get('id') or result.get('filename') or result.get('name')
                        
                        # Try to get text snippet
                        if hasattr(result, 'text'):
                            text = result.text
                            snippet = text[:200] + '...' if len(text) > 200 else text
                        elif isinstance(result, dict) and result.get('text'):
                            text = result.get('text', '')
                            snippet = text[:200] + '...' if len(text) > 200 else text
                        
                        if file_id and file_id not in seen_files:
                            seen_files.add(file_id)
                            
                            # Look up in file_metadata for original filename and URL
                            if file_id in self.file_metadata:
                                metadata = self.file_metadata[file_id]
                                sources.append({
                                    'filename': metadata.get('original_filename', file_id),
                                    'url': metadata.get('github_url', ''),
                                    'snippet': snippet or ''
                                })
                                logger.debug(f"Found source: {metadata.get('original_filename')} -> {metadata.get('github_url')}")
                            else:
                                # Fallback: use file_id as filename
                                source_url = self.get_source_url(category, file_id)
                                sources.append({
                                    'filename': file_id,
                                    'url': source_url,
                                    'snippet': snippet or ''
                                })
        
        # If we couldn't extract from results, try to get files from the vector store
        if not sources:
            sources = self._get_files_from_vector_store(category)
        
        return sources

    def _get_files_from_vector_store(self, category: str) -> List[Dict[str, str]]:
        """Fallback: get all files from the vector store for a category."""
        sources = []
        seen_files = set()
        vector_store_ids = self.get_vector_store_ids(category)
        
        for vs_id in vector_store_ids:
            try:
                # Try to list files in the vector store
                files = self.client.vector_stores.files.list(vector_store_id=vs_id)
                if files:
                    for file_info in files:
                        file_id = getattr(file_info, 'id', None) or getattr(file_info, 'file_id', None)
                        
                        if file_id and file_id not in seen_files:
                            seen_files.add(file_id)
                            
                            # Look up in file_metadata
                            if file_id in self.file_metadata:
                                metadata = self.file_metadata[file_id]
                                sources.append({
                                    'filename': metadata.get('original_filename', file_id),
                                    'url': metadata.get('github_url', ''),
                                    'snippet': ''
                                })
                            else:
                                # Fallback
                                filename = getattr(file_info, 'filename', None) or file_id
                                source_url = self.get_source_url(category, filename)
                                sources.append({
                                    'filename': filename,
                                    'url': source_url,
                                    'snippet': ''
                                })
            except Exception as e:
                logger.debug(f"Could not list files for vector store {vs_id}: {e}")
        
        return sources

    def query_vectors_directly(self, query: str, category: Optional[str] = None, max_chunks: int = 5) -> str:
        """
        Alternative RAG approach: Query vector stores directly and build context.
        This is the second approach shown in ingest_openai.py (lines 420-444).
        Returns context string that can be prepended to prompts.
        """
        vector_store_ids = self.get_vector_store_ids(category)
        
        if not vector_store_ids:
            logger.warning(f"RAG Service: No vector stores for category '{category}'")
            return ""
        
        all_chunks = []
        for vector_db_id in vector_store_ids:
            try:
                query_results = self.client.vector_io.query(
                    vector_db_id=vector_db_id, 
                    query=query, 
                    params={"max_chunks": max_chunks}
                )
                all_chunks.extend(query_results.chunks)
            except Exception as e:
                logger.error(f"RAG Service: Error querying {vector_db_id}: {e}")
        
        if not all_chunks:
            return ""
        
        # Build context from chunks
        context = "\n\n".join([chunk.content for chunk in all_chunks])
        logger.info(f"RAG Service: Retrieved {len(all_chunks)} chunks for query")
        
        return context

