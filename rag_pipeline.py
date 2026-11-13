#!/usr/bin/env python3
"""
RAG Pipeline using ChromaDB and sentence-transformers.
"""

import logging
import re
from datetime import datetime
from typing import Dict, List, Optional

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from document_processor import DocumentProcessor

logger = logging.getLogger(__name__)


class RAGPipeline:
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        chroma_path: str = "chroma_db",
        collection_name: str = "kb_documents_v2",
    ):
        self.model_name = model_name
        self.chroma_path = chroma_path
        self.collection_name = collection_name
        self.embedding_model = None

        self.client = chromadb.PersistentClient(
            path=chroma_path,
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self._get_or_create_collection()
        self.doc_processor = DocumentProcessor()
        logger.info("RAG Pipeline initialized (embedding model loads on demand)")

    def _get_or_create_collection(self):
        """Get or create collection, handling schema migration issues."""
        try:
            return self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "Knowledge base documents"},
            )
        except Exception as e:
            msg = str(e)
            logger.warning("Collection create failed, attempting recovery: %s", msg)
            # Try to delete and recreate if it's a schema/config issue
            try:
                self.client.delete_collection(name=self.collection_name)
            except Exception:
                pass
            return self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "Knowledge base documents"},
            )

    def _load_embedding_model(self, model_name: str):
        """Load sentence-transformer model with fallbacks for CPU/meta-tensor issues."""
        import os

        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        errors = []

        attempts = [
            lambda: SentenceTransformer(model_name, device="cpu"),
            lambda: SentenceTransformer(
                model_name,
                device="cpu",
                model_kwargs={"low_cpu_mem_usage": False},
            ),
            lambda: SentenceTransformer(
                model_name,
                device="cpu",
                model_kwargs={"low_cpu_mem_usage": False},
                tokenizer_kwargs={"use_fast": True},
            ),
        ]
        for attempt in attempts:
            try:
                return attempt()
            except Exception as e:
                errors.append(str(e))
                logger.warning("Embedding load attempt failed: %s", e)

        raise RuntimeError(
            "Failed to load embedding model. "
            + (" | ".join(errors[-2:]) if errors else "unknown error")
        )


    def _get_embedding_model(self):
        if self.embedding_model is None:
            logger.info("Loading embedding model: %s", self.model_name)
            self.embedding_model = self._load_embedding_model(self.model_name)
        return self.embedding_model

    @staticmethod
    def _safe_id(value: str) -> str:
        return re.sub(r"[^A-Za-z0-9_-]+", "_", value)

    def chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Split text into overlapping chunks with forward progress."""
        text = (text or "").strip()
        if not text:
            return []
        if len(text) <= chunk_size:
            return [text]

        chunks: List[str] = []
        start = 0
        text_len = len(text)
        while start < text_len:
            end = min(start + chunk_size, text_len)
            if end < text_len:
                sentence_end = max(
                    text.rfind(". ", start, end),
                    text.rfind("? ", start, end),
                    text.rfind("! ", start, end),
                    text.rfind("\n", start, end),
                )
                if sentence_end > start + chunk_size // 2:
                    end = sentence_end + 1
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end >= text_len:
                break
            next_start = end - overlap
            start = next_start if next_start > start else end
        return chunks

    def _delete_file_chunks(self, file_id: str):
        try:
            self.collection.delete(where={"file_id": file_id})
        except Exception:
            pass

    def add_document(
        self,
        content: str,
        metadata: Optional[Dict] = None,
        chunk: bool = True,
        replace_existing: bool = False,
    ) -> bool:
        """Add a document (optionally chunked) to the collection."""
        try:
            content = (content or "").strip()
            if not content:
                return False

            metadata = dict(metadata or {})
            file_id = str(
                metadata.get("file_id")
                or metadata.get("filename")
                or metadata.get("source")
                or f"doc_{int(datetime.now().timestamp() * 1000)}"
            )
            file_name = metadata.get("file_name") or metadata.get("filename") or file_id

            if replace_existing:
                self._delete_file_chunks(file_id)

            chunks = self.chunk_text(content) if chunk else [content]
            if not chunks:
                return False

            documents, metadatas, ids = [], [], []
            for i, chunk_text in enumerate(chunks):
                chunk_id = f"{self._safe_id(file_id)}_{i}"
                documents.append(chunk_text)
                metadatas.append(
                    {
                        **metadata,
                        "file_id": file_id,
                        "file_name": file_name,
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                        "char_count": len(chunk_text),
                        "word_count": len(chunk_text.split()),
                        "timestamp": datetime.now().isoformat(),
                    }
                )
                ids.append(chunk_id)

            embeddings = self._get_embedding_model().encode(documents).tolist()
            # Ensure collection exists
            if self.collection is None:
                self.collection = self._get_or_create_collection()
            self.collection.upsert(documents=documents, metadatas=metadatas, embeddings=embeddings, ids=ids)
            logger.info("Added %s chunks for %s", len(chunks), file_name)
            return True
        except Exception as e:
            logger.error("Error adding document: %s", e)
            return False

    def process_and_index_documents(self, files: List[Dict], drive_service=None) -> Dict:
        """Process files and add them to the vector database."""
        if drive_service:
            self.doc_processor.drive_service = drive_service

        results = {"processed": 0, "failed": 0, "chunks_added": 0, "files": []}

        for file_info in files:
            name = file_info.get("name", "unknown")
            try:
                logger.info("Processing file: %s", name)
                doc_data = self.doc_processor.process_file(file_info)
                if not doc_data:
                    logger.warning("Failed to process %s", name)
                    results["failed"] += 1
                    continue

                chunks = self.chunk_text(doc_data["text"])
                if not chunks:
                    results["failed"] += 1
                    continue

                self._delete_file_chunks(str(doc_data["file_id"]))
                ok = self.add_document(
                    doc_data["text"],
                    metadata={
                        "file_id": doc_data["file_id"],
                        "file_name": doc_data["file_name"],
                        "mime_type": doc_data.get("mime_type", ""),
                        "source": "google_drive",
                    },
                    chunk=True,
                )
                if not ok:
                    results["failed"] += 1
                    continue

                results["processed"] += 1
                results["chunks_added"] += len(chunks)
                results["files"].append(
                    {
                        "file_name": doc_data["file_name"],
                        "chunks": len(chunks),
                        "char_count": doc_data.get("char_count", len(doc_data["text"])),
                        "word_count": doc_data.get("word_count", len(doc_data["text"].split())),
                    }
                )
            except Exception as e:
                logger.error("Error processing %s: %s", name, e)
                results["failed"] += 1

        return results

    def search(self, query: str, n_results: int = 5) -> Dict:
        """Search for relevant documents."""
        try:
            total = self.collection.count()
            if total == 0:
                return {"query": query, "results": [], "total_results": 0}
            n_results = max(1, min(n_results, total))
            query_embedding = self._get_embedding_model().encode([query]).tolist()
            results = self.collection.query(
                query_embeddings=query_embedding,
                n_results=n_results,
                include=["documents", "metadatas", "distances"],
            )

            formatted_results = []
            docs = (results.get("documents") or [[]])[0]
            metas = (results.get("metadatas") or [[]])[0]
            dists = (results.get("distances") or [[]])[0]
            for i in range(len(docs)):
                dist = dists[i] if i < len(dists) else None
                formatted_results.append(
                    {
                        "document": docs[i],
                        "metadata": metas[i] or {},
                        "distance": dist,
                    }
                )


            return {"query": query, "results": formatted_results, "total_results": len(formatted_results)}
        except Exception as e:
            logger.error("Error searching: %s", e)
            return {"query": query, "results": [], "total_results": 0}

    def get_collection_stats(self) -> Dict:
        """Get statistics about the collection."""
        try:
            count = self.collection.count()
            files = self.list_files()
            return {
                "total_chunks": count,
                "total_documents": count,
                "total_files": len(files),
                "collection_name": self.collection_name,
                "model_name": self.model_name,
                "chroma_path": self.chroma_path,
            }
        except Exception as e:
            logger.error("Error getting collection stats: %s", e)
            return {}

    def clear_collection(self):
        """Clear all documents from the collection."""
        try:
            self.client.delete_collection(name=self.collection_name)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "Knowledge base documents"},
            )
            logger.info("Collection cleared successfully")
        except Exception as e:
            logger.error("Error clearing collection: %s", e)

    def list_files(self) -> List[Dict]:
        """List all unique files/sources in the collection."""
        try:
            results = self.collection.get(include=["metadatas"])
            files = {}
            for metadata in results.get("metadatas") or []:
                if not metadata:
                    continue
                file_id = metadata.get("file_id") or metadata.get("filename") or metadata.get("source")
                if not file_id:
                    continue
                file_id = str(file_id)
                if file_id not in files:
                    files[file_id] = {
                        "file_id": file_id,
                        "file_name": metadata.get("file_name") or metadata.get("filename") or file_id,
                        "chunks": 0,
                    }
                files[file_id]["chunks"] += 1
            return list(files.values())
        except Exception as e:
            logger.error("Error listing files: %s", e)
            return []