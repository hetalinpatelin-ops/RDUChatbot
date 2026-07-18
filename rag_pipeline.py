#!/usr/bin/env python3
"""
RAG Pipeline using ChromaDB and sentence-transformers.
"""

import os
import logging
from typing import List, Dict, Optional, Tuple
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import re
import json
from datetime import datetime

from document_processor import DocumentProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGPipeline:
    def __init__(self, 
                 model_name: str = "all-MiniLM-L6-v2",
                 chroma_path: str = "chroma_db",
                 collection_name: str = "kb_documents"):
        
        self.model_name = model_name
        self.chroma_path = chroma_path
        self.collection_name = collection_name
        
        # Initialize sentence transformer
        logger.info(f"Loading model: {model_name}")
        self.embedding_model = SentenceTransformer(model_name)
        
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(
            path=chroma_path,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "Knowledge base documents"}
        )
        
        # Document processor
        self.doc_processor = DocumentProcessor()
        
        logger.info("RAG Pipeline initialized successfully")
    
    def chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Split text into overlapping chunks."""
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # Try to end at a sentence boundary
            if end < len(text):
                # Look for sentence ending
                sentence_end = text.find('.', end - 50, end + 50)
                if sentence_end != -1:
                    end = sentence_end + 1
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - overlap
        
        return chunks
    
    def process_and_index_documents(self, files: List[Dict], drive_service = None) -> Dict:
        """Process files and add them to the vector database."""
        if drive_service:
            self.doc_processor.drive_service = drive_service
        
        results = {
            'processed': 0,
            'failed': 0,
            'chunks_added': 0,
            'files': []
        }
        
        for file_info in files:
            try:
                logger.info(f"Processing file: {file_info['name']}")
                
                # Extract text
                doc_data = self.doc_processor.process_file(file_info)
                
                if not doc_data:
                    logger.warning(f"Failed to process {file_info['name']}")
                    results['failed'] += 1
                    continue
                
                # Chunk the text
                chunks = self.chunk_text(doc_data['text'])
                
                # Add to vector database
                for i, chunk in enumerate(chunks):
                    chunk_id = f"{doc_data['file_id']}_{i}"
                    
                    # Add to ChromaDB
                    self.collection.add(
                        documents=[chunk],
                        metadatas=[{
                            'file_id': doc_data['file_id'],
                            'file_name': doc_data['file_name'],
                            'chunk_index': i,
                            'total_chunks': len(chunks),
                            'char_count': len(chunk),
                            'word_count': len(chunk.split()),
                            'timestamp': datetime.now().isoformat()
                        }],
                        ids=[chunk_id]
                    )
                
                results['processed'] += 1
                results['chunks_added'] += len(chunks)
                results['files'].append({
                    'file_name': doc_data['file_name'],
                    'chunks': len(chunks),
                    'char_count': doc_data['char_count'],
                    'word_count': doc_data['word_count']
                })
                
                logger.info(f"Added {len(chunks)} chunks from {doc_data['file_name']}")
                
            except Exception as e:
                logger.error(f"Error processing {file_info['name']}: {e}")
                results['failed'] += 1
        
        return results
    
    def search(self, query: str, n_results: int = 5) -> Dict:
        """Search for relevant documents."""
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query])
            
            # Search ChromaDB
            results = self.collection.query(
                query_embeddings=query_embedding.tolist(),
                n_results=n_results,
                include=['documents', 'metadatas', 'distances']
            )
            
            # Format results
            formatted_results = []
            for i in range(len(results['documents'][0])):
                formatted_results.append({
                    'document': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i]
                })
            
            return {
                'query': query,
                'results': formatted_results,
                'total_results': len(formatted_results)
            }
            
        except Exception as e:
            logger.error(f"Error searching: {e}")
            return {'query': query, 'results': [], 'total_results': 0}
    
    def get_collection_stats(self) -> Dict:
        """Get statistics about the collection."""
        try:
            count = self.collection.count()
            files = self.list_files()
            
            return {
                'total_documents': count,
                'total_files': len(files),
                'collection_name': self.collection_name,
                'model_name': self.model_name,
                'chroma_path': self.chroma_path
            }
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {}
    
    def clear_collection(self):
        """Clear all documents from the collection."""
        try:
            self.client.delete_collection(name=self.collection_name)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "Knowledge base documents"}
            )
            logger.info("Collection cleared successfully")
        except Exception as e:
            logger.error(f"Error clearing collection: {e}")
    
    def save_metadata(self, metadata: Dict):
        """Save metadata to JSON file."""
        try:
            with open(os.path.join(self.chroma_path, 'metadata.json'), 'w') as f:
                json.dump(metadata, f, indent=2)
            logger.info("Metadata saved successfully")
        except Exception as e:
            logger.error(f"Error saving metadata: {e}")
    
    def load_metadata(self) -> Dict:
        """Load metadata from JSON file."""
        try:
            metadata_file = os.path.join(self.chroma_path, 'metadata.json')
            if os.path.exists(metadata_file):
                with open(metadata_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Error loading metadata: {e}")
        return {}
    
    def list_files(self) -> List[Dict]:
        """List all unique files in the collection."""
        try:
            # Get all documents
            results = self.collection.get(include=['metadatas'])
            
            # Group by file_id
            files = {}
            for metadata in results['metadatas']:
                if not metadata:  # Skip empty metadata
                    continue
                
                file_id = metadata.get('file_id')
                if not file_id:  # Skip if no file_id
                    continue
                
                if file_id not in files:
                    files[file_id] = {
                        'file_id': file_id,
                        'file_name': metadata.get('file_name', 'Unknown'),
                        'chunks': 0,
                        'total_char_count': 0,
                        'total_word_count': 0
                    }
                
                files[file_id]['chunks'] += 1
                files[file_id]['total_char_count'] += metadata.get('char_count', 0)
                files[file_id]['total_word_count'] += metadata.get('word_count', 0)
            
            return list(files.values())
            
        except Exception as e:
            logger.error(f"Error listing files: {e}")
            return []