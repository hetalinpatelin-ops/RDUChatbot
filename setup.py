#!/usr/bin/env python3
"""
Setup and test script for the Knowledge Base Chatbot.
"""

import os
import sys
import logging
from pathlib import Path

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from document_processor import DocumentProcessor
from rag_pipeline import RAGPipeline

def setup_environment():
    """Setup the environment and test components."""
    print("🚀 Setting up Knowledge Base Chatbot...")
    
    # Create necessary directories
    os.makedirs("downloads", exist_ok=True)
    os.makedirs("chroma_db", exist_ok=True)
    
    # Test imports
    try:
        print("✅ Testing imports...")
        import chromadb
        from sentence_transformers import SentenceTransformer
        from googleapiclient.discovery import build
        import fitz
        from docx import Document
        print("✅ All imports successful!")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    
    # Initialize components
    try:
        print("✅ Testing RAG Pipeline...")
        rag_pipeline = RAGPipeline()
        stats = rag_pipeline.get_collection_stats()
        print(f"✅ RAG Pipeline initialized: {stats}")
        
        print("✅ Testing Document Processor...")
        doc_processor = DocumentProcessor()
        print("✅ Document Processor initialized!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error initializing components: {e}")
        return False

def test_drive_connection():
    """Test Google Drive connection."""
    print("\n🔍 Testing Google Drive connection...")
    
    try:
        from google.oauth2.credentials import Credentials
        from googleapiclient.discovery import build
        # Load credentials
        creds = None
        # Try multiple token paths
        token_paths = [
            os.path.expanduser('~/.hermes/google_token.json'),
            os.path.expanduser('~/AppData/Local/hermes/google_token.json'),
            'C:/Users/Hetal/AppData/Local/hermes/google_token.json'
        ]
            
        for token_path in token_paths:
            if os.path.exists(token_path):
                print(f"Found token at: {token_path}")
                creds = Credentials.from_authorized_user_file(token_path, ['https://www.googleapis.com/auth/drive'])
                break
        service = build('drive', 'v3', credentials=creds)
        
        # Test API call
        results = service.files().list(pageSize=1, fields="files(id, name)").execute()
        files = results.get('files', [])
        
        if files:
            print(f"✅ Connected to Google Drive. Found {len(files)} files.")
            return True
        else:
            print("❌ No files found in Drive.")
            return False
            
    except Exception as e:
        print(f"❌ Error connecting to Google Drive: {e}")
        return False

def test_rag_pipeline():
    """Test RAG pipeline with sample data."""
    print("\n🔍 Testing RAG Pipeline...")
    
    try:
        rag_pipeline = RAGPipeline()
        
        # Add sample document
        sample_text = """
        Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed.
        It involves algorithms that can identify patterns in data and make decisions based on those patterns.
        
        There are several types of machine learning:
        1. Supervised Learning: Uses labeled data to train models
        2. Unsupervised Learning: Finds patterns in unlabeled data
        3. Reinforcement Learning: Learns through trial and error with rewards
        
        Common algorithms include linear regression, decision trees, neural networks, and support vector machines.
        """
        
        # Process and add to database
        chunks = rag_pipeline.chunk_text(sample_text)
        
        for i, chunk in enumerate(chunks):
            rag_pipeline.collection.add(
                documents=[chunk],
                metadatas=[{
                    'file_name': 'sample_machine_learning.txt',
                    'chunk_index': i,
                    'total_chunks': len(chunks),
                    'timestamp': '2024-01-01T00:00:00'
                }],
                ids=[f"sample_{i}"]
            )
        
        # Test search
        results = rag_pipeline.search("What is machine learning?")
        
        if results['results']:
            print(f"✅ RAG Pipeline working! Found {len(results['results'])} relevant documents.")
            for result in results['results']:
                print(f"   - Distance: {result['distance']:.3f}")
                print(f"   - File: {result['metadata']['file_name']}")
            return True
        else:
            print("❌ No search results found.")
            return False
            
    except Exception as e:
        print(f"❌ Error testing RAG pipeline: {e}")
        return False

def main():
    """Main setup function."""
    logging.basicConfig(level=logging.INFO)
    
    print("🔧 Knowledge Base Chatbot Setup")
    print("=" * 40)
    
    # Setup environment
    if not setup_environment():
        print("\n❌ Setup failed. Please check the error messages above.")
        return
    
    # Test Drive connection
    if test_drive_connection():
        print("✅ Google Drive connection successful!")
    else:
        print("⚠️  Google Drive connection failed. You can still use the app with local files.")
    
    # Test RAG pipeline
    if test_rag_pipeline():
        print("✅ RAG Pipeline test successful!")
    else:
        print("❌ RAG Pipeline test failed.")
    
    print("\n🎉 Setup complete! You can now run the Streamlit app:")
    print("   streamlit run app.py")
    
    print("\n📋 Next steps:")
    print("1. Get your Google AI API key from https://makersuite.google.com/app/apikey")
    print("2. Run the Streamlit app: streamlit run app.py")
    print("3. Enter your API key in the sidebar")
    print("4. Click 'Setup Google Drive' to authenticate")
    print("5. Click 'Process Documents' to index your files")
    print("6. Start chatting!")

if __name__ == "__main__":
    main()