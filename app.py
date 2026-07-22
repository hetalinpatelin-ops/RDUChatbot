#!/usr/bin/env python3
"""
Knowledge Base Chatbot with RAG pipeline and Google Drive integration.
"""

import streamlit as st
from google import genai
import os
import warnings
import logging
from dotenv import load_dotenv
from typing import List, Dict

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChatInterface:
    def __init__(self):
        # Initialize session state
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        
        if 'rag_pipeline' not in st.session_state:
            st.session_state.rag_pipeline = None
        
        if 'doc_processor' not in st.session_state:
            st.session_state.doc_processor = None
        
        if 'drive_service' not in st.session_state:
            st.session_state.drive_service = None
        
        # Auto-initialize RAG pipeline if chroma_db exists
        if st.session_state.rag_pipeline is None:
            try:
                import os
                if os.path.exists("chroma_db"):
                    st.session_state.rag_pipeline = RAGPipeline(
                        model_name="all-MiniLM-L6-v2",
                        chroma_path="chroma_db",
                        collection_name="kb_documents"
                    )
                    stats = st.session_state.rag_pipeline.get_collection_stats()
                    logger.info(f"RAG pipeline auto-initialized - {stats.get('total_files', 0)} files, {stats.get('total_documents', 0)} chunks")
            except Exception as e:
                logger.warning(f"Could not auto-initialize RAG pipeline: {e}")
    
    def setup_local_llm(self, api_key: str, base_url: str):
        """Setup local LLM using FreeLLM API proxy."""
        try:
            from openai import OpenAI
            
            # Configure OpenAI client for local LLM
            client = OpenAI(
                api_key=api_key,
                base_url=base_url
            )
            
            # Test the connection
            response = client.chat.completions.create(
                model="auto",  # Let the proxy auto-select
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=5
            )
            
            st.session_state.local_llm_client = client
            st.session_state.local_llm_model = "auto"
            st.success("✅ Local LLM configured!")
            return True
            
        except Exception as e:
            st.error(f"Error setting up local LLM: {e}")
            return False
    
    def setup_google_ai(self, api_key: str):
        """Setup Google AI client - auto-detect available model."""
        try:
            client = genai.Client(api_key=api_key)
            
            # Check if API key changed (re-init needed)
            if hasattr(st.session_state, 'gemini_client') and st.session_state.get('_last_api_key') == api_key:
                # Already initialized, don't override user's model choice
                return
            
            st.session_state._last_api_key = api_key
            
            # Models to try in order (most to least preferred)
            models_to_try = [
                'gemini-flash-lite-latest',
                'gemini-2.0-flash-lite',
                'gemini-2.0-flash',
                'gemini-1.5-flash',
                'gemini-1.5-flash-8b',
                'gemini-1.5-pro',
                'gemini-pro',
            ]
            
            # Try to list available models to find a working one
            try:
                available = client.models.list()
                model_names = [m.name for m in available]
                # genai client returns names like "models/gemini-2.0-flash"
                # Strip the "models/" prefix for comparison
                model_short = [n.replace('models/', '') for n in model_names]
                st.session_state.available_models = model_short
            except Exception:
                model_short = []
                model_names = []
            
            if model_short:
                # Find the first preferred model that exists
                for model_name in models_to_try:
                    if model_name in model_short:
                        st.session_state.gemini_client = client
                        st.session_state.gemini_model_name = model_name
                        st.success(f"✅ Google AI configured! Using: {model_name}")
                        return
                # Fallback: use first available chat model
                first_model = model_short[0]
                st.session_state.gemini_client = client
                st.session_state.gemini_model_name = first_model
                st.success(f"✅ Google AI configured! Using: {first_model}")
            else:
                # Fallback - try model names directly
                for model_name in models_to_try:
                    try:
                        response = client.models.generate_content(
                            model=model_name,
                            contents="test"
                        )
                        st.session_state.gemini_client = client
                        st.session_state.gemini_model_name = model_name
                        st.success(f"✅ Google AI configured! Using: {model_name}")
                        return
                    except Exception:
                        continue
                st.error("Could not find a working Gemini model. Check your API key.")
                
        except Exception as e:
            st.error(f"Error setting up Google AI: {e}")
    
    def setup_drive_service(self):
        """Setup Google Drive service."""
        try:
            from google.oauth2.credentials import Credentials
            from google_auth_oauthlib.flow import InstalledAppFlow
            from googleapiclient.discovery import build
            from google.auth.transport.requests import Request
            import pickle
            import os
            
            # If modifying these scopes, delete the file token.pickle.
            SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
            creds = None
            
            # The file token.pickle stores the user's access and refresh tokens.
            if os.path.exists('token.pickle'):
                with open('token.pickle', 'rb') as token:
                    creds = pickle.load(token)
            
            # If there are no (valid) credentials available, let the user log in.
            if not creds or not creds.valid:
                if creds and creds.expired and creds.refresh_token:
                    creds.refresh(Request())
                else:
                    flow = InstalledAppFlow.from_client_secrets_file(
                        'credentials.json', SCOPES)
                    creds = flow.run_local_server(port=0)
                
                # Save the credentials for the next run
                with open('token.pickle', 'wb') as token:
                    pickle.dump(creds, token)
            
            service = build('drive', 'v3', credentials=creds)
            st.session_state.drive_service = service
            st.success("✅ Google Drive connected!")
            return True
            
        except Exception as e:
            st.error(f"Error setting up Google Drive: {e}")
            return False
    
    def setup_rag_pipeline(self):
        """Setup RAG pipeline."""
        try:
            if st.session_state.rag_pipeline is None:
                st.session_state.rag_pipeline = RAGPipeline(
                    model_name="all-MiniLM-L6-v2",
                    chroma_path="chroma_db",
                    collection_name="kb_documents"
                )
                st.success("✅ RAG pipeline ready!")
            return True
        except Exception as e:
            st.error(f"Error setting up RAG pipeline: {e}")
            return False
    
    def validate_folder_id(self, folder_id: str) -> bool:
        """Validate if folder ID is accessible."""
        try:
            if not st.session_state.drive_service:
                return False
            
            # Try to get folder metadata
            response = st.session_state.drive_service.files().get(
                fileId=folder_id,
                fields="id, name"
            ).execute()
            
            return 'id' in response
        except Exception:
            return False
    
    def process_documents(self, folder_id: str = None):
        """Process documents from Google Drive."""
        try:
            if not st.session_state.drive_service or not st.session_state.rag_pipeline:
                st.error("Drive service or RAG pipeline not initialized")
                return
            
            from document_processor import DocumentProcessor
            
            # Initialize document processor
            st.session_state.doc_processor = DocumentProcessor(
                drive_service=st.session_state.drive_service,
                rag_pipeline=st.session_state.rag_pipeline
            )
            
            # Process documents
            if folder_id:
                st.session_state.doc_processor.process_folder(folder_id)
            else:
                st.session_state.doc_processor.process_all_documents()
            
            st.success("✅ Documents processed successfully!")
            
        except Exception as e:
            st.error(f"Error processing documents: {e}")
    
    def show_collection_stats(self):
        """Show collection statistics."""
        if st.session_state.rag_pipeline:
            stats = st.session_state.rag_pipeline.get_collection_stats()
            st.write(f"**Total Files:** {stats.get('total_files', 0)}")
            st.write(f"**Total Documents:** {stats.get('total_documents', 0)}")
            st.write(f"**Total Chunks:** {stats.get('total_chunks', 0)}")
        else:
            st.warning("RAG pipeline not initialized")
    
    def search_documents(self, query: str) -> str:
        """Search for relevant documents."""
        if not st.session_state.rag_pipeline:
            return ""
        
        try:
            results = st.session_state.rag_pipeline.search(query, k=3)
            context = "\n\n".join([doc['content'] for doc in results])
            return context
        except Exception as e:
            st.error(f"Error searching documents: {e}")
            return ""
    
    def generate_response(self, query: str, context: str) -> str:
        """Generate response using LLM."""
        try:
            if hasattr(st.session_state, 'use_local_llm') and st.session_state.use_local_llm and hasattr(st.session_state, 'local_llm_client'):
                # Use local LLM
                response = st.session_state.local_llm_client.chat.completions.create(
                    model=st.session_state.local_llm_model,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant. Use the following context to answer the user's question. If you don't know the answer, say you don't know.\n\nContext:"},
                        {"role": "system", "content": context},
                        {"role": "user", "content": query}
                    ],
                    max_tokens=1000
                )
                return response.choices[0].message.content
            elif hasattr(st.session_state, 'gemini_client'):
                # Use Google Gemini
                response = st.session_state.gemini_client.models.generate_content(
                    model=st.session_state.gemini_model_name,
                    contents=f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
                )
                return response.text
            else:
                return "No LLM configured. Please set up an LLM first."
        except Exception as e:
            st.error(f"Error generating response: {e}")
            return "Sorry, I encountered an error generating a response."
    
    def run(self):
        """Run the chat interface."""
        st.title("🤖 Knowledge Base Chatbot")
        
        # Only show admin sidebar if ADMIN_MODE is enabled
        admin_mode = st.secrets.get("ADMIN_MODE", False) or os.getenv("ADMIN_MODE", "") == "true"
        
        if admin_mode:
            with st.sidebar:
                st.header("Configuration")
                
                # Google AI API Key (loaded from secrets/env)
                api_key = os.getenv("GOOGLE_API_KEY", "")
                if "GOOGLE_API_KEY" in st.secrets:
                    api_key = st.secrets["GOOGLE_API_KEY"]
                
                if api_key:
                    self.setup_google_ai(api_key)
                    st.info("✅ Using Google Gemini from environment secrets")
                else:
                    api_key = st.text_input(
                        "Google AI API Key",
                        type="password",
                        key="admin_google_api_key",
                        value="",
                        help="Get your free API key from https://aistudio.google.com/app/apikey"
                    )
                    if api_key:
                        self.setup_google_ai(api_key)
                
                # Show model status
                if hasattr(st.session_state, 'gemini_model_name'):
                    st.info(f"**Current Model:** {st.session_state.gemini_model_name}")
                
                # Manual model selection
                if api_key and st.button("🔍 Show Available Models", key="admin_show_models"):
                    try:
                        client = genai.Client(api_key=api_key)
                        available = client.models.list()
                        model_names = [m.name.replace('models/', '') for m in available]
                        st.session_state.available_models = model_names
                        st.subheader("Available Models:")
                        for name in model_names:
                            st.write(f"- {name}")
                    except Exception as e:
                        st.error(f"Error listing models: {e}")
                
                # Manual model override
                if hasattr(st.session_state, 'available_models') and st.session_state.available_models:
                    selected_model = st.selectbox(
                        "Select Model",
                        st.session_state.available_models,
                        index=0,
                        key="admin_select_model",
                        help="Choose a model with available quota"
                    )
                    if st.button("Use Selected Model", key="admin_use_model"):
                        try:
                            client = genai.Client(api_key=api_key)
                            st.session_state.gemini_client = client
                            st.session_state.gemini_model_name = selected_model
                            st.success(f"✅ Switched to model: {selected_model}")
                        except Exception as e:
                            st.error(f"Error switching model: {e}")
                
                # Drive Setup
                st.subheader("Google Drive Setup")
                if st.button("Setup Google Drive", key="admin_setup_drive"):
                    if self.setup_drive_service():
                        self.setup_rag_pipeline()
                
                # Document Processing
                st.subheader("Document Processing")
                folder_id = st.text_input(
                    "Folder ID",
                    placeholder="Enter folder ID or leave empty for root",
                    key="admin_folder_id",
                    help="Get folder ID from URL: https://drive.google.com/drive/folders/YOUR_FOLDER_ID"
                )
                
                # Validate folder ID
                if folder_id:
                    if self.validate_folder_id(folder_id):
                        st.success(f"✅ Valid folder ID: {folder_id}")
                    else:
                        st.warning(f"⚠️ Folder ID may be invalid or inaccessible: {folder_id}")
                
                if st.button("Process Documents", type="primary", key="admin_process_docs"):
                    st.info("Starting document processing...")
                    
                    # Setup services
                    drive_ok = self.setup_drive_service()
                    rag_ok = self.setup_rag_pipeline()
                    
                    st.info(f"Drive setup: {drive_ok}, RAG setup: {rag_ok}")
                    
                    if drive_ok and rag_ok:
                        st.info(f"Processing folder ID: {folder_id if folder_id else 'root'}")
                        self.process_documents(folder_id if folder_id else None)
                    else:
                        st.error("Failed to setup services. Check logs for details.")
                
                # Collection Stats
                st.subheader("Collection Info")
                if st.button("Show Collection Stats", key="admin_show_stats"):
                    self.show_collection_stats()
                
                # Clear Collection
                if st.button("Clear Collection", key="admin_clear_collection"):
                    if st.session_state.rag_pipeline:
                        st.session_state.rag_pipeline.clear_collection()
                        st.success("Collection cleared!")
                
                st.markdown("---")
                
                # Local LLM Configuration
                st.subheader("Local LLM Setup")
                local_api_key = st.text_input(
                    "Local LLM API Key",
                    type="password",
                    key="admin_local_api_key",
                    help="API key for your local LLM container"
                )
                local_base_url = st.text_input(
                    "Local LLM Base URL",
                    value="http://localhost:11434/v1",
                    key="admin_local_base_url",
                    help="Base URL for your local LLM container"
                )
                
                if st.button("Setup Local LLM", key="admin_setup_llm"):
                    if self.setup_local_llm(local_api_key, local_base_url):
                        st.session_state.use_local_llm = True
                
                # Toggle between local and cloud LLM
                if hasattr(st.session_state, 'local_llm_client') and hasattr(st.session_state, 'gemini_client'):
                    st.radio(
                        "LLM Provider",
                        options=["Local LLM", "Google Gemini"],
                        key="admin_llm_provider",
                        help="Switch between local and cloud LLM"
                    )
        else:
            # Visitor mode - minimal sidebar
            with st.sidebar:
                st.markdown("### 🤖 Knowledge Base Chatbot")
                st.markdown("Ask questions about your documents")
                st.markdown("---")
                st.markdown("**Current Model:**")
                if hasattr(st.session_state, 'gemini_model_name'):
                    st.info(st.session_state.gemini_model_name)
                elif hasattr(st.session_state, 'local_llm_model'):
                    st.info(st.session_state.local_llm_model)
                
                st.markdown("---")
                st.markdown("**Documents Loaded:**")
                if st.session_state.rag_pipeline:
                    stats = st.session_state.rag_pipeline.get_collection_stats()
                    st.write(f"{stats.get('total_files', 0)} files")
                    st.write(f"{stats.get('total_documents', 0)} chunks")
                else:
                    st.write("Not loaded")
        
        # Chat Interface
        st.subheader("Chat")
        
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])
        
        # User input
        if prompt := st.chat_input("Ask a question about your documents..."):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Display user message
            with st.chat_message("user"):
                st.write(prompt)
            
            # Search for relevant documents
            with st.chat_message("assistant"):
                with st.spinner("Searching documents..."):
                    context = self.search_documents(prompt)
                
                if context:
                    with st.spinner("Generating response..."):
                        response = self.generate_response(prompt, context)
                    
                    st.write(response)
                    
                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": response})
                else:
                    st.write("I couldn't find relevant documents to answer your question.")
                    st.session_state.messages.append({"role": "assistant", "content": "I couldn't find relevant documents to answer your question."})

if __name__ == "__main__":
    chat_app = ChatInterface()
    chat_app.run()