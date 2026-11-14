#!/usr/bin/env python3
"""
Knowledge Base Chatbot with RAG pipeline and Google Drive integration.
"""

import streamlit as st
from google import genai
import os
import warnings
import logging
import json
from datetime import datetime
from dotenv import load_dotenv
from typing import Dict, List, Optional

from rag_pipeline import RAGPipeline
from document_processor import DocumentProcessor
from feedback_store import is_configured as feedback_store_configured, add_feedback as feedback_store_add, fetch_recent as feedback_store_fetch, rehydrate_chroma as feedback_store_rehydrate, delete_feedback as feedback_store_delete, update_feedback as feedback_store_update

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
        
        if 'rag_init_error' not in st.session_state:
            st.session_state.rag_init_error = None

        # Best-effort auto-init (non-fatal; retried on demand)
        if st.session_state.rag_pipeline is None and os.path.exists("chroma_db"):
            self.ensure_rag_pipeline(show_errors=False)
    
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
    
    def setup_google_ai(self, api_key: str, admin_mode: bool = False):
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
            available = client.models.list()
            model_names = [m.name for m in available]
            # genai client returns names like "models/gemini-2.0-flash"
            # Strip the "models/" prefix for comparison
            model_short = [n.replace('models/', '') for n in model_names]
            st.session_state.available_models = model_short

            if model_short:
                # Find the first preferred model that exists
                for model_name in models_to_try:
                    if model_name in model_short:
                        st.session_state.gemini_client = client
                        st.session_state.gemini_model_name = model_name
                        if admin_mode:
                            st.success(f"Google AI configured! Using: {model_name}")
                        return
                # Fallback: use first available chat model
                first_model = model_short[0]
                st.session_state.gemini_client = client
                st.session_state.gemini_model_name = first_model
                if admin_mode:
                    st.success(f"Google AI configured! Using: {first_model}")
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
                        if admin_mode:
                            st.success(f"Google AI configured! Using: {model_name}")
                        return
                    except Exception:
                        continue
                st.error("Could not find a working Gemini model. Check your API key.")

        except Exception as e:
            if admin_mode:
                st.error(f"Error setting up Google AI: {e}")
            else:
                logger.error(f"Error setting up Google AI: {e}")
    
    def setup_drive_service(self):
        """Setup Google Drive service (local/admin environments only)."""
        try:
            if os.getenv("STREAMLIT_SHARING_MODE") or os.getenv("STREAMLIT_SERVER_HEADLESS") == "true":
                st.warning("Google Drive OAuth is unavailable in cloud/headless deploys. Use prebuilt chroma_db or local documents/.")
                return False
            if not os.path.exists('credentials.json'):
                st.warning("credentials.json not found. Google Drive setup is local-admin only.")
                return False

            from google_auth_oauthlib.flow import InstalledAppFlow
            from googleapiclient.discovery import build
            from google.auth.transport.requests import Request
            import pickle

            SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
            creds = None
            if os.path.exists('token.pickle'):
                with open('token.pickle', 'rb') as token:
                    creds = pickle.load(token)

            if not creds or not creds.valid:
                if creds and getattr(creds, 'expired', False) and getattr(creds, 'refresh_token', None):
                    creds.refresh(Request())
                else:
                    flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
                    creds = flow.run_local_server(port=0)
                with open('token.pickle', 'wb') as token:
                    pickle.dump(creds, token)

            service = build('drive', 'v3', credentials=creds)
            st.session_state.drive_service = service
            st.session_state.doc_processor = DocumentProcessor(drive_service=service)
            st.success("Google Drive connected.")
            return True
        except Exception as e:
            st.error(f"Error setting up Google Drive: {e}")
            return False
    
    def ensure_rag_pipeline(self, show_errors: bool = True) -> bool:
        """Initialize RAG pipeline on demand."""
        if st.session_state.get('rag_pipeline') is not None:
            return True
        try:
            st.session_state.rag_pipeline = RAGPipeline(
                model_name="all-MiniLM-L6-v2",
                chroma_path="chroma_db",
                collection_name="kb_documents_v2",
            )
            # One-time rehydrate from Supabase if configured
            if feedback_store_configured() and not st.session_state.get('_feedback_rehydrated'):
                feedback_store_rehydrate(st.session_state.rag_pipeline)
                st.session_state._feedback_rehydrated = True
            st.session_state.rag_init_error = None
            return True
        except Exception as e:
            msg = f"Error initializing RAG pipeline: {e}"
            st.session_state.rag_init_error = msg
            logger.exception("RAG pipeline initialization failed")
            if show_errors:
                st.error(msg)
            return False

    def setup_rag_pipeline(self):
        """Setup RAG pipeline."""
        ok = self.ensure_rag_pipeline(show_errors=True)
        if ok and st.session_state.get('rag_pipeline') is not None:
            st.success("RAG pipeline ready.")
        return ok
    
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
        """Process documents from Google Drive folder."""
        try:
            if not st.session_state.drive_service or not st.session_state.rag_pipeline:
                st.error("Drive service or RAG pipeline not initialized")
                return False

            from document_processor import DocumentProcessor
            doc_processor = DocumentProcessor(drive_service=st.session_state.drive_service)
            st.session_state.doc_processor = doc_processor

            with st.spinner("Processing documents..."):
                files = doc_processor.list_drive_files(folder_id)
                if not files:
                    st.warning("No supported files found in the specified folder")
                    return False

                st.info(f"Found {len(files)} files to process")
                results = st.session_state.rag_pipeline.process_and_index_documents(
                    files, st.session_state.drive_service
                )

            st.success("✅ Processing complete!")
            st.write(f"**Processed:** {results.get('processed', 0)} files")
            st.write(f"**Failed:** {results.get('failed', 0)} files")
            st.write(f"**Chunks added:** {results.get('chunks_added', 0)}")
            for file_info in results.get('files', []):
                st.write(f"- {file_info['file_name']}: {file_info.get('chunks', 0)} chunks")
            return True
        except Exception as e:
            st.error(f"Error processing documents: {e}")
            logger.error(f"Error processing documents: {e}")
            return False
    
    def show_collection_stats(self):
        """Show collection statistics."""
        if not self.ensure_rag_pipeline(show_errors=True):
            return
        stats = st.session_state.rag_pipeline.get_collection_stats()
        st.write(f"**Total Sources:** {stats.get('total_files', 0)}")
        st.write(f"**Total Chunks:** {stats.get('total_chunks', stats.get('total_documents', 0))}")
        st.write(f"**Collection:** {stats.get('collection_name', 'n/a')}")
        st.write(f"**Embedding Model:** {stats.get('model_name', 'n/a')}")
        if st.session_state.get('rag_init_error'):
            st.caption(st.session_state['rag_init_error'])
    
    def process_local_documents(self, folder_path: str) -> int:
        """Process documents from a local folder."""
        if not st.session_state.rag_pipeline:
            if not self.setup_rag_pipeline():
                return 0

        if not os.path.exists(folder_path):
            st.error(f"Folder not found: {folder_path}")
            return 0

        count = 0
        supported_extensions = ('.txt', '.md', '.pdf', '.docx', '.json', '.xlsx', '.xlsm')

        for filename in os.listdir(folder_path):
            if not filename.lower().endswith(supported_extensions):
                continue
            filepath = os.path.join(folder_path, filename)
            try:
                content = self.extract_text_from_file(filepath)
                if not content.strip():
                    st.write(f"⚠️ Empty: {filename}")
                    continue
                metadata = {
                    'file_id': f"local::{filename}",
                    'file_name': filename,
                    'filename': filename,
                    'filepath': filepath,
                    'source': 'local_folder',
                    'processed_at': datetime.now().isoformat()
                }
                success = st.session_state.rag_pipeline.add_document(content, metadata, chunk=True, replace_existing=True)
                if success:
                    count += 1
                    st.write(f"✅ Processed: {filename}")
                else:
                    st.write(f"❌ Failed: {filename}")
            except Exception as e:
                st.write(f"❌ Error processing {filename}: {e}")

        return count
    
    def extract_text_from_file(self, filepath: str) -> str:
        """Extract text content from various file types."""
        ext = os.path.splitext(filepath)[1].lower()
        
        if ext in ('.txt', '.md', '.json'):
            with open(filepath, 'r', encoding='utf-8') as f:
                return f.read()
        
        elif ext == '.pdf':
            try:
                import fitz  # PyMuPDF
                doc = fitz.open(filepath)
                text = ""
                for page in doc:
                    text += page.get_text()
                doc.close()
                return text
            except ImportError:
                st.error("PyMuPDF not installed. Run: pip install pymupdf")
                return ""
            except Exception as e:
                st.error(f"Error reading PDF: {e}")
                return ""
        
        elif ext == '.docx':
            try:
                import docx
                doc = docx.Document(filepath)
                return "\n".join([para.text for para in doc.paragraphs])
            except ImportError:
                st.error("python-docx not installed. Run: pip install python-docx")
                return ""
            except Exception as e:
                st.error(f"Error reading DOCX: {e}")
                return ""

        elif ext in ('.xlsx', '.xlsm'):
            try:
                import openpyxl
                wb = openpyxl.load_workbook(filepath, read_only=True, data_only=True)
                text_parts = []
                for sheet_name in wb.sheetnames:
                    ws = wb[sheet_name]
                    rows = list(ws.iter_rows(values_only=True))
                    if not rows:
                        continue
                    headers = [str(h).strip() if h is not None else f"col_{i}" for i, h in enumerate(rows[0])]
                    text_parts.append(f"=== Sheet: {sheet_name} ===")
                    # Find name and phone column indices
                    name_idx = None
                    phone_indices = []
                    for i, h in enumerate(headers):
                        hl = h.lower()
                        if name_idx is None and ("name" in hl or "contact" in hl):
                            name_idx = i
                        if "phone" in hl:
                            phone_indices.append(i)
                    # Format 1: Header: value rows
                    for row in rows[1:]:
                        row_vals = []
                        phone_vals = []
                        name_val = None
                        for i, v in enumerate(row):
                            if v is not None:
                                h = headers[i] if i < len(headers) else f"col_{i}"
                                row_vals.append(f"{h}: {v}")
                                if "phone" in h.lower() and str(v).strip():
                                    phone_vals.append(str(v).strip())
                                if name_idx is not None and i == name_idx:
                                    name_val = str(v).strip()
                        if row_vals:
                            text_parts.append(" | ".join(row_vals))
                        # Format 2a: Combined name + phone line (keeps them in same chunk)
                        if name_val and phone_vals:
                            for p in phone_vals:
                                text_parts.append(f"{name_val} phone number: {p}")
                        # Format 2b: Dedicated phone line with "phone number" keyword
                        for p in phone_vals:
                            text_parts.append(f"phone number: {p}")
                    # Format 2c: Concatenated values
                    for row in rows[1:]:
                        vals = [str(v).strip() for v in row if v is not None]
                        if vals:
                            text_parts.append(" ".join(vals))
                wb.close()
                return "\n".join(text_parts)
            except ImportError:
                st.error("openpyxl not installed. Run: pip install openpyxl")
                return ""
            except Exception as e:
                st.error(f"Error reading XLSX: {e}")
                return ""
        
        return ""
    
    def search_documents(self, query: str, n_results: int = 4, max_distance: Optional[float] = 1.5) -> List[Dict]:
        """Search for relevant document chunks with optional relevance filtering."""
        if not self.ensure_rag_pipeline(show_errors=True):
            return []

        try:
            search_result = st.session_state.rag_pipeline.search(query, n_results=n_results)
            results = search_result.get('results', []) or []
            if max_distance is None:
                return results
            filtered = []
            for r in results:
                d = r.get('distance')
                try:
                    if d is None or float(d) <= float(max_distance):
                        filtered.append(r)
                except Exception:
                    filtered.append(r)
            return filtered
        except Exception as e:
            st.error(f"Error searching documents: {e}")
            return []

    def _format_context(self, results: List[Dict]) -> str:
        blocks = []
        for r in results:
            meta = r.get('metadata', {}) or {}
            source = meta.get('file_name') or meta.get('filename') or meta.get('source') or 'Unknown source'
            text = (r.get('document') or '').strip()
            if text:
                blocks.append(f"Source: {source}\n{text}")
        return "\n\n---\n\n".join(blocks)

    def _active_llm_provider(self) -> str:
        provider = st.session_state.get('admin_llm_provider')
        if provider == "Local LLM" and st.session_state.get('local_llm_client'):
            return 'local'
        if provider == "Google Gemini" and st.session_state.get('gemini_client'):
            return 'gemini'
        if st.session_state.get('local_llm_client') and st.session_state.get('use_local_llm'):
            return 'local'
        if st.session_state.get('gemini_client'):
            return 'gemini'
        if st.session_state.get('local_llm_client'):
            return 'local'
        return 'none'

    def generate_response(self, query: str, results: List[Dict]) -> str:
        """Generate response using selected LLM and retrieved context."""
        context = self._format_context(results)
        if not context:
            return "I couldn't find relevant information in the knowledge base for that question."

        prompt = (
            "You are a helpful knowledge-base assistant. Answer using only the provided context. "
            "If the context does not contain the answer, say you don't know based on the available documents.\n\n"
            f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
        )
        try:
            provider = self._active_llm_provider()
            if provider == 'local':
                response = st.session_state.local_llm_client.chat.completions.create(
                    model=st.session_state.get('local_llm_model', 'auto'),
                    messages=[
                        {"role": "system", "content": "Answer from provided context only. If missing, say you don't know."},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=1000,
                )
                return response.choices[0].message.content
            if provider == 'gemini':
                response = st.session_state.gemini_client.models.generate_content(
                    model=st.session_state.gemini_model_name,
                    contents=prompt,
                )
                return getattr(response, "text", "") or "Sorry, I couldn't generate a response."
            return "No LLM configured. Add a Gemini API key (or configure a local LLM) in admin mode."
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"Sorry, I encountered an error generating a response: {e}"
    
    def run(self):
        """Run the chat interface."""
        st.title("Welcome to the RDU Chatbot")

        # Determine admin mode first
        try:
            admin_mode = bool(st.secrets.get("ADMIN_MODE", False)) or os.getenv("ADMIN_MODE", "") == "true"
        except Exception:
            admin_mode = os.getenv("ADMIN_MODE", "") == "true"

        # Configure Gemini from secrets/env for all users if available
        secret_api_key = ""
        try:
            if "GOOGLE_API_KEY" in st.secrets:
                secret_api_key = st.secrets.get("GOOGLE_API_KEY") or ""
        except Exception:
            secret_api_key = ""
        secret_api_key = secret_api_key or os.getenv("GOOGLE_API_KEY", "")
        if secret_api_key and not st.session_state.get('gemini_client'):
            self.setup_google_ai(secret_api_key, admin_mode)
        
        if admin_mode:
            with st.sidebar:
                st.header("Configuration")
                
                # Google AI API Key (loaded from secrets/env)
                api_key = os.getenv("GOOGLE_API_KEY", "")
                if "GOOGLE_API_KEY" in st.secrets:
                    api_key = st.secrets["GOOGLE_API_KEY"]
                
                if api_key:
                    self.setup_google_ai(api_key, admin_mode)
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
                
                # Drive Setup (only show if credentials.json exists locally)
                if os.path.exists('credentials.json'):
                    st.subheader("Google Drive Setup")
                    if st.button("Setup Google Drive", key="admin_setup_drive"):
                        if self.setup_drive_service():
                            self.setup_rag_pipeline()
                else:
                    st.info("📁 Documents are already loaded from chroma_db")
                
                # Local Document Processing (from documents/ folder in repo)
                st.subheader("📂 Process Local Documents Folder")
                st.markdown("Upload documents to the `documents/` folder in your GitHub repo, then click below to process them.")
                
                if os.path.exists("documents"):
                    doc_files = os.listdir("documents")
                    st.info(f"📄 Found {len(doc_files)} files in documents/ folder")
                    
                    if st.button("Process documents/ folder", key="admin_process_local_docs", type="primary"):
                        with st.spinner("Processing documents..."):
                            count = self.process_local_documents("documents")
                            st.success(f"✅ Processed {count} documents into knowledge base!")
                            st.rerun()
                else:
                    st.warning("⚠️ No `documents/` folder found. Create one in your GitHub repo and add files.")
                    if st.button("Create documents/ folder", key="admin_create_docs_folder"):
                        os.makedirs("documents", exist_ok=True)
                        st.success("✅ Created documents/ folder. Add files and push to GitHub.")
                        st.rerun()
                
                # Document Processing (only show if credentials.json exists locally)
                if os.path.exists('credentials.json'):
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
                else:
                    st.info("Documents are already processed and available in chroma_db")

                st.subheader("Feedback Log")
                cloudish = os.getenv("STREAMLIT_SHARING_MODE") or os.getenv("STREAMLIT_SERVER_HEADLESS") == "true"
                log_entries = list(st.session_state.get("feedback_log") or [])

                # Fetch from Supabase if configured (persistent across reboots on Streamlit Cloud)
                if feedback_store_configured():
                    try:
                        remote = feedback_store_fetch(limit=50)
                        for row in remote:
                            ts = row.get("timestamp")
                            txt = row.get("feedback_text")
                            src = row.get("source")
                            fb_id = row.get("id")
                            if ts and txt:
                                log_entries.append({"timestamp": ts, "feedback": txt, "source": src, "id": fb_id})
                    except Exception:
                        pass

                # Fallback to local file (non-cloud only) - only add if not already from Supabase
                if not cloudish and os.path.exists("feedback_log.json"):
                    try:
                        with open("feedback_log.json", "r", encoding="utf-8") as f:
                            for line in f:
                                line = line.strip()
                                if line:
                                    entry = json.loads(line)
                                    # Skip if we already have this from Supabase (match by normalized timestamp+feedback)
                                    ts = (entry.get("timestamp") or "")[:20]
                                    txt = (entry.get("feedback") or "").strip()
                                    if ts and txt:
                                        already = any((e.get("timestamp") or "")[:20] == ts and (e.get("feedback") or "").strip() == txt for e in log_entries)
                                        if not already:
                                            log_entries.append(entry)
                    except Exception:
                        pass
                if log_entries:
                    st.info(f"Total feedback entries: {len(log_entries)}")
                    for entry in reversed(log_entries[-10:]):
                        ts = str(entry.get("timestamp", "Unknown"))
                        text = entry.get("feedback") or ""
                        fb_id = entry.get("id")
                        with st.expander(ts[:19]):
                            st.write(text)
                            st.caption(f"Source: {entry.get('source', entry.get('metadata', {}).get('source', 'Unknown'))}")
                            # Delete/Edit for Supabase entries
                            if fb_id and feedback_store_configured():
                                col1, col2 = st.columns(2)
                                with col1:
                                    if st.button("🗑️ Delete", key=f"del_admin_{fb_id}"):
                                        if feedback_store_delete(fb_id):
                                            st.success("Deleted from Supabase.")
                                            st.rerun()
                                        else:
                                            st.error("Delete failed.")
                                with col2:
                                    new_text = st.text_input("Edit", value=text, key=f"edit_admin_{fb_id}")
                                    if st.button("✏️ Save", key=f"save_admin_{fb_id}"):
                                        if feedback_store_update(fb_id, new_text):
                                            st.success("Updated in Supabase.")
                                            st.rerun()
                                        else:
                                            st.error("Update failed.")
                    if not cloudish:
                        st.download_button(
                            label="Download feedback log (JSONL)",
                            data="\n".join(json.dumps(e) for e in log_entries),
                            file_name="feedback_log.json",
                            mime="application/json",
                        )
                else:
                    st.info("No feedback entries yet")

                st.markdown("---")

                st.subheader("Collection Info")
                if st.button("Show Collection Stats", key="admin_show_stats"):
                    self.show_collection_stats()

                if st.button("Clear Collection", key="admin_clear_collection"):
                    if st.session_state.rag_pipeline:
                        st.session_state.rag_pipeline.clear_collection()
                        st.success("Collection cleared!")

                if st.button("Clear Local Feedback Log", key="admin_clear_local_feedback"):
                    try:
                        if os.path.exists("feedback_log.json"):
                            os.remove("feedback_log.json")
                            st.success("Local feedback_log.json deleted.")
                        else:
                            st.info("No local feedback_log.json found.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error deleting local log: {e}")

                if st.button("🔍 Inspect Contacts.xlsx Chunks", key="admin_inspect_contacts"):
                    if st.session_state.rag_pipeline:
                        try:
                            results = st.session_state.rag_pipeline.search("Contacts", n_results=20)
                            chunks = results.get('results', [])
                            contacts_chunks = [c for c in chunks if 'Contacts' in str(c.get('metadata', {}).get('file_name', ''))]
                            if contacts_chunks:
                                st.write(f"Found {len(contacts_chunks)} chunks from Contacts.xlsx:")
                                for i, c in enumerate(contacts_chunks[:10]):
                                    with st.expander(f"Chunk {i+1} (distance: {c.get('distance', 'n/a')})"):
                                        st.code(c.get('document', '')[:500])
                            else:
                                st.warning("No Contacts.xlsx chunks found. Try reprocessing documents.")
                        except Exception as e:
                            st.error(f"Error inspecting: {e}")

                st.markdown("---")

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

                if hasattr(st.session_state, 'local_llm_client') and hasattr(st.session_state, 'gemini_client'):
                    st.radio(
                        "LLM Provider",
                        options=["Google Gemini", "Local LLM"],
                        index=1 if st.session_state.get('use_local_llm') else 0,
                        key="admin_llm_provider",
                        help="Switch between local and cloud LLM"
                    )

                # Feedback Log (admin only)
                st.markdown("---")
                st.subheader("Feedback Log")
                cloudish = os.getenv("STREAMLIT_SHARING_MODE") or os.getenv("STREAMLIT_SERVER_HEADLESS") == "true"
                log_entries = list(st.session_state.get("feedback_log") or [])

                # Fetch from Supabase if configured (persistent across reboots on Streamlit Cloud)
                if feedback_store_configured():
                    try:
                        remote = feedback_store_fetch(limit=50)
                        for row in remote:
                            ts = row.get("timestamp")
                            txt = row.get("feedback_text")
                            src = row.get("source")
                            fb_id = row.get("id")
                            if ts and txt:
                                log_entries.append({"timestamp": ts, "feedback": txt, "source": src, "id": fb_id})
                    except Exception:
                        pass

                # Fallback to local file (non-cloud only) - only add if not already from Supabase
                if not cloudish and os.path.exists("feedback_log.json"):
                    try:
                        with open("feedback_log.json", "r", encoding="utf-8") as f:
                            for line in f:
                                line = line.strip()
                                if line:
                                    entry = json.loads(line)
                                    # Skip if we already have this from Supabase (match by normalized timestamp+feedback)
                                    ts = (entry.get("timestamp") or "")[:20]
                                    txt = (entry.get("feedback") or "").strip()
                                    if ts and txt:
                                        already = any((e.get("timestamp") or "")[:20] == ts and (e.get("feedback") or "").strip() == txt for e in log_entries)
                                        if not already:
                                            log_entries.append(entry)
                    except Exception:
                        pass
                if log_entries:
                    st.info(f"Total feedback entries: {len(log_entries)}")
                    for entry in reversed(log_entries[-10:]):
                        ts = str(entry.get("timestamp", "Unknown"))
                        text = entry.get("feedback") or ""
                        fb_id = entry.get("id")
                        with st.expander(ts[:19]):
                            st.write(text)
                            st.caption(f"Source: {entry.get('source', entry.get('metadata', {}).get('source', 'Unknown'))}")
                            # Delete/Edit for Supabase entries
                            if fb_id and feedback_store_configured():
                                col1, col2 = st.columns(2)
                                with col1:
                                    if st.button("ðŸ—‘ï¸ Delete", key=f"del_main_{fb_id}"):
                                        if feedback_store_delete(fb_id):
                                            st.success("Deleted from Supabase.")
                                            st.rerun()
                                        else:
                                            st.error("Delete failed.")
                                with col2:
                                    new_text = st.text_input("Edit", value=text, key=f"edit_{fb_id}")
                                    if st.button("âœï¸ Save", key=f"save_main_{fb_id}"):
                                        if feedback_store_update(fb_id, new_text):
                                            st.success("Updated in Supabase.")
                                            st.rerun()
                                        else:
                                            st.error("Update failed.")
                else:
                    st.info("No feedback entries yet")

        # Chat Interface
        st.subheader("Chat")

        # Chat messages
        if "show_feedback_dialog" not in st.session_state:
            st.session_state.show_feedback_dialog = False

        # Feedback dialog
        @st.dialog("Add Feedback to Knowledge Base")
        def feedback_dialog():
            fb_text = st.text_area(
                "Feedback or information to add",
                height=150,
                key="feedback_text_dialog",
                help="Adds text to the knowledge base (chunked and indexed)."
            )
            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("Cancel", use_container_width=True):
                    st.session_state.show_feedback_dialog = False
                    st.rerun()
            with col2:
                if st.button("Add to Knowledge Base", type="primary", use_container_width=True):
                    if fb_text.strip():
                        try:
                            metadata = {
                                "file_id": f"feedback::{datetime.now().strftime('%Y%m%d%H%M%S%f')}",
                                "file_name": f"feedback_{datetime.now().strftime('%Y-%m-%d_%H%M%S')}",
                                "source": "user_feedback",
                                "timestamp": datetime.now().isoformat(),
                                "type": "feedback",
                            }
                            success = st.session_state.rag_pipeline.add_document(
                                content=fb_text,
                                metadata=metadata,
                                chunk=True,
                                replace_existing=False,
                            )
                            if success:
                                entry = {
                                    "timestamp": metadata["timestamp"],
                                    "feedback": fb_text,
                                    "source": metadata["source"],
                                }
                                st.session_state.setdefault("feedback_log", []).append(entry)
                                if feedback_store_configured():
                                    feedback_store_add(fb_text, metadata)
                                cloudish = os.getenv("STREAMLIT_SHARING_MODE") or os.getenv("STREAMLIT_SERVER_HEADLESS") == "true"
                                if not cloudish:
                                    try:
                                        with open("feedback_log.json", "a", encoding="utf-8") as f:
                                            f.write(json.dumps(entry) + "\n")
                                    except Exception:
                                        pass
                                st.success("Feedback added to knowledge base.")
                                st.session_state.show_feedback_dialog = False
                                st.rerun()
                            else:
                                st.error("Failed to add feedback to knowledge base")
                        except Exception as e:
                            st.error(f"Error adding feedback: {e}")
                    else:
                        st.warning("Please enter some feedback text")

        # Feedback button (above chat, right side)
        st.markdown(
            """
            <style>
            .feedback-bottom-btn {
                display: flex;
                justify-content: flex-end;
                margin-bottom: 8px;
                padding-right: 8px;
            }
            .feedback-bottom-btn button {
                background: #ff4b4b;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 10px 20px;
                font-size: 14px;
                font-weight: 500;
                cursor: pointer;
                box-shadow: 0 2px 8px rgba(255,75,75,0.3);
            }
            .feedback-bottom-btn button:hover {
                background: #ff3333;
            }
            </style>
            <div class="feedback-bottom-btn">
            """,
            unsafe_allow_html=True,
        )
        if st.button("Add Feedback", key="feedback_bottom_btn", help="Add feedback to knowledge base"):
            st.session_state.show_feedback_dialog = True
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

        if st.session_state.show_feedback_dialog:
            feedback_dialog()

if __name__ == "__main__":
    chat_app = ChatInterface()
    chat_app.run()
