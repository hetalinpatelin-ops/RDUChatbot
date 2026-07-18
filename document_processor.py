#!/usr/bin/env python3
"""
Document processor for Google Drive files.
Supports PDFs, DOCX, Google Docs, and TXT files.
"""

import os
import re
import logging
from typing import List, Dict, Optional
from pathlib import Path

import fitz  # PyMuPDF
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import io
from docx import Document as DocxDocument

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(self, drive_service=None):
        self.drive_service = drive_service
    
    def extract_text_from_pdf(self, file_path: str) -> str:
        """Extract text from PDF files."""
        try:
            text = ""
            with fitz.open(file_path) as doc:
                for page in doc:
                    text += page.get_text()
            return text
        except Exception as e:
            logger.error(f"Error extracting PDF text: {e}")
            return ""
    
    def extract_text_from_docx(self, file_path: str) -> str:
        """Extract text from DOCX files."""
        try:
            doc = DocxDocument(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            logger.error(f"Error extracting DOCX text: {e}")
            return ""
    
    def extract_text_from_google_doc(self, file_id: str) -> str:
        """Extract text from Google Docs via API."""
        try:
            if not self.drive_service:
                raise ValueError("Drive service not initialized")
            
            request = self.drive_service.files().export_media(
                fileId=file_id,
                mimeType='text/plain'
            )
            
            downloaded = io.BytesIO()
            downloader = MediaIoBaseDownload(downloaded, request)
            done = False
            
            while not done:
                status, done = downloader.next_chunk()
            
            return downloaded.getvalue().decode('utf-8')
        except Exception as e:
            logger.error(f"Error extracting Google Doc text: {e}")
            return ""
    
    def extract_text_from_txt(self, file_path: str) -> str:
        """Extract text from TXT files."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Error extracting TXT text: {e}")
            return ""
    
    def clean_text(self, text: str) -> str:
        """Clean extracted text."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters that might cause issues
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)\[\]\{\}\"\']', '', text)
        return text.strip()
    
    def process_file(self, file_info: Dict, download_dir: str = "downloads") -> Optional[Dict]:
        """Process a single file and return extracted text metadata."""
        try:
            file_id = file_info['id']
            file_name = file_info['name']
            mime_type = file_info.get('mimeType', '')
            
            logger.info(f"Processing file: {file_name}")
            
            # Create download directory if it doesn't exist
            os.makedirs(download_dir, exist_ok=True)
            
            # Determine file type and extraction method
            if mime_type == 'application/pdf':
                # Download PDF
                file_path = os.path.join(download_dir, file_name)
                self._download_file(file_id, file_path)
                text = self.extract_text_from_pdf(file_path)
                os.remove(file_path)  # Clean up
                
            elif mime_type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
                # Download DOCX
                file_path = os.path.join(download_dir, file_name)
                self._download_file(file_id, file_path)
                text = self.extract_text_from_docx(file_path)
                os.remove(file_path)  # Clean up
                
            elif mime_type == 'application/vnd.google-apps.document':
                # Google Doc - extract directly
                text = self.extract_text_from_google_doc(file_id)
                
            elif mime_type == 'text/plain':
                # Download TXT
                file_path = os.path.join(download_dir, file_name)
                self._download_file(file_id, file_path)
                text = self.extract_text_from_txt(file_path)
                os.remove(file_path)  # Clean up
                
            else:
                logger.warning(f"Unsupported file type: {mime_type}")
                return None
            
            # Clean and validate text
            text = self.clean_text(text)
            
            if not text or len(text.strip()) < 10:
                logger.warning(f"Empty or too short text extracted from {file_name}")
                return None
            
            return {
                'file_id': file_id,
                'file_name': file_name,
                'mime_type': mime_type,
                'text': text,
                'char_count': len(text),
                'word_count': len(text.split())
            }
            
        except Exception as e:
            logger.error(f"Error processing file {file_info.get('name', 'unknown')}: {e}")
            return None
    
    def _download_file(self, file_id: str, file_path: str):
        """Download a file from Google Drive."""
        if not self.drive_service:
            raise ValueError("Drive service not initialized")
        
        request = self.drive_service.files().get_media(fileId=file_id)
        downloaded = io.BytesIO()
        downloader = MediaIoBaseDownload(downloaded, request)
        done = False
        
        while not done:
            status, done = downloader.next_chunk()
        
        with open(file_path, 'wb') as f:
            f.write(downloaded.getvalue())
    
    def list_drive_files(self, folder_id: str = None, recursive: bool = True) -> List[Dict]:
        """List files in Google Drive folder, optionally recursing into subfolders."""
        if not self.drive_service:
            raise ValueError("Drive service not initialized")
        
        try:
            supported_types = [
                'application/pdf',
                'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
                'application/vnd.google-apps.document',
                'text/plain'
            ]
            supported_folder_types = [
                'application/vnd.google-apps.folder'
            ]
            
            query = f"'{folder_id}' in parents" if folder_id else "'root' in parents"
            query += " and trashed = false"
            
            files = []
            page_token = None
            
            while True:
                results = self.drive_service.files().list(
                    q=query,
                    spaces='drive',
                    fields='nextPageToken, files(id, name, mimeType, modifiedTime, size)',
                    pageToken=page_token
                ).execute()
                
                items = results.get('files', [])
                
                for item in items:
                    mime = item.get('mimeType', '')
                    if mime in supported_types:
                        files.append(item)
                    elif recursive and mime in supported_folder_types:
                        # Recurse into subfolder
                        sub_files = self.list_drive_files(folder_id=item['id'], recursive=True)
                        files.extend(sub_files)
                
                page_token = results.get('nextPageToken')
                if not page_token:
                    break
            
            return files
            
        except Exception as e:
            logger.error(f"Error listing Drive files: {e}")
            return []
    
    def get_file_metadata(self, file_id: str) -> Dict:
        """Get metadata for a specific file."""
        if not self.drive_service:
            raise ValueError("Drive service not initialized")
        
        try:
            file = self.drive_service.files().get(
                fileId=file_id,
                fields='id, name, mimeType, modifiedTime, size, parents'
            ).execute()
            return file
        except Exception as e:
            logger.error(f"Error getting file metadata: {e}")
            return {}