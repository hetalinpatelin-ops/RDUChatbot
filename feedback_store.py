#!/usr/bin/env python3
"""
Optional Supabase-backed feedback persistence for Streamlit Cloud.
Falls back to local-only if not configured.
"""

import os
import json
import logging
from datetime import datetime
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

_SUPABASE_URL = os.getenv("SUPABASE_URL") or os.getenv("supabase_url")
_SUPABASE_KEY = os.getenv("SUPABASE_ANON_KEY") or os.getenv("SUPABASE_SERVICE_KEY") or os.getenv("supabase_key")
_TABLE = "kb_feedback"


def _client():
    if not _SUPABASE_URL or not _SUPABASE_KEY:
        return None
    try:
        from supabase import create_client
        return create_client(_SUPABASE_URL, _SUPABASE_KEY)
    except Exception as e:
        logger.warning("Supabase client init failed: %s", e)
        return None


def is_configured() -> bool:
    return _client() is not None


def add_feedback(text: str, metadata: Dict) -> bool:
    """Persist feedback to Supabase."""
    client = _client()
    if not client:
        return False
    try:
        row = {
            "feedback_text": text,
            "source": metadata.get("source", "user_feedback"),
            "file_id": metadata.get("file_id"),
            "file_name": metadata.get("file_name"),
            "timestamp": metadata.get("timestamp") or datetime.now().isoformat(),
            "meta": metadata,
        }
        client.table(_TABLE).insert(row).execute()
        return True
    except Exception as e:
        logger.error("Supabase insert failed: %s", e)
        return False


def fetch_recent(limit: int = 50) -> List[Dict]:
    """Fetch recent feedback entries from Supabase."""
    client = _client()
    if not client:
        return []
    try:
        res = client.table(_TABLE).select("*").order("timestamp", desc=True).limit(limit).execute()
        return res.data or []
    except Exception as e:
        logger.error("Supabase fetch failed: %s", e)
        return []


def delete_feedback(feedback_id: int) -> bool:
    """Delete a feedback entry from Supabase by its ID."""
    client = _client()
    if not client:
        return False
    try:
        client.table(_TABLE).delete().eq("id", feedback_id).execute()
        return True
    except Exception as e:
        logger.error("Supabase delete failed: %s", e)
        return False


def update_feedback(feedback_id: int, new_text: str) -> bool:
    """Update a feedback entry's text in Supabase."""
    client = _client()
    if not client:
        return False
    try:
        client.table(_TABLE).update({"feedback_text": new_text, "timestamp": datetime.now().isoformat()}).eq("id", feedback_id).execute()
        return True
    except Exception as e:
        logger.error("Supabase update failed: %s", e)
        return False


def rehydrate_chroma(rag_pipeline) -> int:
    """Load feedback from Supabase into local ChromaDB on startup."""
    client = _client()
    if not client or not rag_pipeline:
        return 0
    try:
        res = client.table(_TABLE).select("*").order("timestamp", desc=False).execute()
        rows = res.data or []
        count = 0
        for row in rows:
            text = row.get("feedback_text") or ""
            if not text.strip():
                continue
            metadata = row.get("meta") or {
                "file_id": row.get("file_id"),
                "file_name": row.get("file_name"),
                "source": row.get("source", "user_feedback"),
                "timestamp": row.get("timestamp"),
            }
            # Ensure file_id
            metadata["file_id"] = metadata.get("file_id") or f"feedback::{row.get('id', datetime.now().timestamp())}"
            ok = rag_pipeline.add_document(text, metadata, chunk=True, replace_existing=False)
            if ok:
                count += 1
        if count:
            logger.info("Rehydrated %d feedback entries from Supabase into ChromaDB", count)
        return count
    except Exception as e:
        logger.error("Supabase rehydrate failed: %s", e)
        return 0