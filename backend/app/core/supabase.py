"""
Supabase Client Configuration

Provides configured Supabase clients for auth and database operations.
Using supabase 1.x which uses requests library (no proxy issues).
"""

from supabase import create_client, Client
from functools import lru_cache
from .config import get_settings


@lru_cache()
def get_supabase_client() -> Client:
    """
    Get Supabase client with anon key.
    Used for client-side operations (with JWT).
    """
    settings = get_settings()
    return create_client(settings.supabase_url, settings.supabase_anon_key)


@lru_cache()
def get_supabase_admin() -> Client:
    """
    Get Supabase client with service role key.
    Used for server-side operations (bypasses RLS).
    """
    settings = get_settings()
    return create_client(settings.supabase_url, settings.supabase_service_key)


# Convenience functions for common operations
def get_user_by_id(user_id: str) -> dict | None:
    """Get user profile by ID."""
    client = get_supabase_admin()
    result = client.table("users").select("*").eq("id", user_id).single().execute()
    return result.data if result.data else None


def create_chat_session(user_id: str, title: str = "New Chat") -> dict:
    """Create a new chat session."""
    client = get_supabase_admin()
    result = client.table("chat_sessions").insert({
        "user_id": user_id,
        "title": title
    }).execute()
    return result.data[0] if result.data else None


def get_user_sessions(user_id: str, limit: int = 50) -> list:
    """Get user's chat sessions ordered by most recent."""
    client = get_supabase_admin()
    result = client.table("chat_sessions")\
        .select("*")\
        .eq("user_id", user_id)\
        .order("updated_at", desc=True)\
        .limit(limit)\
        .execute()
    return result.data or []


def get_session_messages(session_id: str) -> list:
    """Get all messages for a session."""
    client = get_supabase_admin()
    result = client.table("chat_messages")\
        .select("*")\
        .eq("session_id", session_id)\
        .order("created_at")\
        .execute()
    return result.data or []


def save_message(
    session_id: str,
    role: str,
    content: str,
    parsed_query: dict = None,
    predictions: dict = None,
    report_data: dict = None
) -> dict:
    """Save a chat message."""
    client = get_supabase_admin()
    message_data = {
        "session_id": session_id,
        "role": role,
        "content": content,
    }
    if parsed_query:
        message_data["parsed_query"] = parsed_query
    if predictions:
        message_data["predictions"] = predictions
    if report_data:
        message_data["report_data"] = report_data
    
    result = client.table("chat_messages").insert(message_data).execute()
    return result.data[0] if result.data else None


def update_session_title(session_id: str, title: str) -> None:
    """Update session title."""
    client = get_supabase_admin()
    client.table("chat_sessions").update({"title": title}).eq("id", session_id).execute()


def delete_session(session_id: str) -> None:
    """Delete a session and its messages (cascades)."""
    client = get_supabase_admin()
    client.table("chat_sessions").delete().eq("id", session_id).execute()

