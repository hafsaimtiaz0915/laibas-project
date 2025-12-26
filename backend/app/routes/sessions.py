"""
Sessions API Routes

Handles chat session management.
"""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any

from ..core.auth import get_current_user, User
from ..core.supabase import (
    get_user_sessions,
    get_session_messages,
    delete_session,
    get_supabase_admin
)

router = APIRouter()


class SessionSummary(BaseModel):
    """Session summary for list view."""
    id: str
    title: str
    created_at: str
    updated_at: str


class Message(BaseModel):
    """Chat message."""
    id: str
    role: str
    content: str
    parsed_query: Dict[str, Any] | None = None
    predictions: Dict[str, Any] | None = None
    report_data: Dict[str, Any] | None = None
    report_id: str | None = None
    pdf_url: str | None = None
    pdf_storage_path: str | None = None
    created_at: str


class SessionDetail(BaseModel):
    """Session with messages."""
    id: str
    title: str
    created_at: str
    updated_at: str
    messages: List[Message]


@router.get("/sessions", response_model=List[SessionSummary])
async def list_sessions(
    user: User = Depends(get_current_user),
    limit: int = 50
) -> List[SessionSummary]:
    """
    List user's chat sessions.
    
    Args:
        user: Authenticated user
        limit: Maximum number of sessions to return
        
    Returns:
        List of session summaries
    """
    sessions = get_user_sessions(user.id, limit=limit)
    
    return [
        SessionSummary(
            id=s["id"],
            title=s["title"],
            created_at=s["created_at"],
            updated_at=s["updated_at"]
        )
        for s in sessions
    ]


@router.get("/sessions/{session_id}", response_model=SessionDetail)
async def get_session(
    session_id: str,
    user: User = Depends(get_current_user)
) -> SessionDetail:
    """
    Get session details with messages.
    
    Args:
        session_id: Session UUID
        user: Authenticated user
        
    Returns:
        Session with all messages
    """
    # Verify session belongs to user
    client = get_supabase_admin()
    result = client.table("chat_sessions")\
        .select("*")\
        .eq("id", session_id)\
        .eq("user_id", user.id)\
        .execute()
    
    if not result.data or len(result.data) == 0:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = result.data[0]
    messages = get_session_messages(session_id)

    # Attach report metadata (if any) so the UI can show Download PDF in chat history
    report_by_message_id: Dict[str, Dict[str, Any]] = {}
    try:
        msg_ids = [m["id"] for m in messages if m.get("id")]
        if msg_ids:
            reports = client.table("reports")\
                .select("id, message_id, pdf_url, pdf_storage_path")\
                .eq("user_id", user.id)\
                .in_("message_id", msg_ids)\
                .execute()
            for r in (reports.data or []):
                mid = r.get("message_id")
                if mid:
                    report_by_message_id[mid] = r
    except Exception:
        report_by_message_id = {}
    
    return SessionDetail(
        id=session["id"],
        title=session["title"],
        created_at=session["created_at"],
        updated_at=session["updated_at"],
        messages=[
            Message(
                id=m["id"],
                role=m["role"],
                content=m["content"],
                parsed_query=m.get("parsed_query"),
                predictions=m.get("predictions"),
                report_data=m.get("report_data"),
                report_id=report_by_message_id.get(m["id"], {}).get("id"),
                pdf_url=report_by_message_id.get(m["id"], {}).get("pdf_url"),
                pdf_storage_path=report_by_message_id.get(m["id"], {}).get("pdf_storage_path"),
                created_at=m["created_at"]
            )
            for m in messages
        ]
    )


@router.delete("/sessions/{session_id}")
async def remove_session(
    session_id: str,
    user: User = Depends(get_current_user)
) -> Dict[str, str]:
    """
    Delete a chat session.
    
    Args:
        session_id: Session UUID
        user: Authenticated user
        
    Returns:
        Success message
    """
    # Verify session belongs to user
    client = get_supabase_admin()
    result = client.table("chat_sessions")\
        .select("id")\
        .eq("id", session_id)\
        .eq("user_id", user.id)\
        .single()\
        .execute()
    
    if not result.data:
        raise HTTPException(status_code=404, detail="Session not found")
    
    delete_session(session_id)
    
    return {"status": "deleted", "session_id": session_id}


@router.put("/sessions/{session_id}/title")
async def update_title(
    session_id: str,
    title: str,
    user: User = Depends(get_current_user)
) -> Dict[str, str]:
    """
    Update session title.
    
    Args:
        session_id: Session UUID
        title: New title
        user: Authenticated user
        
    Returns:
        Success message
    """
    # Verify session belongs to user
    client = get_supabase_admin()
    result = client.table("chat_sessions")\
        .select("id")\
        .eq("id", session_id)\
        .eq("user_id", user.id)\
        .single()\
        .execute()
    
    if not result.data:
        raise HTTPException(status_code=404, detail="Session not found")
    
    client.table("chat_sessions")\
        .update({"title": title})\
        .eq("id", session_id)\
        .execute()
    
    return {"status": "updated", "title": title}

