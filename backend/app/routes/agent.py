"""
Agent Settings API Routes

Handles agent profile and settings.
"""

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import Dict, Any, Optional
import uuid

from ..core.auth import get_current_user, User
from ..core.supabase import get_supabase_admin

router = APIRouter()


class AgentSettings(BaseModel):
    """Agent settings model."""
    name: Optional[str] = None
    company_name: Optional[str] = None
    phone: Optional[str] = None
    logo_url: Optional[str] = None
    primary_color: str = "#0f766e"
    secondary_color: str = "#10b981"
    report_footer: Optional[str] = None
    show_contact_info: bool = True


class AgentProfile(BaseModel):
    """Full agent profile."""
    id: str
    email: str
    name: Optional[str] = None
    company_name: Optional[str] = None
    phone: Optional[str] = None
    logo_url: Optional[str] = None
    primary_color: str = "#0f766e"
    secondary_color: str = "#10b981"
    report_footer: Optional[str] = None
    show_contact_info: bool = True


@router.get("/agent/profile", response_model=AgentProfile)
async def get_profile(
    user: User = Depends(get_current_user)
) -> AgentProfile:
    """
    Get agent profile and settings.
    
    Args:
        user: Authenticated user
        
    Returns:
        Agent profile
    """
    client = get_supabase_admin()
    
    result = client.table("users")\
        .select("*")\
        .eq("id", user.id)\
        .single()\
        .execute()
    
    if not result.data:
        # Create profile if not exists
        client.table("users").insert({
            "id": user.id,
            "email": user.email,
        }).execute()
        
        return AgentProfile(
            id=user.id,
            email=user.email,
        )
    
    data = result.data
    return AgentProfile(
        id=data["id"],
        email=data["email"],
        name=data.get("name"),
        company_name=data.get("company_name"),
        phone=data.get("phone"),
        logo_url=data.get("logo_url"),
        primary_color=data.get("primary_color", "#0f766e"),
        secondary_color=data.get("secondary_color", "#10b981"),
        report_footer=data.get("report_footer"),
        show_contact_info=data.get("show_contact_info", True),
    )


@router.put("/agent/settings", response_model=AgentProfile)
async def update_settings(
    settings: AgentSettings,
    user: User = Depends(get_current_user)
) -> AgentProfile:
    """
    Update agent settings.
    
    Args:
        settings: New settings
        user: Authenticated user
        
    Returns:
        Updated agent profile
    """
    client = get_supabase_admin()
    
    update_data = settings.model_dump(exclude_none=True)
    
    result = client.table("users")\
        .update(update_data)\
        .eq("id", user.id)\
        .execute()
    
    if not result.data:
        raise HTTPException(status_code=404, detail="Profile not found")
    
    data = result.data[0]
    return AgentProfile(
        id=data["id"],
        email=data["email"],
        name=data.get("name"),
        company_name=data.get("company_name"),
        phone=data.get("phone"),
        logo_url=data.get("logo_url"),
        primary_color=data.get("primary_color", "#0f766e"),
        secondary_color=data.get("secondary_color", "#10b981"),
        report_footer=data.get("report_footer"),
        show_contact_info=data.get("show_contact_info", True),
    )


@router.post("/agent/logo")
async def upload_logo(
    file: UploadFile = File(...),
    user: User = Depends(get_current_user)
) -> Dict[str, str]:
    """
    Upload agent logo to Supabase Storage.
    
    Args:
        file: Image file
        user: Authenticated user
        
    Returns:
        Public URL of uploaded logo
    """
    # Validate file type
    allowed_types = ["image/jpeg", "image/png", "image/webp", "image/gif"]
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {', '.join(allowed_types)}"
        )
    
    # Read file content
    content = await file.read()
    
    # Check file size (max 5MB)
    if len(content) > 5 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large. Max 5MB.")
    
    # Generate unique filename
    ext = file.filename.split(".")[-1] if file.filename else "png"
    filename = f"{user.id}/{uuid.uuid4()}.{ext}"
    
    client = get_supabase_admin()
    
    try:
        # Upload to Supabase Storage
        result = client.storage.from_("agent-logos").upload(
            filename,
            content,
            {"content-type": file.content_type}
        )
        
        # Get public URL
        public_url = client.storage.from_("agent-logos").get_public_url(filename)
        
        # Update user profile with logo URL
        client.table("users")\
            .update({"logo_url": public_url})\
            .eq("id", user.id)\
            .execute()
        
        return {"logo_url": public_url}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@router.delete("/agent/logo")
async def delete_logo(
    user: User = Depends(get_current_user)
) -> Dict[str, str]:
    """
    Delete agent logo.
    
    Args:
        user: Authenticated user
        
    Returns:
        Success message
    """
    client = get_supabase_admin()
    
    # Get current logo URL
    result = client.table("users")\
        .select("logo_url")\
        .eq("id", user.id)\
        .single()\
        .execute()
    
    if result.data and result.data.get("logo_url"):
        # Extract path from URL and delete from storage
        logo_url = result.data["logo_url"]
        # URL format: .../storage/v1/object/public/agent-logos/{path}
        try:
            path = logo_url.split("/agent-logos/")[-1]
            client.storage.from_("agent-logos").remove([path])
        except Exception:
            pass  # Ignore deletion errors
    
    # Clear logo URL in profile
    client.table("users")\
        .update({"logo_url": None})\
        .eq("id", user.id)\
        .execute()
    
    return {"status": "deleted"}

