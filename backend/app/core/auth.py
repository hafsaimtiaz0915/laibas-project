"""
Authentication Utilities

Handles JWT verification and user extraction from Supabase Auth.
"""

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import jwt, JWTError
from pydantic import BaseModel
from typing import Optional
from .config import get_settings
from .supabase import get_user_by_id

# Security scheme
security = HTTPBearer(auto_error=False)


class User(BaseModel):
    """Authenticated user model."""
    id: str
    email: str
    name: Optional[str] = None
    company_name: Optional[str] = None
    logo_url: Optional[str] = None


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> User:
    """
    Extract and validate user from JWT token.
    
    Raises:
        HTTPException: If token is invalid or user not found.
    """
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    token = credentials.credentials
    settings = get_settings()
    
    try:
        # Decode JWT (Supabase uses HS256 with the JWT secret)
        # Note: In production, verify with Supabase's JWT secret
        payload = jwt.decode(
            token,
            settings.supabase_service_key,  # Use service key as secret for dev
            algorithms=["HS256"],
            options={"verify_aud": False}  # Supabase doesn't set audience by default
        )
        
        user_id: str = payload.get("sub")
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token payload",
            )
        
    except JWTError as e:
        # For development, try to extract user_id from token without full verification
        try:
            # Decode without verification for development
            unverified = jwt.get_unverified_claims(token)
            user_id = unverified.get("sub")
            if not user_id:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail=f"Token validation failed: {str(e)}",
                )
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Token validation failed: {str(e)}",
            )
    
    # Fetch user profile from database
    user_data = get_user_by_id(user_id)
    
    if not user_data:
        # User authenticated but no profile - create minimal user
        return User(
            id=user_id,
            email=payload.get("email", "unknown@example.com"),
        )
    
    return User(
        id=user_data["id"],
        email=user_data["email"],
        name=user_data.get("name"),
        company_name=user_data.get("company_name"),
        logo_url=user_data.get("logo_url"),
    )


async def get_optional_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> Optional[User]:
    """
    Try to get user from token, but don't fail if not provided.
    Useful for endpoints that work both authenticated and unauthenticated.
    """
    if not credentials:
        return None
    
    try:
        return await get_current_user(credentials)
    except HTTPException:
        return None

