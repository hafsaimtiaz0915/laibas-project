"""
Chat API Routes

Handles chat messages and property analysis.
"""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any

from ..core.auth import get_current_user, User
from ..services.chat_service import process_chat, ChatResponse

router = APIRouter()


class ChatRequest(BaseModel):
    """Request body for chat endpoint."""
    query: str
    session_id: Optional[str] = None


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    user: User = Depends(get_current_user)
) -> ChatResponse:
    """
    Process a property analysis chat message.
    
    Args:
        request: Chat request with query and optional session_id
        user: Authenticated user
        
    Returns:
        ChatResponse with analysis and predictions
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    try:
        response = await process_chat(
            user_id=user.id,
            query=request.query,
            session_id=request.session_id
        )
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing chat: {str(e)}")


@router.post("/chat/test")
async def chat_test(request: ChatRequest) -> Dict[str, Any]:
    """
    Test endpoint for chat without authentication.
    Useful for development and testing.
    """
    from ..services.query_parser import parse_query
    from ..services.entity_validator import validate_entities
    from ..services.tft_inference import predict as tft_predict
    from ..services.trend_lookup import get_trends
    from ..services.response_generator import generate_response
    from ..services.chat_service import AREA_CONFIDENCE_THRESHOLD, DEVELOPER_CONFIDENCE_THRESHOLD
    
    # Parse query
    parsed = await parse_query(request.query)
    
    # Validate entities
    entities = await validate_entities(
        developer=parsed.developer,
        area=parsed.area,
        bedroom=parsed.bedroom,
        property_type=parsed.property_type,
        reg_type=parsed.reg_type
    )
    
    # Get prediction
    prediction = await tft_predict(
        area=entities.area_name or "Unknown",
        property_type=entities.property_type or "Unit",
        bedroom=entities.bedroom or "2BR",
        reg_type=entities.reg_type or "OffPlan",
        developer=entities.developer_arabic or "Unknown",
        price=parsed.price,
        handover_months=parsed.handover_months,
        unit_sqft=parsed.unit_sqft
    )
    
    # Get trends - use English name if Arabic is Unknown (for building developers)
    developer_for_trends = entities.developer_arabic
    if developer_for_trends == "Unknown" and entities.developer_english:
        developer_for_trends = entities.developer_english
    
    # Confidence gating for lookups (fail closed)
    area_for_trends = entities.area_name if (entities.area_confidence or 0) >= AREA_CONFIDENCE_THRESHOLD else None
    bedroom_for_trends = entities.bedroom if area_for_trends else None
    if (entities.developer_confidence or 0) < DEVELOPER_CONFIDENCE_THRESHOLD:
        developer_for_trends = None
    
    trends = await get_trends(
        developer_arabic=developer_for_trends,
        area_name=area_for_trends,
        bedroom=bedroom_for_trends
    )
    
    # Generate response
    response = await generate_response(
        query=request.query,
        entities=entities,
        prediction=prediction,
        trends=trends,
        price=parsed.price,
        handover_months=parsed.handover_months,
        unit_sqft=parsed.unit_sqft
    )
    
    return {
        "query": request.query,
        "parsed": parsed.model_dump(),
        "entities": entities.model_dump(),
        "prediction": {
            "price_forecast": prediction.price_forecast.model_dump(),
            "rent_forecast": prediction.rent_forecast.model_dump(),
            "match_type": prediction.match_type,
            "confidence": prediction.confidence,
        },
        "trends": {
            "developer": trends.get("developer_stats").model_dump() if trends.get("developer_stats") else None,
            "area": trends.get("area_stats").model_dump() if trends.get("area_stats") else None,
            "rent": trends.get("rent_benchmark").model_dump() if trends.get("rent_benchmark") else None,
            "lookup_audit": trends.get("lookup_audit"),
        },
        "response": response.content,
        "report_data": response.report_data,
    }

