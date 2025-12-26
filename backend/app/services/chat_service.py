"""
Chat Service

Orchestrates the full chat flow:
1. Parse query
2. Validate entities
3. Get TFT prediction
4. Lookup trends
5. Generate response
6. Save to Supabase

All external calls are wrapped in try/except for resilience.
"""

import logging
from typing import Optional, Dict, Any
from pydantic import BaseModel

from ..core.supabase import (
    create_chat_session,
    save_message,
    update_session_title,
    get_session_messages
)
from .query_parser import parse_query, ParsedQuery
from .entity_validator import validate_entities, ValidatedEntities
from .tft_inference import predict as tft_predict, TFTPrediction, PriceForecast, RentForecast
from .trend_lookup import get_trends
from .response_generator import generate_response, GeneratedResponse

logger = logging.getLogger(__name__)

AREA_CONFIDENCE_THRESHOLD = 80.0
DEVELOPER_CONFIDENCE_THRESHOLD = 70.0


class ChatRequest(BaseModel):
    """Chat request from client."""
    query: str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    """Chat response to client."""
    session_id: str
    message_id: str
    content: str
    parsed_query: Optional[Dict[str, Any]] = None
    predictions: Optional[Dict[str, Any]] = None
    report_data: Optional[Dict[str, Any]] = None
    summary: str = ""


def _create_fallback_prediction() -> TFTPrediction:
    """Create a fallback prediction when TFT fails."""
    return TFTPrediction(
        price_forecast=PriceForecast(
            current_sqft=1200,
            forecast_sqft_low=1200,
            forecast_sqft_median=1350,
            forecast_sqft_high=1500,
            forecast_horizon_months=6,
            appreciation_percent=12.5
        ),
        rent_forecast=RentForecast(
            current_annual=None,
            forecast_annual_low=70000,
            forecast_annual_median=80000,
            forecast_annual_high=90000,
            has_actual_rent=False,
            estimated_yield_percent=6.0
        ),
        match_type="fallback",
        matched_group_id=None,
        confidence=20.0
    )


async def process_chat(
    user_id: str,
    query: str,
    session_id: Optional[str] = None
) -> ChatResponse:
    """
    Process a chat message through the full pipeline.
    
    Args:
        user_id: Authenticated user ID
        query: User's natural language query
        session_id: Existing session ID or None for new session
        
    Returns:
        ChatResponse with generated content and predictions
    """
    logger.info(f"Processing chat for user {user_id}: {query[:50]}...")
    
    # Step 1: Get or create session
    try:
        if not session_id:
            session = create_chat_session(user_id)
            if session:
                session_id = session.get("id", "temp_session")
            else:
                session_id = "temp_session"
            logger.info(f"Created new session: {session_id}")
    except Exception as e:
        logger.error(f"Error creating session: {e}")
        session_id = "temp_session"
    
    # Save user message
    try:
        save_message(session_id, "user", query)
    except Exception as e:
        logger.error(f"Error saving user message: {e}")
    
    # Step 2: Parse query
    try:
        parsed: ParsedQuery = await parse_query(query)
        logger.info(f"Parsed query: developer={parsed.developer}, area={parsed.area}, bedroom={parsed.bedroom}")
    except Exception as e:
        logger.error(f"Error parsing query: {e}")
        parsed = ParsedQuery(raw_query=query, confidence=0.0)
    
    # Step 3: Validate entities
    try:
        entities: ValidatedEntities = await validate_entities(
            developer=parsed.developer,
            area=parsed.area,
            bedroom=parsed.bedroom,
            property_type=parsed.property_type,
            reg_type=parsed.reg_type
        )
        logger.info(f"Validated: arabic_dev={entities.developer_arabic}, area={entities.area_name}, group_id={entities.group_id}")
    except Exception as e:
        logger.error(f"Error validating entities: {e}")
        entities = ValidatedEntities(
            area_name=parsed.area,
            bedroom=parsed.bedroom or "2BR",
            property_type=parsed.property_type or "Unit",
            reg_type=parsed.reg_type or "OffPlan",
            developer_arabic="Unknown"
        )
    
    # Step 4: Get TFT prediction
    try:
        prediction: TFTPrediction = await tft_predict(
            area=entities.area_name or "Business Bay",
            property_type=entities.property_type,
            bedroom=entities.bedroom or "2BR",
            reg_type=entities.reg_type or "OffPlan",
            developer=entities.developer_arabic or "Unknown",
            price=parsed.price,
            handover_months=parsed.handover_months,
            unit_sqft=parsed.unit_sqft
        )
        logger.info(f"TFT prediction: match_type={prediction.match_type}, confidence={prediction.confidence}")
    except Exception as e:
        logger.error(f"Error getting TFT prediction: {e}")
        prediction = _create_fallback_prediction()
    
    # Step 5: Lookup trends
    # For developer execution stats, prefer the user-facing developer name (brand) when available.
    # For building developers, model series may use a registered master developer, but execution stats (duration) should map from the brand.
    developer_for_trends = entities.developer_arabic
    if entities.is_building_developer and entities.building_developer_name:
        developer_for_trends = entities.building_developer_name
    elif developer_for_trends == "Unknown" and entities.developer_english:
        developer_for_trends = entities.developer_english

    # Confidence gating for lookups (fail closed)
    area_for_trends = entities.area_name if (entities.area_confidence or 0) >= AREA_CONFIDENCE_THRESHOLD else None
    bedroom_for_trends = entities.bedroom if area_for_trends else None
    if (entities.developer_confidence or 0) < DEVELOPER_CONFIDENCE_THRESHOLD:
        developer_for_trends = None
    
    try:
        trends = await get_trends(
            developer_arabic=developer_for_trends,
            area_name=area_for_trends,
            bedroom=bedroom_for_trends
        )
    except Exception as e:
        logger.error(f"Error looking up trends: {e}")
        trends = {"developer_stats": None, "area_stats": None, "rent_benchmark": None}
    
    # Step 6: Generate response
    try:
        response: GeneratedResponse = await generate_response(
            query=query,
            entities=entities,
            prediction=prediction,
            trends=trends,
            price=parsed.price,
            handover_months=parsed.handover_months,
            unit_sqft=parsed.unit_sqft
        )
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        # Create minimal response
        response = GeneratedResponse(
            content=f"I found some information about {entities.area_name or 'this area'}. Current market estimate: AED {prediction.price_forecast.forecast_sqft_median}/sqft with {prediction.price_forecast.appreciation_percent or 10}% forecast appreciation.",
            summary=f"Analysis for {entities.area_name or 'property'}",
            report_data={}
        )
    
    # Step 7: Save assistant message
    message_id = ""
    try:
        assistant_msg = save_message(
            session_id=session_id,
            role="assistant",
            content=response.content,
            parsed_query=parsed.model_dump(),
            predictions={
                "price_forecast": prediction.price_forecast.model_dump(),
                "rent_forecast": prediction.rent_forecast.model_dump(),
                "match_type": prediction.match_type,
                "confidence": prediction.confidence,
            },
            report_data=response.report_data
        )
        if assistant_msg:
            message_id = assistant_msg.get("id", "")
    except Exception as e:
        logger.error(f"Error saving assistant message: {e}")
    
    # Update session title if first message
    try:
        messages = get_session_messages(session_id)
        if messages and len(messages) <= 2:  # First exchange
            title = response.summary[:50] or query[:50]
            update_session_title(session_id, title)
    except Exception as e:
        logger.error(f"Error updating session title: {e}")
    
    return ChatResponse(
        session_id=session_id,
        message_id=message_id,
        content=response.content,
        parsed_query=parsed.model_dump(),
        predictions={
            "price_forecast": prediction.price_forecast.model_dump(),
            "rent_forecast": prediction.rent_forecast.model_dump(),
            "match_type": prediction.match_type,
            "confidence": prediction.confidence,
        },
        report_data=response.report_data,
        summary=response.summary
    )
