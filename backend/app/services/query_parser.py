"""
Query Parser Service

Uses OpenAI tools API to extract property query parameters from natural language.
"""

import json
import logging
from typing import Optional
from datetime import datetime
import re
from pydantic import BaseModel
from openai import OpenAI

from ..core.config import get_settings

logger = logging.getLogger(__name__)


class ParsedQuery(BaseModel):
    """Structured property query extracted from user input."""
    developer: Optional[str] = None
    area: Optional[str] = None
    bedroom: Optional[str] = None
    price: Optional[float] = None
    unit_sqft: Optional[float] = None  # Unit size in square feet
    property_type: Optional[str] = None  # "Unit", "Villa"
    reg_type: Optional[str] = None  # "OffPlan", "Ready"
    handover_months: Optional[int] = None
    raw_query: str = ""
    confidence: float = 0.0


# OpenAI tool definition for property query extraction
PROPERTY_QUERY_TOOL = {
    "type": "function",
    "function": {
        "name": "extract_property_query",
        "description": "Extract property search parameters from a user's natural language query about Dubai real estate.",
        "parameters": {
            "type": "object",
            "properties": {
                "developer": {
                    "type": "string",
                    "description": "The property developer name (e.g., 'Emaar', 'Binghatti', 'Damac', 'Sobha', 'Nakheel')"
                },
                "area": {
                    "type": "string",
                    "description": "The area or community name (e.g., 'JVC', 'Dubai Marina', 'Business Bay', 'Downtown Dubai')"
                },
                "bedroom": {
                    "type": "string",
                    "description": "Number of bedrooms (e.g., 'Studio', '1BR', '2BR', '3BR', '4BR', '5BR', 'Penthouse')"
                },
                "price": {
                    "type": "number",
                    "description": "The property price in AED. Convert millions (e.g., '2.2M' = 2200000)"
                },
                "unit_sqft": {
                    "type": "number",
                    "description": "The property size in square feet (sqft). Convert if given in square meters (1 sqm = 10.764 sqft)"
                },
                "property_type": {
                    "type": "string",
                    "enum": ["Unit", "Villa"],
                    "description": "Type of property - 'Unit' for apartments/flats, 'Villa' for houses/villas/townhouses"
                },
                "reg_type": {
                    "type": "string",
                    "enum": ["OffPlan", "Ready"],
                    "description": "Registration type - 'OffPlan' for under construction/new launches, 'Ready' for completed/resale"
                },
                "handover_months": {
                    "type": "integer",
                    "description": "Expected months until handover (for off-plan properties)"
                }
            },
            "required": []
        }
    }
}


def infer_reg_type(query: str, parsed: dict) -> Optional[str]:
    """
    Infer registration type from query keywords if not explicitly extracted.
    """
    query_lower = query.lower()
    
    # Keywords indicating off-plan
    offplan_keywords = [
        "off-plan", "offplan", "off plan", "under construction",
        "launching", "launch", "pre-launch", "new project",
        "handover", "completion", "expected", "phase", "payment plan"
    ]
    
    # Keywords indicating ready
    ready_keywords = [
        "ready", "completed", "finished", "move-in",
        "immediate", "existing", "resale", "secondary"
    ]
    
    # Check keywords
    for keyword in offplan_keywords:
        if keyword in query_lower:
            return "OffPlan"
    
    for keyword in ready_keywords:
        if keyword in query_lower:
            return "Ready"
    
    # If handover months specified, likely off-plan
    if parsed.get("handover_months"):
        return "OffPlan"
    
    # If developer is mentioned without "ready", often off-plan in Dubai context
    if parsed.get("developer") and not any(kw in query_lower for kw in ready_keywords):
        return "OffPlan"
    
    # Default to None (let entity validator decide)
    return None


def infer_handover_months(query: str) -> Optional[int]:
    """
    Infer handover months from year/month mentions when not explicitly extracted.
    Heuristic:
    - If query contains a year (e.g., 2026), assume handover at June of that year unless a month is specified.
    - If month is specified (e.g., "Mar 2026"), use that month.
    """
    q = query.lower()
    now = datetime.utcnow()

    # Look for explicit "handover in X months"
    m = re.search(r"handover\\s*(in)?\\s*(\\d{1,2})\\s*months?", q)
    if m:
        try:
            return int(m.group(2))
        except Exception:
            pass

    # Find year
    year_match = re.search(r"\\b(20\\d{2})\\b", q)
    if not year_match:
        return None
    year = int(year_match.group(1))

    # Optional month parsing
    month_map = {
        "jan": 1, "january": 1,
        "feb": 2, "february": 2,
        "mar": 3, "march": 3,
        "apr": 4, "april": 4,
        "may": 5,
        "jun": 6, "june": 6,
        "jul": 7, "july": 7,
        "aug": 8, "august": 8,
        "sep": 9, "sept": 9, "september": 9,
        "oct": 10, "october": 10,
        "nov": 11, "november": 11,
        "dec": 12, "december": 12,
    }
    month = 6  # default mid-year
    for key, val in month_map.items():
        if re.search(rf"\\b{re.escape(key)}\\b\\s*{year}\\b", q):
            month = val
            break

    target = datetime(year=year, month=month, day=15)
    delta_months = (target.year - now.year) * 12 + (target.month - now.month)
    if delta_months < 0:
        return None
    return int(delta_months)


def normalize_bedroom(bedroom: str) -> Optional[str]:
    """Normalize bedroom string to standard format."""
    if not bedroom:
        return None
    
    bedroom_lower = bedroom.lower().strip()
    
    # Map common variations
    mapping = {
        "studio": "Studio",
        "0": "Studio",
        "0br": "Studio",
        "0 bed": "Studio",
        "1": "1BR",
        "1br": "1BR",
        "1 bed": "1BR",
        "1 bedroom": "1BR",
        "one bed": "1BR",
        "one bedroom": "1BR",
        "2": "2BR",
        "2br": "2BR",
        "2 bed": "2BR",
        "2 bedroom": "2BR",
        "two bed": "2BR",
        "two bedroom": "2BR",
        "3": "3BR",
        "3br": "3BR",
        "3 bed": "3BR",
        "3 bedroom": "3BR",
        "three bed": "3BR",
        "three bedroom": "3BR",
        "4": "4BR",
        "4br": "4BR",
        "4 bed": "4BR",
        "4 bedroom": "4BR",
        "four bed": "4BR",
        "four bedroom": "4BR",
        "5": "5BR",
        "5br": "5BR",
        "5 bed": "5BR",
        "5 bedroom": "5BR",
        "five bed": "5BR",
        "five bedroom": "5BR",
        "penthouse": "Penthouse",
        "ph": "Penthouse",
    }
    
    return mapping.get(bedroom_lower, bedroom)


async def parse_query(query: str) -> ParsedQuery:
    """
    Parse a natural language property query into structured parameters.
    
    Args:
        query: User's natural language query
        
    Returns:
        ParsedQuery with extracted parameters
    """
    settings = get_settings()
    
    if not settings.openai_api_key:
        logger.warning("OpenAI API key not configured, returning empty parse")
        return ParsedQuery(raw_query=query, confidence=0.0)
    
    client = OpenAI(api_key=settings.openai_api_key)
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",  # Use latest GPT-4o model
            messages=[
                {
                    "role": "system",
                    "content": """You are a Dubai real estate query parser. Extract property search parameters from user queries.
                    
Common patterns:
- "Binghatti JVC 2BR at 2.2M 900sqft" → developer: Binghatti, area: JVC, bedroom: 2BR, price: 2200000, unit_sqft: 900
- "3 bed Dubai Marina ready 1500 sq ft" → bedroom: 3BR, area: Dubai Marina, reg_type: Ready, unit_sqft: 1500
- "Emaar off-plan Business Bay 1BR" → developer: Emaar, area: Business Bay, bedroom: 1BR, reg_type: OffPlan
- "villa in Arabian Ranches" → property_type: Villa, area: Arabian Ranches
- "2BR 1200 sqft at 1.8M" → bedroom: 2BR, unit_sqft: 1200, price: 1800000

Area abbreviations:
- JVC = Jumeirah Village Circle
- JVT = Jumeirah Village Triangle
- JLT = Jumeirah Lake Towers
- BB = Business Bay
- DM = Dubai Marina
- DH = Dubai Hills Estate
- MBR = Mohammed Bin Rashid City

Always convert price mentions like "2.2M" or "2.2 million" to full numbers (2200000)."""
                },
                {
                    "role": "user",
                    "content": query
                }
            ],
            tools=[PROPERTY_QUERY_TOOL],
            tool_choice={"type": "function", "function": {"name": "extract_property_query"}},
            # Determinism: same input text should yield the same structured extraction.
            # (This is critical because downstream group matching + horizons can swing forecasts materially.)
            temperature=0.0
        )
        
        # Extract tool call result
        if response.choices[0].message.tool_calls:
            tool_call = response.choices[0].message.tool_calls[0]
            parsed = json.loads(tool_call.function.arguments)
            
            # Normalize bedroom
            if parsed.get("bedroom"):
                parsed["bedroom"] = normalize_bedroom(parsed["bedroom"])
            
            # Infer reg_type if not extracted
            if not parsed.get("reg_type"):
                parsed["reg_type"] = infer_reg_type(query, parsed)

            # Infer handover months if not extracted
            if not parsed.get("handover_months"):
                parsed["handover_months"] = infer_handover_months(query)
            
            # Calculate confidence based on fields extracted
            extracted_fields = sum(1 for k, v in parsed.items() if v is not None)
            confidence = min(extracted_fields / 5 * 100, 100)  # Max 100%
            
            return ParsedQuery(
                developer=parsed.get("developer"),
                area=parsed.get("area"),
                bedroom=parsed.get("bedroom"),
                price=parsed.get("price"),
                unit_sqft=parsed.get("unit_sqft"),
                property_type=parsed.get("property_type", "Unit"),
                reg_type=parsed.get("reg_type"),
                handover_months=parsed.get("handover_months"),
                raw_query=query,
                confidence=confidence
            )
        
        return ParsedQuery(raw_query=query, confidence=0.0)
        
    except Exception as e:
        logger.error(f"Error parsing query: {e}")
        return ParsedQuery(raw_query=query, confidence=0.0)


# Test cases for validation
TEST_CASES = [
    ("Binghatti JVC 2BR at 2.2M", {"developer": "Binghatti", "area": "JVC", "bedroom": "2BR", "price": 2200000}),
    ("ready 3 bed Dubai Marina", {"area": "Dubai Marina", "bedroom": "3BR", "reg_type": "Ready"}),
    ("Emaar off-plan Business Bay 1BR", {"developer": "Emaar", "area": "Business Bay", "bedroom": "1BR", "reg_type": "OffPlan"}),
    ("villa in Arabian Ranches", {"property_type": "Villa", "area": "Arabian Ranches"}),
]
