# LLM Service Code

> **Document Version**: 2.0  
> **Last Updated**: 2025-12-11  
> **Purpose**: Complete code for LLM services (aligned with plan v2.0)

---

## Overview

This document contains implementation code for all LLM services, including:
- Supabase integration
- Arabic developer name resolution
- TFT inference with proper input construction
- Updated OpenAI tools API

---

## 1. Configuration

**File**: `backend/app/core/config.py`

```python
from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    # OpenAI
    OPENAI_API_KEY: str
    OPENAI_MODEL: str = "gpt-4-turbo-preview"
    
    # Supabase
    SUPABASE_URL: str
    SUPABASE_ANON_KEY: str
    SUPABASE_SERVICE_KEY: str
    
    # TFT Model
    TFT_MODEL_PATH: str = "models/tft_final.ckpt"
    TFT_DATA_PATH: str = "Data/tft/tft_training_data.csv"
    
    # Lookups
    LOOKUPS_PATH: str = "Data/lookups"
    
    class Config:
        env_file = ".env"

@lru_cache()
def get_settings():
    return Settings()

settings = get_settings()
```

---

## 2. Supabase Client

**File**: `backend/app/core/supabase.py`

```python
from supabase import create_client, Client
from app.core.config import settings

def get_supabase() -> Client:
    """Get Supabase client for backend operations."""
    return create_client(
        settings.SUPABASE_URL,
        settings.SUPABASE_SERVICE_KEY  # Service key for backend
    )

# For frontend/auth operations (anon key)
def get_supabase_anon() -> Client:
    return create_client(
        settings.SUPABASE_URL,
        settings.SUPABASE_ANON_KEY
    )
```

---

## 3. Query Parser Service (Updated)

**File**: `backend/app/services/query_parser.py`

```python
"""
Query Parser Service - v2.0
- Uses OpenAI tools API (not deprecated functions)
- Includes reg_type extraction
- Includes handover_months
"""

import json
from typing import Optional
from pydantic import BaseModel
from openai import AsyncOpenAI
from app.core.config import settings

# Updated schema with reg_type
PARSE_QUERY_FUNCTION = {
    "name": "parse_property_query",
    "description": "Extract property details from Dubai real estate query",
    "parameters": {
        "type": "object",
        "properties": {
            "developer": {
                "type": "string",
                "description": "Developer name in ENGLISH. Examples: Emaar, Binghatti, Sobha, Damac, Nakheel"
            },
            "area": {
                "type": "string", 
                "description": "Area/community. Expand abbreviations: JVC=Jumeirah Village Circle, BB=Business Bay, DM=Dubai Marina"
            },
            "bedroom": {
                "type": "string",
                "enum": ["Studio", "1BR", "2BR", "3BR", "4BR", "5BR", "6BR+", "Penthouse"],
                "description": "Bedrooms. Convert: '2 bed' -> '2BR', '2 bedroom' -> '2BR', 'studio' -> 'Studio'"
            },
            "price": {
                "type": "number",
                "description": "Purchase price in AED. Convert: '2.2M' = 2200000, '2.2 million' = 2200000"
            },
            "property_type": {
                "type": "string",
                "enum": ["Unit", "Villa"],
                "description": "Unit = apartment/flat, Villa = villa/townhouse. Default to Unit."
            },
            "reg_type": {
                "type": "string",
                "enum": ["OffPlan", "Ready"],
                "description": "OffPlan = under construction/new launch. Ready = completed/secondary/resale."
            },
            "handover_months": {
                "type": "integer",
                "description": "Months until expected handover. Extract from phrases like '2 years to completion'."
            }
        },
        "required": []
    }
}

SYSTEM_PROMPT = """You are a Dubai real estate query parser. Extract property details accurately.

AREA ABBREVIATIONS (expand these):
- JVC = Jumeirah Village Circle
- JVT = Jumeirah Village Triangle  
- JLT = Jumeirah Lake Towers
- BB = Business Bay
- DM = Dubai Marina
- DH = Dubai Hills Estate
- DIFC = Dubai International Financial Centre
- DSO = Dubai Silicon Oasis
- MBR = Mohammed Bin Rashid City

DEVELOPER NAMES (use English):
Emaar, Binghatti, Sobha, Damac, Nakheel, Meraas, Dubai Properties, Azizi, Danube, Ellington, MAG, Select Group

PRICE FORMATS:
- "2.2M" = 2200000
- "2.2 million" = 2200000
- "AED 2,200,000" = 2200000

BEDROOM FORMATS:
- "2 bed" -> "2BR"
- "2 bedroom" -> "2BR"
- "studio" -> "Studio"
- "penthouse" -> "Penthouse"

REG_TYPE:
- "off-plan", "under construction", "launch", "handover in X" -> OffPlan
- "ready", "completed", "secondary", "resale" -> Ready

If information is not provided, omit that field."""


class ParsedQuery(BaseModel):
    """Structured output from query parsing."""
    developer: Optional[str] = None
    area: Optional[str] = None
    bedroom: Optional[str] = None
    price: Optional[float] = None
    property_type: str = "Unit"
    reg_type: Optional[str] = None
    handover_months: Optional[int] = None
    raw_query: str
    parse_confidence: float = 1.0


class QueryParser:
    """
    Parses natural language property queries using OpenAI tools API.
    """
    
    def __init__(self):
        self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        self.model = settings.OPENAI_MODEL
    
    async def parse(self, query: str) -> ParsedQuery:
        """
        Parse user query into structured property data.
        
        Uses OpenAI tools API (not deprecated functions API).
        """
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": query}
                ],
                tools=[{
                    "type": "function",
                    "function": PARSE_QUERY_FUNCTION
                }],
                tool_choice={
                    "type": "function",
                    "function": {"name": "parse_property_query"}
                },
                temperature=0  # Deterministic
            )
            
            # Extract from tools response
            tool_call = response.choices[0].message.tool_calls[0]
            args = json.loads(tool_call.function.arguments)
            
            return ParsedQuery(
                developer=args.get("developer"),
                area=args.get("area"),
                bedroom=args.get("bedroom"),
                price=args.get("price"),
                property_type=args.get("property_type", "Unit"),
                reg_type=args.get("reg_type"),
                handover_months=args.get("handover_months"),
                raw_query=query,
                parse_confidence=1.0
            )
            
        except Exception as e:
            return ParsedQuery(raw_query=query, parse_confidence=0.0)
    
    def infer_reg_type(self, query: str, explicit: Optional[str] = None) -> str:
        """
        Infer OffPlan vs Ready if not explicitly parsed.
        """
        if explicit:
            return explicit
        
        query_lower = query.lower()
        
        offplan_keywords = [
            "off-plan", "offplan", "off plan",
            "under construction", "launching", "launch",
            "handover", "completion date", "payment plan",
            "post-handover", "phase", "under development"
        ]
        
        ready_keywords = [
            "ready", "completed", "move-in", "secondary",
            "resale", "immediate", "existing"
        ]
        
        for kw in offplan_keywords:
            if kw in query_lower:
                return "OffPlan"
        
        for kw in ready_keywords:
            if kw in query_lower:
                return "Ready"
        
        # Default to OffPlan (tool's primary use case)
        return "OffPlan"
```

---

## 4. Entity Validator (Arabic Support)

**File**: `backend/app/services/entity_validator.py`

```python
"""
Entity Validator Service - v2.0
- Arabic ↔ English developer name resolution
- Area abbreviation expansion
- Fuzzy matching with confidence scores
"""

import json
import os
from typing import Optional, Tuple, Dict, List
from rapidfuzz import fuzz, process
from functools import lru_cache
from app.core.config import settings

class EntityValidator:
    """
    Validates and resolves entities:
    - English developer names -> Arabic (for TFT group_id)
    - Area abbreviations -> Full names
    """
    
    def __init__(self):
        self.developer_mapping = self._load_developer_mapping()
        self.area_mapping = self._load_area_mapping()
        self.english_to_arabic = self._build_english_to_arabic()
        self.all_areas = self._load_all_areas()
    
    def _load_developer_mapping(self) -> List[Dict]:
        """Load Arabic-English developer mapping."""
        path = os.path.join(settings.LOOKUPS_PATH, "developer_mapping.json")
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get("mappings", [])
        return []
    
    def _load_area_mapping(self) -> Dict[str, str]:
        """Load area abbreviations."""
        path = os.path.join(settings.LOOKUPS_PATH, "area_mapping.json")
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get("abbreviations", {})
        return {
            "JVC": "Jumeirah Village Circle",
            "JVT": "Jumeirah Village Triangle",
            "JLT": "Jumeirah Lake Towers",
            "BB": "Business Bay",
            "DM": "Dubai Marina",
            "DH": "Dubai Hills Estate",
            "DIFC": "Dubai International Financial Centre",
            "DSO": "Dubai Silicon Oasis",
            "MBR": "Mohammed Bin Rashid City",
        }
    
    def _build_english_to_arabic(self) -> Dict[str, str]:
        """Build reverse lookup: English/alias -> Arabic."""
        lookup = {}
        for m in self.developer_mapping:
            arabic = m.get("arabic", "")
            english = m.get("english", "")
            aliases = m.get("aliases", [])
            
            if arabic and english:
                lookup[english.lower()] = arabic
                for alias in aliases:
                    lookup[alias.lower()] = arabic
        
        return lookup
    
    def _load_all_areas(self) -> List[str]:
        """Load all valid area names from TFT data or area_mapping."""
        path = os.path.join(settings.LOOKUPS_PATH, "area_mapping.json")
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get("all_areas", [])
        return []
    
    def resolve_developer(self, english_name: Optional[str]) -> Tuple[Optional[str], Optional[str], float]:
        """
        Resolve English developer name to Arabic.
        
        Args:
            english_name: Developer name in English (e.g., "Binghatti")
            
        Returns:
            Tuple of (arabic_name, matched_english, confidence)
        """
        if not english_name:
            return None, None, 0.0
        
        name_lower = english_name.lower().strip()
        
        # 1. Direct match
        if name_lower in self.english_to_arabic:
            arabic = self.english_to_arabic[name_lower]
            return arabic, english_name, 1.0
        
        # 2. Fuzzy match on English names
        all_english = list(self.english_to_arabic.keys())
        if all_english:
            match = process.extractOne(
                name_lower, 
                all_english, 
                scorer=fuzz.token_sort_ratio,
                score_cutoff=70
            )
            if match:
                matched_english = match[0]
                arabic = self.english_to_arabic[matched_english]
                return arabic, matched_english.title(), match[1] / 100
        
        # 3. Fallback to Unknown
        return "Unknown", english_name, 0.3
    
    def validate_area(self, name: Optional[str]) -> Tuple[Optional[str], float]:
        """
        Validate and expand area name.
        
        Args:
            name: Area name (may be abbreviation like "JVC")
            
        Returns:
            Tuple of (validated_name, confidence)
        """
        if not name:
            return None, 0.0
        
        name_upper = name.upper().strip()
        name_lower = name.lower().strip()
        
        # 1. Check abbreviations
        if name_upper in self.area_mapping:
            return self.area_mapping[name_upper], 1.0
        
        # 2. Direct match on full names
        if self.all_areas:
            for area in self.all_areas:
                if area.lower() == name_lower:
                    return area, 1.0
        
        # 3. Fuzzy match
        if self.all_areas:
            match = process.extractOne(
                name,
                self.all_areas,
                scorer=fuzz.token_sort_ratio,
                score_cutoff=70
            )
            if match:
                return match[0], match[1] / 100
        
        # 4. Return original
        return name, 0.5
    
    def validate_bedroom(self, bedroom: Optional[str]) -> Optional[str]:
        """Normalize bedroom format."""
        if not bedroom:
            return None
        
        valid = ["Studio", "1BR", "2BR", "3BR", "4BR", "5BR", "6BR+", "Penthouse", "Room"]
        
        if bedroom in valid:
            return bedroom
        
        bedroom_upper = bedroom.upper().strip()
        mapping = {
            "STUDIO": "Studio",
            "1BR": "1BR", "1 BR": "1BR", "1BED": "1BR", "1 BED": "1BR", "1 BEDROOM": "1BR",
            "2BR": "2BR", "2 BR": "2BR", "2BED": "2BR", "2 BED": "2BR", "2 BEDROOM": "2BR",
            "3BR": "3BR", "3 BR": "3BR", "3BED": "3BR", "3 BED": "3BR", "3 BEDROOM": "3BR",
            "4BR": "4BR", "4 BR": "4BR", "4BED": "4BR", "4 BED": "4BR", "4 BEDROOM": "4BR",
            "5BR": "5BR", "5 BR": "5BR", "5BED": "5BR", "5 BED": "5BR", "5 BEDROOM": "5BR",
            "6BR": "6BR+", "6 BR": "6BR+", "6BED": "6BR+", "6 BED": "6BR+",
            "PENTHOUSE": "Penthouse",
        }
        
        return mapping.get(bedroom_upper, bedroom)


# Singleton
@lru_cache()
def get_validator() -> EntityValidator:
    return EntityValidator()
```

---

## 5. TFT Inference Service (NEW - Critical)

**File**: `backend/app/services/tft_inference.py`

```python
"""
TFT Model Inference Service - v2.0

Handles:
1. Loading trained TFT checkpoint
2. Constructing group_id (with Arabic developer names)
3. Building encoder input from historical data
4. Building decoder input with future known values
5. Running prediction and extracting quantiles
6. Fallback for unknown groups
"""

import pandas as pd
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from functools import lru_cache
from rapidfuzz import process, fuzz
import logging

from pytorch_forecasting import TemporalFusionTransformer
from app.core.config import settings

logger = logging.getLogger(__name__)


class TFTInference:
    """
    TFT Model inference for Dubai property predictions.
    
    Predicts both median_price and median_rent.
    """
    
    # Quantile indices from QuantileLoss [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]
    QUANTILE_NAMES = ['p02', 'p10', 'p25', 'p50', 'p75', 'p90', 'p98']
    
    def __init__(self):
        """Initialize with model and historical data."""
        self.model_path = Path(settings.TFT_MODEL_PATH)
        self.data_path = Path(settings.TFT_DATA_PATH)
        
        # Load model
        self.model = self._load_model()
        
        # Load training data (for encoder history)
        self.data = self._load_data()
        
        # Track current time boundaries
        self.current_time_idx = int(self.data['time_idx'].max())
        self.current_year_month = self.data['year_month'].max()
        
        # Get all unique groups
        self.all_groups = set(self.data['group_id'].unique())
        
        logger.info(f"TFT initialized: {len(self.all_groups)} groups, time_idx up to {self.current_time_idx}")
    
    def _load_model(self) -> TemporalFusionTransformer:
        """Load trained TFT checkpoint."""
        logger.info(f"Loading TFT model from {self.model_path}")
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"TFT model not found: {self.model_path}")
        
        model = TemporalFusionTransformer.load_from_checkpoint(str(self.model_path))
        model.eval()
        model.freeze()
        
        return model
    
    def _load_data(self) -> pd.DataFrame:
        """Load training data for encoder history lookup."""
        logger.info(f"Loading historical data from {self.data_path}")
        
        if not self.data_path.exists():
            raise FileNotFoundError(f"TFT data not found: {self.data_path}")
        
        data = pd.read_csv(self.data_path)
        
        # Ensure correct dtypes (must match training)
        categorical_cols = ['area_name', 'property_type', 'bedroom', 'reg_type', 
                           'developer_name', 'group_id', 'month', 'quarter']
        for col in categorical_cols:
            if col in data.columns:
                data[col] = data[col].astype(str)
        
        return data
    
    def construct_group_id(
        self, 
        area: str, 
        property_type: str, 
        bedroom: str, 
        reg_type: str, 
        developer_arabic: str
    ) -> str:
        """
        Construct group_id in exact format used in training data.
        
        Format: {area}_{property_type}_{bedroom}_{reg_type}_{developer}
        Example: Business_Bay_Unit_2BR_OffPlan_اعمار_العقارية_ش__م__ع
        """
        # Replace spaces with underscores (matching training data)
        area_clean = area.replace(" ", "_") if area else "Unknown"
        dev_clean = developer_arabic.replace(" ", "_") if developer_arabic else "Unknown"
        
        return f"{area_clean}_{property_type or 'Unit'}_{bedroom or '2BR'}_{reg_type or 'OffPlan'}_{dev_clean}"
    
    def find_matching_group(self, group_id: str) -> Tuple[Optional[str], str]:
        """
        Find matching group in training data.
        
        Returns:
            Tuple of (matched_group_id, match_type)
            match_type: "exact", "fuzzy", "fallback", or "none"
        """
        # 1. Exact match
        if group_id in self.all_groups:
            return group_id, "exact"
        
        # 2. Fuzzy match
        match = process.extractOne(
            group_id, 
            list(self.all_groups), 
            scorer=fuzz.token_sort_ratio,
            score_cutoff=70
        )
        
        if match:
            logger.info(f"Fuzzy match: {group_id} -> {match[0]} ({match[1]}%)")
            return match[0], "fuzzy"
        
        # 3. Try fallback strategies
        fallback = self._find_fallback_group(group_id)
        if fallback:
            return fallback, "fallback"
        
        return None, "none"
    
    def _find_fallback_group(self, group_id: str) -> Optional[str]:
        """
        Find fallback group when exact/fuzzy match fails.
        
        Strategy:
        1. Same developer, any area with same bedroom/reg_type
        2. Same area, any developer with same bedroom/reg_type
        """
        parts = group_id.split("_")
        if len(parts) < 5:
            return None
        
        # Parse components (area may have underscores)
        reg_type = parts[-2] if len(parts) >= 2 else "OffPlan"
        bedroom = parts[-3] if len(parts) >= 3 else "2BR"
        property_type = parts[-4] if len(parts) >= 4 else "Unit"
        developer = parts[-1] if len(parts) >= 1 else "Unknown"
        
        # Find any group with same developer
        for g in self.all_groups:
            if g.endswith(f"_{developer}") and f"_{bedroom}_{reg_type}_" in g:
                logger.info(f"Fallback (same developer): {g}")
                return g
        
        return None
    
    def get_group_history(self, group_id: str, max_length: int = 96) -> pd.DataFrame:
        """
        Get historical data for a group (encoder input).
        
        Args:
            group_id: Group identifier
            max_length: Maximum encoder length
            
        Returns:
            DataFrame with historical data, sorted by time_idx
        """
        group_data = self.data[self.data['group_id'] == group_id].copy()
        group_data = group_data.sort_values('time_idx')
        
        # Take last N months (up to max_length)
        return group_data.tail(max_length)
    
    def construct_future_known(
        self, 
        group_history: pd.DataFrame, 
        forecast_months: int
    ) -> pd.DataFrame:
        """
        Construct future time-varying known values.
        
        TFT needs: month, quarter, month_sin, month_cos, units_completing
        """
        if group_history.empty:
            raise ValueError("No history to base future on")
        
        last_row = group_history.iloc[-1]
        last_time_idx = int(last_row['time_idx'])
        
        # Parse year_month
        ym_parts = str(last_row['year_month']).split('-')
        year = int(ym_parts[0])
        month = int(ym_parts[1])
        
        future_rows = []
        for i in range(1, forecast_months + 1):
            # Advance month
            month += 1
            if month > 12:
                month = 1
                year += 1
            
            row = {
                # Time indices
                'time_idx': last_time_idx + i,
                'year_month': f"{year:04d}-{month:02d}",
                
                # Time-varying KNOWN categoricals
                'month': str(month),
                'quarter': str((month - 1) // 3 + 1),
                
                # Time-varying KNOWN reals
                'month_sin': np.sin(2 * np.pi * month / 12),
                'month_cos': np.cos(2 * np.pi * month / 12),
                'units_completing': 0,  # TODO: Lookup from supply schedule
                
                # Static categoricals (copy from last row)
                'group_id': last_row['group_id'],
                'area_name': last_row['area_name'],
                'property_type': last_row['property_type'],
                'bedroom': last_row['bedroom'],
                'reg_type': last_row['reg_type'],
                'developer_name': last_row['developer_name'],
            }
            
            # Fill unknown reals with last known value
            # (TFT will predict these, but needs placeholder)
            unknown_cols = [
                'median_price', 'transaction_count', 'median_rent', 'rent_count',
                'median_rent_sqft', 'months_since_launch', 'months_to_handover',
                'project_percent_complete', 'project_duration_months', 'phase_ratio',
                'supply_units', 'supply_buildings', 'supply_villas', 'active_projects',
                'units_registered', 'buildings_registered', 'avg_building_floors',
                'avg_building_flats', 'dev_total_projects', 'dev_completed_projects',
                'dev_total_units', 'dev_avg_completion', 'dev_overall_median_price',
                'dev_overall_transactions', 'market_median_price', 'market_transactions',
                'govt_valuation_median', 'valuation_count', 'eibor_overnight',
                'eibor_1w', 'eibor_1m', 'eibor_3m', 'eibor_6m', 'eibor_12m',
                'visitors_total', 'hotel_rooms', 'hotel_apartments', 'has_actual_rent'
            ]
            
            for col in unknown_cols:
                if col in last_row.index:
                    row[col] = last_row.get(col, 0) if pd.notna(last_row.get(col)) else 0
                else:
                    row[col] = 0
            
            future_rows.append(row)
        
        return pd.DataFrame(future_rows)
    
    def predict(
        self,
        area: str,
        property_type: str,
        bedroom: str,
        reg_type: str,
        developer_arabic: str,
        forecast_months: int = 12,
        purchase_price: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Run TFT prediction for a property combination.
        
        Args:
            area: Area name (English)
            property_type: "Unit" or "Villa"
            bedroom: "Studio", "1BR", "2BR", etc.
            reg_type: "OffPlan" or "Ready"
            developer_arabic: Developer name in Arabic
            forecast_months: Number of months to forecast
            purchase_price: User's purchase price (optional, for appreciation calc)
            
        Returns:
            Prediction results with quantiles for price and rent
        """
        # 1. Construct group_id
        group_id = self.construct_group_id(
            area, property_type, bedroom, reg_type, developer_arabic
        )
        logger.info(f"Predicting for group: {group_id}")
        
        # 2. Find matching group
        matched_group, match_type = self.find_matching_group(group_id)
        
        if matched_group is None:
            logger.warning(f"No matching group found for {group_id}")
            return self._no_match_response(group_id, area, developer_arabic)
        
        # 3. Get historical data
        history = self.get_group_history(matched_group)
        
        if len(history) < 3:
            logger.warning(f"Insufficient history for {matched_group}: {len(history)} months")
            return self._insufficient_history_response(matched_group, len(history))
        
        # 4. Get current values for comparison
        current_price = float(history.iloc[-1]['median_price']) if pd.notna(history.iloc[-1]['median_price']) else 0
        current_rent = float(history.iloc[-1]['median_rent']) if pd.notna(history.iloc[-1]['median_rent']) else 0
        
        # 5. Construct prediction input
        future = self.construct_future_known(history, forecast_months)
        prediction_data = pd.concat([history, future], ignore_index=True)
        
        # 6. Run prediction
        try:
            with torch.no_grad():
                raw_predictions = self.model.predict(
                    prediction_data,
                    mode="quantiles",
                    return_x=False
                )
        except Exception as e:
            logger.error(f"TFT prediction failed: {e}")
            return {"error": str(e), "group_id": group_id}
        
        # 7. Extract quantiles
        price_quantiles = self._extract_quantiles(raw_predictions, target_idx=0, forecast_months=forecast_months)
        rent_quantiles = self._extract_quantiles(raw_predictions, target_idx=1, forecast_months=forecast_months)
        
        # 8. Calculate appreciation
        final_price_median = price_quantiles['median'][-1]
        appreciation_pct = ((final_price_median - current_price) / current_price * 100) if current_price > 0 else 0
        
        # 9. User-specific calculations
        user_specific = None
        if purchase_price:
            user_appreciation = purchase_price * (1 + appreciation_pct / 100)
            user_specific = {
                "purchase_price": purchase_price,
                "estimated_handover_value": user_appreciation,
                "appreciation_aed": user_appreciation - purchase_price,
                "appreciation_percent": appreciation_pct
            }
        
        return {
            "group_id": group_id,
            "matched_group": matched_group,
            "match_type": match_type,
            "history_months": len(history),
            "forecast_months": forecast_months,
            
            "price_forecast": {
                "current_sqft": current_price,
                "final_sqft_median": final_price_median,
                "final_sqft_low": price_quantiles['p10'][-1],
                "final_sqft_high": price_quantiles['p90'][-1],
                "appreciation_percent": appreciation_pct,
                "monthly": price_quantiles
            },
            
            "rent_forecast": {
                "current_annual": current_rent,
                "final_annual_median": rent_quantiles['median'][-1],
                "final_annual_low": rent_quantiles['p10'][-1],
                "final_annual_high": rent_quantiles['p90'][-1],
                "monthly": rent_quantiles
            },
            
            "user_specific": user_specific
        }
    
    def _extract_quantiles(
        self, 
        predictions: torch.Tensor, 
        target_idx: int,
        forecast_months: int
    ) -> Dict[str, List[float]]:
        """
        Extract quantiles for a specific target from TFT output.
        
        For multi-target with 7 quantiles each:
        - Shape: [batch, horizon, 14]
        - Target 0 (price): indices 0-6
        - Target 1 (rent): indices 7-13
        """
        num_quantiles = 7
        start_idx = target_idx * num_quantiles
        end_idx = start_idx + num_quantiles
        
        # Get last forecast_months predictions
        preds = predictions[0, -forecast_months:, start_idx:end_idx].cpu().numpy()
        
        return {
            'p02': preds[:, 0].tolist(),
            'p10': preds[:, 1].tolist(),
            'p25': preds[:, 2].tolist(),
            'median': preds[:, 3].tolist(),  # p50
            'p75': preds[:, 4].tolist(),
            'p90': preds[:, 5].tolist(),
            'p98': preds[:, 6].tolist(),
        }
    
    def _no_match_response(self, group_id: str, area: str, developer: str) -> Dict[str, Any]:
        """Response when no matching group found."""
        return {
            "error": "no_matching_group",
            "group_id": group_id,
            "message": f"No training data for this combination: {area} / {developer}",
            "suggestions": [
                "Try a different area",
                "Try a different developer",
                "Check spelling of area/developer name"
            ]
        }
    
    def _insufficient_history_response(self, group_id: str, months: int) -> Dict[str, Any]:
        """Response when group has insufficient history."""
        return {
            "error": "insufficient_history",
            "group_id": group_id,
            "history_months": months,
            "message": f"Only {months} months of history available. Need at least 3 months."
        }


# Singleton
_tft_instance: Optional[TFTInference] = None

def get_tft() -> TFTInference:
    global _tft_instance
    if _tft_instance is None:
        _tft_instance = TFTInference()
    return _tft_instance
```

---

## 6. Trend Lookup (Supabase)

**File**: `backend/app/services/trend_lookup.py`

```python
"""
Trend Lookup Service - v2.0
Uses Supabase for lookups, with CSV fallback.
"""

import pandas as pd
import os
from typing import Dict, Any, Optional
from functools import lru_cache
from supabase import Client
from app.core.config import settings
from app.core.supabase import get_supabase


class TrendLookup:
    """
    Looks up developer stats, area trends from Supabase or CSV.
    """
    
    def __init__(self, supabase: Optional[Client] = None):
        self.supabase = supabase or get_supabase()
        self.use_supabase = self._check_supabase()
        
        # Fallback to CSV
        if not self.use_supabase:
            self._load_csv_data()
    
    def _check_supabase(self) -> bool:
        """Check if Supabase tables exist."""
        try:
            self.supabase.table('developer_stats').select('*').limit(1).execute()
            return True
        except:
            return False
    
    def _load_csv_data(self):
        """Load CSV fallback data."""
        lookups_path = settings.LOOKUPS_PATH
        
        dev_path = os.path.join(lookups_path, "developer_stats.csv")
        self.dev_df = pd.read_csv(dev_path) if os.path.exists(dev_path) else pd.DataFrame()
        
        area_path = os.path.join(lookups_path, "area_stats.csv")
        self.area_df = pd.read_csv(area_path) if os.path.exists(area_path) else pd.DataFrame()
        
        rent_path = os.path.join(lookups_path, "rent_benchmarks.csv")
        self.rent_df = pd.read_csv(rent_path) if os.path.exists(rent_path) else pd.DataFrame()
    
    def get_developer_stats(self, developer_arabic: str) -> Dict[str, Any]:
        """Get developer statistics."""
        if not developer_arabic or developer_arabic == "Unknown":
            return self._empty_developer()
        
        if self.use_supabase:
            result = self.supabase.table('developer_stats') \
                .select('*') \
                .eq('developer_name', developer_arabic) \
                .single() \
                .execute()
            
            if result.data:
                return result.data
        else:
            # CSV fallback
            if not self.dev_df.empty:
                row = self.dev_df[self.dev_df['developer_name'] == developer_arabic]
                if not row.empty:
                    return row.iloc[0].to_dict()
        
        return self._empty_developer()
    
    def _empty_developer(self) -> Dict[str, Any]:
        return {
            "developer_name": "Unknown",
            "projects_completed": "N/A",
            "projects_active": "N/A",
            "total_units": "N/A",
            "avg_delay_months": "N/A",
            "completion_rate": "N/A"
        }
    
    def get_area_trends(self, area: str, bedroom: str = None) -> Dict[str, Any]:
        """Get area price trends."""
        if not area:
            return self._empty_area()
        
        if self.use_supabase:
            result = self.supabase.table('area_stats') \
                .select('*') \
                .eq('area_name', area) \
                .single() \
                .execute()
            
            if result.data:
                data = result.data
                # Add rent if bedroom specified
                if bedroom:
                    data['median_rent'] = self._get_rent(area, bedroom)
                return data
        else:
            if not self.area_df.empty:
                row = self.area_df[self.area_df['area_name'] == area]
                if not row.empty:
                    data = row.iloc[0].to_dict()
                    if bedroom:
                        data['median_rent'] = self._get_rent(area, bedroom)
                    return data
        
        return self._empty_area()
    
    def _get_rent(self, area: str, bedroom: str) -> float:
        """Get median rent for area/bedroom."""
        if self.use_supabase:
            result = self.supabase.table('rent_benchmarks') \
                .select('median_annual_rent') \
                .eq('area_name', area) \
                .eq('bedrooms', bedroom) \
                .order('year_month', desc=True) \
                .limit(1) \
                .execute()
            
            if result.data:
                return result.data[0].get('median_annual_rent', 0)
        else:
            if not self.rent_df.empty:
                filtered = self.rent_df[
                    (self.rent_df['area_name'] == area) &
                    (self.rent_df['bedrooms'] == bedroom)
                ]
                if not filtered.empty:
                    return float(filtered.iloc[0].get('median_annual_rent', 0))
        
        return 0
    
    def _empty_area(self) -> Dict[str, Any]:
        return {
            "area_name": "Unknown",
            "current_median_sqft": 0,
            "price_change_12m": 0,
            "price_change_36m": 0,
            "supply_pipeline": 0,
            "median_rent": 0
        }
    
    def get_market_data(self) -> Dict[str, Any]:
        """Get current market conditions."""
        # Get latest EIBOR from TFT data
        return {
            "eibor_12m": "N/A",  # TODO: Get from data
            "market_trend": "N/A"
        }


@lru_cache()
def get_trend_lookup() -> TrendLookup:
    return TrendLookup()
```

---

## 7. Chat Session Service (Supabase)

**File**: `backend/app/services/chat_service.py`

```python
"""
Chat Session Service
Manages chat sessions and messages in Supabase.
"""

from typing import Optional, List, Dict, Any
from uuid import uuid4
from datetime import datetime
from supabase import Client
from app.core.supabase import get_supabase


class ChatService:
    """Manages chat sessions and messages."""
    
    def __init__(self, supabase: Optional[Client] = None):
        self.supabase = supabase or get_supabase()
    
    async def create_session(self, user_id: str, title: str = None) -> str:
        """Create new chat session."""
        session_id = str(uuid4())
        
        self.supabase.table('chat_sessions').insert({
            'id': session_id,
            'user_id': user_id,
            'title': title or 'New Chat',
            'created_at': datetime.utcnow().isoformat(),
            'updated_at': datetime.utcnow().isoformat()
        }).execute()
        
        return session_id
    
    async def get_sessions(self, user_id: str) -> List[Dict[str, Any]]:
        """Get all sessions for a user."""
        result = self.supabase.table('chat_sessions') \
            .select('*') \
            .eq('user_id', user_id) \
            .order('updated_at', desc=True) \
            .execute()
        
        return result.data or []
    
    async def save_message(
        self,
        session_id: str,
        role: str,
        content: str,
        parsed_query: Dict = None,
        predictions: Dict = None,
        report_data: Dict = None
    ) -> str:
        """Save a chat message."""
        message_id = str(uuid4())
        
        self.supabase.table('chat_messages').insert({
            'id': message_id,
            'session_id': session_id,
            'role': role,
            'content': content,
            'parsed_query': parsed_query,
            'predictions': predictions,
            'report_data': report_data,
            'created_at': datetime.utcnow().isoformat()
        }).execute()
        
        # Update session timestamp
        self.supabase.table('chat_sessions') \
            .update({'updated_at': datetime.utcnow().isoformat()}) \
            .eq('id', session_id) \
            .execute()
        
        # Update title if first message
        if role == 'user':
            session = self.supabase.table('chat_sessions') \
                .select('title') \
                .eq('id', session_id) \
                .single() \
                .execute()
            
            if session.data and session.data.get('title') == 'New Chat':
                # Use first 50 chars of first message as title
                new_title = content[:50] + ('...' if len(content) > 50 else '')
                self.supabase.table('chat_sessions') \
                    .update({'title': new_title}) \
                    .eq('id', session_id) \
                    .execute()
        
        return message_id
    
    async def get_messages(self, session_id: str) -> List[Dict[str, Any]]:
        """Get all messages for a session."""
        result = self.supabase.table('chat_messages') \
            .select('*') \
            .eq('session_id', session_id) \
            .order('created_at', asc=True) \
            .execute()
        
        return result.data or []


def get_chat_service() -> ChatService:
    return ChatService()
```

---

## 8. Requirements

**File**: `backend/requirements.txt`

```txt
# Web framework
fastapi==0.104.1
uvicorn==0.24.0
pydantic==2.5.2
pydantic-settings==2.1.0
python-multipart==0.0.6

# OpenAI
openai==1.6.0

# Supabase
supabase==2.3.0

# Data processing
pandas==2.1.3
numpy==1.26.2

# Fuzzy matching
rapidfuzz==3.5.2

# ML model
pytorch-forecasting==1.5.0
torch==2.1.0
lightning==2.4.0

# PDF generation
reportlab==4.0.7
```

---

## 9. Environment Template

**File**: `backend/.env.example`

```bash
# OpenAI API
OPENAI_API_KEY=sk-your-key-here
OPENAI_MODEL=gpt-4-turbo-preview

# Supabase
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_ANON_KEY=eyJ...
SUPABASE_SERVICE_KEY=eyJ...

# TFT Model Paths
TFT_MODEL_PATH=models/tft_final.ckpt
TFT_DATA_PATH=Data/tft/tft_training_data.csv

# Lookups
LOOKUPS_PATH=Data/lookups
```
