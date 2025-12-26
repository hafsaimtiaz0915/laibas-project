"""
TFT Inference Service

Handles loading the TFT model and making predictions.
Properly handles multi-target predictions (price + rent) with quantile outputs.

Group ID Format (from build_tft_data.py):
    {area}_{property_type}_{bedroom}_{reg_type}_{developer}
    
    Examples:
    - Al_Barsha_First_Unit_1BR_Ready_ALL_DEVELOPERS
    - Business_Bay_Unit_2BR_OffPlan_بن_غاتي_للتطوير_العقاري
"""

# ============================================================
# PATCH: Fix torchmetrics CUDA error on Mac (must be BEFORE imports)
# Model was trained on GPU but Mac has no CUDA
# ============================================================
import torch
import torchmetrics

_orig_metric_apply = torchmetrics.Metric._apply

def _safe_metric_apply(self, fn, *args, **kwargs):
    """Patch to force CPU device when CUDA is unavailable."""
    if hasattr(self, '_device') and self._device is not None:
        if 'cuda' in str(self._device) and not torch.cuda.is_available():
            self._device = torch.device('cpu')
    return _orig_metric_apply(self, fn, *args, **kwargs)

torchmetrics.Metric._apply = _safe_metric_apply
# ============================================================

import logging
import re
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime
from pydantic import BaseModel
import pandas as pd
import numpy as np

from ..core.config import get_settings

logger = logging.getLogger(__name__)

# Quantile indices from PyTorch Forecasting default QuantileLoss
# Default quantiles: [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]
QUANTILE_10 = 1  # Index for 10th percentile (low)
QUANTILE_50 = 3  # Index for median
QUANTILE_90 = 5  # Index for 90th percentile (high)

# Conversion factor: DLD data is in square meters, display in square feet
# 1 sqm = 10.764 sqft, so AED/sqm ÷ 10.764 = AED/sqft
SQM_TO_SQFT = 10.764


class PriceForecast(BaseModel):
    """Price forecast results (price per sqft)."""
    current_sqft: Optional[float] = None
    forecast_sqft_low: Optional[float] = None  # 10th percentile
    forecast_sqft_median: Optional[float] = None  # 50th percentile
    forecast_sqft_high: Optional[float] = None  # 90th percentile
    forecast_horizon_months: int = 12
    appreciation_percent: Optional[float] = None
    # Longer-horizon projections (derived from model-implied monthly growth; not directly modeled unless trained for it)
    forecast_sqft_low_12m: Optional[float] = None
    forecast_sqft_median_12m: Optional[float] = None
    forecast_sqft_high_12m: Optional[float] = None
    appreciation_percent_12m: Optional[float] = None
    forecast_sqft_low_24m: Optional[float] = None
    forecast_sqft_median_24m: Optional[float] = None
    forecast_sqft_high_24m: Optional[float] = None
    appreciation_percent_24m: Optional[float] = None
    long_horizon_method: Optional[str] = None
    # Total value calculations (if unit_sqft provided)
    unit_sqft: Optional[float] = None
    current_total_value: Optional[float] = None
    forecast_total_value_low: Optional[float] = None
    forecast_total_value_median: Optional[float] = None
    forecast_total_value_high: Optional[float] = None
    # Total value for 12m horizon (for off-plan: handover + 12m)
    forecast_total_value_low_12m: Optional[float] = None
    forecast_total_value_median_12m: Optional[float] = None
    forecast_total_value_high_12m: Optional[float] = None


class RentForecast(BaseModel):
    """Rent forecast results (annual rent)."""
    current_annual: Optional[float] = None
    forecast_annual_low: Optional[float] = None
    forecast_annual_median: Optional[float] = None
    forecast_annual_high: Optional[float] = None
    has_actual_rent: bool = True
    estimated_yield_percent: Optional[float] = None


class ModelDriver(BaseModel):
    """Model attribution driver (non-causal): feature importance weight from TFT interpretability outputs."""
    feature: str
    importance: float
    target: str  # e.g. "price" or "rent"
    current_value: Optional[float] = None
    change_6m: Optional[float] = None


class ModelAttribution(BaseModel):
    """Model-driven attribution summary for this forecast (non-causal)."""
    top_drivers: List[ModelDriver] = []
    # Local sensitivity ("what-if") impacts computed by perturbing selected future inputs and rerunning the model.
    # This is NOT causal; it is a ceteris-paribus model response.
    what_if_impacts: List[Dict[str, Any]] = []
    method: str = "tft_interpret_output"


class TrendInsights(BaseModel):
    """Trend insights computed from historical data."""
    # Price momentum
    price_change_3m: Optional[float] = None
    price_change_6m: Optional[float] = None
    price_change_12m: Optional[float] = None
    price_trend_direction: Optional[str] = None  # accelerating, stable, decelerating
    
    # Rent trends
    rent_change_12m: Optional[float] = None
    
    # Transaction volume
    transaction_volume_trend: Optional[float] = None
    transaction_volume_recent_avg: Optional[float] = None
    transaction_volume_year_ago_avg: Optional[float] = None
    area_transaction_volume_trend: Optional[float] = None
    area_transaction_volume_recent_avg: Optional[float] = None
    area_transaction_volume_year_ago_avg: Optional[float] = None
    
    # Supply dynamics
    supply_pipeline: Optional[int] = None
    units_completing_6m: Optional[int] = None
    active_projects: Optional[int] = None
    
    # Developer track record (from training data)
    developer_projects: Optional[int] = None
    developer_completed: Optional[int] = None
    developer_units: Optional[int] = None
    developer_avg_completion: Optional[float] = None
    
    # Market context
    current_eibor_3m: Optional[float] = None
    eibor_change_6m: Optional[float] = None
    
    # Tourism
    tourism_visitors: Optional[int] = None
    tourism_change_12m: Optional[float] = None
    
    # Data quality
    data_as_of: Optional[str] = None
    months_of_history: Optional[int] = None


class TFTPrediction(BaseModel):
    """Complete TFT prediction result."""
    price_forecast: PriceForecast
    rent_forecast: RentForecast
    trend_insights: Optional[TrendInsights] = None
    model_attribution: Optional[ModelAttribution] = None
    match_type: str  # "exact", "partial_area_bedroom", "area_only", "fallback", "mock"
    matched_group_id: Optional[str] = None
    confidence: float = 0.0


def parse_group_id(group_id: str) -> Dict[str, str]:
    """
    Parse a group_id into its components.
    
    Format: {area}_{property_type}_{bedroom}_{reg_type}_{developer}
    
    Challenge: Area names can contain underscores (e.g., "Al_Barsha_First")
    
    Strategy: Parse from the END since property_type, bedroom, reg_type have known formats.
    """
    # Developer names ALWAYS contain underscores (spaces -> underscores) so rsplit-based parsing
    # is not reliable. Instead, split on reg_type token first, then parse the left side.
    result = {"area": None, "property_type": None, "bedroom": None, "reg_type": None, "developer": None}

    reg_token = None
    if "_OffPlan_" in group_id:
        reg_token = "_OffPlan_"
        result["reg_type"] = "OffPlan"
    elif "_Ready_" in group_id:
        reg_token = "_Ready_"
        result["reg_type"] = "Ready"

    if not reg_token:
        return result

    left, developer_part = group_id.split(reg_token, 1)
    result["developer"] = developer_part.replace("_", " ")

    # left format: {area}_{property_type}_{bedroom}
    bedroom_pattern = r'_(Studio|1BR|2BR|3BR|4BR|5BR|6BR\+|Penthouse|Room)$'
    bedroom_match = re.search(bedroom_pattern, left)
    if not bedroom_match:
        return result

    result["bedroom"] = bedroom_match.group(1)
    left_prefix = left[: bedroom_match.start()]  # {area}_{property_type}

    if left_prefix.endswith("_Unit"):
        result["property_type"] = "Unit"
        result["area"] = left_prefix[: -len("_Unit")].replace("_", " ")
    elif left_prefix.endswith("_Villa"):
        result["property_type"] = "Villa"
        result["area"] = left_prefix[: -len("_Villa")].replace("_", " ")
    else:
        # Unknown property type; best effort
        result["area"] = left_prefix.replace("_", " ")

    return result


def construct_group_id(
    area: str,
    property_type: str,
    bedroom: str,
    reg_type: str,
    developer: str
) -> str:
    """
    Construct a group_id matching the format from build_tft_data.py.
    """
    # Apply same transformations as build_tft_data.py
    area_clean = area.replace(' ', '_').replace("'", "")
    prop_clean = property_type.replace(' ', '_')
    bed_clean = bedroom.replace(' ', '_')
    dev_clean = developer.replace(' ', '_').replace("'", "").replace(",", "").replace(".", "")
    
    return f"{area_clean}_{prop_clean}_{bed_clean}_{reg_type}_{dev_clean}"


class TFTInferenceService:
    """
    TFT Model Inference Service.
    
    Loads the trained multi-target TFT model and performs inference.
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.model = None
        self.training_dataset = None  # Need this for inference
        self.data: Optional[pd.DataFrame] = None
        self.groups: List[str] = []
        self.is_loaded = False
        
        # Group ID component indices for fast lookup
        self._group_components: Dict[str, Dict[str, str]] = {}
        
        # Initialize
        self._initialize()
    
    def _initialize(self):
        """Initialize model and data."""
        # Load full TFT training data - this is the source of truth for group matching
        # Using the same CSV as local development ensures identical matching behavior
        data_path = Path(self.settings.tft_data_path)
        if data_path.exists():
            try:
                self.data = pd.read_csv(data_path, low_memory=False)
                self.groups = self.data['group_id'].unique().tolist()
                
                # Pre-parse all group IDs for fast fuzzy matching
                for gid in self.groups:
                    self._group_components[gid] = parse_group_id(gid)
                
                logger.info(f"Loaded TFT data: {len(self.data):,} rows, {len(self.groups):,} groups")
            except Exception as e:
                logger.error(f"Error loading TFT data: {e}")
        else:
            logger.warning(f"TFT data not found: {data_path}")
        
        # Load model (when available)
        model_path = Path(self.settings.tft_model_path)
        if model_path.exists():
            try:
                # Import here to avoid loading PyTorch if not needed
                from pytorch_forecasting import TemporalFusionTransformer
                
                try:
                    self.model = TemporalFusionTransformer.load_from_checkpoint(
                        str(model_path),
                        map_location='cpu'  # Force CPU loading (Mac has no CUDA)
                    )
                except Exception as e:
                    # Some checkpoints can contain stale/typo hyperparameters that break loading
                    # across pytorch-forecasting/lightning versions (e.g. "monotone_constaints").
                    # Best-effort patch: remove the offending keys and retry from a patched checkpoint.
                    msg = str(e)
                    if "monotone_constaints" in msg:
                        import torch
                        ckpt = torch.load(str(model_path), map_location="cpu")
                        hp = ckpt.get("hyper_parameters") or {}
                        if isinstance(hp, dict):
                            hp.pop("monotone_constaints", None)
                            dp = hp.get("dataset_parameters")
                            if isinstance(dp, dict):
                                dp.pop("monotone_constaints", None)
                                hp["dataset_parameters"] = dp
                            ckpt["hyper_parameters"] = hp

                        patched_path = model_path.with_name(f"{model_path.stem}__patched.ckpt")
                        try:
                            torch.save(ckpt, str(patched_path))
                            self.model = TemporalFusionTransformer.load_from_checkpoint(
                                str(patched_path),
                                map_location="cpu",
                            )
                            logger.warning(
                                f"Loaded TFT model from patched checkpoint due to stale hyperparameters: {patched_path}"
                            )
                        except Exception:
                            raise e
                    else:
                        raise
                self.model.eval()  # Set to evaluation mode
                self.is_loaded = True
                
                logger.info(f"TFT model loaded successfully: {model_path}")
            except ImportError:
                logger.warning("pytorch_forecasting not installed - model inference disabled")
            except Exception as e:
                logger.error(f"Error loading TFT model: {e}")
                self.is_loaded = False
        else:
            logger.info(f"TFT model not yet available: {model_path}")
    
    def find_matching_group(
        self,
        area: str,
        property_type: str,
        bedroom: str,
        reg_type: str,
        developer: str
    ) -> Tuple[Optional[str], str, float]:
        """
        Find best matching group in training data.
        
        Returns:
            Tuple of (matched_group_id, match_type, confidence)
        """
        if not self.groups:
            return None, "no_data", 0.0
        
        # Build target group_id
        target_gid = construct_group_id(area, property_type, bedroom, reg_type, developer)
        
        # 1. Try exact match
        if target_gid in self.groups:
            return target_gid, "exact", 95.0
        
        # 2. Try matching with the explicit market bucket developer
        # (the ETL should not produce "Unknown", but we still want a deterministic fallback series)
        target_all = construct_group_id(area, property_type, bedroom, reg_type, "ALL_DEVELOPERS")
        if target_all in self.groups:
            return target_all, "exact_all_developers", 85.0
        
        # 3. Segment match within SAME AREA:
        #    same area + bedroom + property_type (+ reg_type if possible; developer optional)
        area_clean = area.replace(' ', '_').replace("'", "").lower()
        
        matches = []
        for gid, components in self._group_components.items():
            if not components.get("area"):
                continue
                
            comp_area = components["area"].replace(' ', '_').replace("'", "").lower()
            comp_bedroom = components.get("bedroom", "").lower()
            comp_prop = components.get("property_type", "").lower()

            # Hard gate: only consider groups in the SAME area.
            # Without this, we can accidentally match on bedroom/prop/reg_type alone and pick a random area.
            if comp_area != area_clean:
                continue
            
            # Score based on matching components
            score = 0
            score += 40  # area matches (hard-gated above)
            if comp_bedroom == bedroom.lower():
                score += 25
            if comp_prop == property_type.lower():
                score += 15
            if components.get("reg_type") == reg_type:
                score += 10
            if components.get("developer", "").lower() == developer.lower():
                score += 10
            
            matches.append((gid, score))
        
        if matches:
            # Sort by score descending
            matches.sort(key=lambda x: x[1], reverse=True)
            best_gid, best_score = matches[0]
            
            if best_score >= 80:
                match_type = "partial_area_bedroom"
                confidence = 75.0
            elif best_score >= 55:
                match_type = "partial_area"
                confidence = 60.0
            else:
                match_type = "area_only"
                confidence = 50.0
            
            return best_gid, match_type, confidence
        
        # 4. Fallback: any group with similar bedroom
        for gid, components in self._group_components.items():
            if components.get("bedroom", "").lower() == bedroom.lower():
                return gid, "fallback_bedroom", 30.0
        
        return None, "no_match", 0.0
    
    def get_group_history(self, group_id: str, max_encoder_length: int = 96) -> pd.DataFrame:
        """Get most recent history for a group for encoder input."""
        if self.data is None or group_id not in self.groups:
            return pd.DataFrame()
        
        group_data = self.data[self.data['group_id'] == group_id].copy()
        group_data = group_data.sort_values('time_idx')
        
        # Get last N rows for encoder
        if len(group_data) > max_encoder_length:
            group_data = group_data.tail(max_encoder_length)
        
        return group_data
    
    def get_current_values(self, group_id: str) -> Dict[str, Any]:
        """Get most recent values for a group."""
        if self.data is None or group_id not in self.groups:
            return {}
        
        group_data = self.data[self.data['group_id'] == group_id].sort_values('time_idx')
        
        if group_data.empty:
            return {}
        
        latest = group_data.iloc[-1]
        
        return {
            "median_price": latest.get("median_price"),
            "median_rent": latest.get("median_rent"),
            "has_actual_rent": (latest.get("median_rent") or 0) > 0,
            "transaction_count": latest.get("transaction_count"),
            "rent_count": latest.get("rent_count"),
            "time_idx": latest.get("time_idx"),
            "year_month": latest.get("year_month"),
        }
    
    def compute_trend_insights(self, group_id: str) -> Dict[str, Any]:
        """
        Compute rich trend insights from historical data.
        
        Returns insights on:
        - Price momentum (3m, 6m, 12m trends)
        - Rent trends
        - Transaction volume trends
        - Supply dynamics
        - Market context (EIBOR, tourism)
        - Developer track record
        """
        history = self.get_group_history(group_id)
        
        if len(history) < 6:
            return {}
        
        insights = {}
        
        # Get latest and historical values
        latest = history.iloc[-1]

        # If the latest point is in the current calendar month, treat it as potentially incomplete.
        # For trend calculations, prefer excluding the current in-progress month to avoid artificial drops/spikes.
        hist_for_trends = history
        try:
            now_ym = datetime.utcnow().strftime("%Y-%m")
            latest_ym = str(latest.get("year_month") or "")
            if latest_ym == now_ym and len(history) >= 4:
                hist_for_trends = history.iloc[:-1]
        except Exception:
            hist_for_trends = history
        
        # === Price Trend Analysis ===
        price_col = 'median_price'
        if price_col in hist_for_trends.columns:
            prices = hist_for_trends[price_col].dropna()
            if len(prices) >= 13:
                # Calculate momentum at different horizons
                current_price = prices.iloc[-1]
                # Note: series is monthly; "3 months ago" means ~3 months prior to the latest point.
                # Using -4 (not -3) because -2 is 1 month ago, -3 is 2 months ago, -4 is 3 months ago.
                price_3m_ago = prices.iloc[-4] if len(prices) >= 4 else current_price
                price_6m_ago = prices.iloc[-7] if len(prices) >= 7 else current_price
                price_12m_ago = prices.iloc[-13] if len(prices) >= 13 else current_price
                
                insights['price_change_3m'] = ((current_price / price_3m_ago) - 1) * 100 if price_3m_ago > 0 else 0
                insights['price_change_6m'] = ((current_price / price_6m_ago) - 1) * 100 if price_6m_ago > 0 else 0
                insights['price_change_12m'] = ((current_price / price_12m_ago) - 1) * 100 if price_12m_ago > 0 else 0
                
                # Trend direction (accelerating, stable, decelerating)
                recent_momentum = insights['price_change_3m']
                longer_momentum = insights['price_change_6m'] / 2  # Annualized for comparison
                
                if recent_momentum > longer_momentum + 2:
                    insights['price_trend_direction'] = 'accelerating'
                elif recent_momentum < longer_momentum - 2:
                    insights['price_trend_direction'] = 'decelerating'
                else:
                    insights['price_trend_direction'] = 'stable'
        
        # === Rent Trend Analysis ===
        rent_col = 'median_rent'
        if rent_col in history.columns:
            rents = history[rent_col].dropna()
            if len(rents) >= 12:
                current_rent = rents.iloc[-1]
                rent_12m_ago = rents.iloc[-12] if len(rents) >= 12 else current_rent
                insights['rent_change_12m'] = ((current_rent / rent_12m_ago) - 1) * 100 if rent_12m_ago > 0 else 0
        
        # === Transaction Volume Trends ===
        if 'transaction_count' in hist_for_trends.columns:
            txn = hist_for_trends['transaction_count'].dropna()
            if len(txn) >= 15:
                recent_avg = float(txn.tail(3).mean())
                year_ago_avg = float(txn.iloc[-15:-12].mean())
                insights['transaction_volume_recent_avg'] = recent_avg
                insights['transaction_volume_year_ago_avg'] = year_ago_avg
                insights['transaction_volume_trend'] = ((recent_avg / year_ago_avg) - 1) * 100 if year_ago_avg > 0 else 0

        # Area-wide transaction volume trend (sum across all segments in the same DLD area)
        try:
            area_name = str(latest.get("area_name") or "")
            if self.data is not None and area_name and "transaction_count" in self.data.columns and "year_month" in self.data.columns:
                area_df = self.data[self.data["area_name"] == area_name].copy()
                # Exclude current in-progress month if present
                now_ym = datetime.utcnow().strftime("%Y-%m")
                area_df["year_month"] = area_df["year_month"].astype(str)
                area_df = area_df[area_df["year_month"] != now_ym]
                by_month = area_df.groupby("year_month")["transaction_count"].sum().sort_index()
                if len(by_month) >= 15:
                    recent_avg = float(by_month.tail(3).mean())
                    year_ago_avg = float(by_month.iloc[-15:-12].mean())
                    insights["area_transaction_volume_recent_avg"] = recent_avg
                    insights["area_transaction_volume_year_ago_avg"] = year_ago_avg
                    insights["area_transaction_volume_trend"] = ((recent_avg / year_ago_avg) - 1) * 100 if year_ago_avg > 0 else 0
        except Exception:
            pass
        
        # === Supply Dynamics ===
        if 'supply_units' in history.columns:
            supply = latest.get('supply_units')
            insights['supply_pipeline'] = int(supply) if pd.notna(supply) else None
        if 'units_completing' in history.columns:
            # Sum of units completing in recent months
            completing_sum = history['units_completing'].tail(6).sum()
            insights['units_completing_6m'] = int(completing_sum) if pd.notna(completing_sum) else None
        if 'active_projects' in history.columns:
            active = latest.get('active_projects')
            insights['active_projects'] = int(active) if pd.notna(active) else None
        
        # === Developer Performance (from training data features) ===
        if 'dev_total_projects' in history.columns:
            dev_projects = latest.get('dev_total_projects')
            if pd.notna(dev_projects):
                insights['developer_projects'] = int(dev_projects)
                insights['developer_completed'] = int(latest.get('dev_completed_projects') or 0) if pd.notna(latest.get('dev_completed_projects')) else 0
                insights['developer_units'] = int(latest.get('dev_total_units') or 0) if pd.notna(latest.get('dev_total_units')) else 0
                insights['developer_avg_completion'] = float(latest.get('dev_avg_completion') or 0) if pd.notna(latest.get('dev_avg_completion')) else 0
        
        # === Market Context ===
        if 'eibor_3m' in history.columns:
            eibor_val = latest.get('eibor_3m')
            if pd.notna(eibor_val):
                insights['current_eibor_3m'] = float(eibor_val)
                # Check if rates are rising or falling
                eibor = history['eibor_3m'].dropna()
                if len(eibor) >= 6:
                    eibor_6m_ago = eibor.iloc[-6]
                    if pd.notna(eibor_6m_ago):
                        insights['eibor_change_6m'] = float(eibor_val) - float(eibor_6m_ago)
        
        # === Tourism Data ===
        if 'visitors_total' in history.columns:
            visitors_val = latest.get('visitors_total')
            if pd.notna(visitors_val) and visitors_val > 0:
                insights['tourism_visitors'] = int(visitors_val)
                visitors = history['visitors_total'].dropna()
                if len(visitors) >= 12:
                    current_visitors = visitors.iloc[-1]
                    visitors_12m_ago = visitors.iloc[-12]
                    if pd.notna(visitors_12m_ago) and visitors_12m_ago > 0:
                        insights['tourism_change_12m'] = ((current_visitors / visitors_12m_ago) - 1) * 100
        
        # === Data Recency ===
        insights['data_as_of'] = str(latest.get('year_month', ''))
        insights['months_of_history'] = len(history)
        
        return insights
    
    def _run_model_inference(
        self,
        group_id: str,
        prediction_length: int = 12
    ) -> Optional[Dict[str, Any]]:
        """
        Run actual TFT model inference.
        
        Returns dict with quantile predictions for each target:
        {
            "median_price": np.array([q10, q50, q90] for each horizon),
            "median_rent": np.array([q10, q50, q90] for each horizon)
        }
        """
        if not self.is_loaded or self.model is None:
            return None
        
        try:
            import torch
            from pytorch_forecasting import TimeSeriesDataSet
            
            # Get group history
            history = self.get_group_history(group_id)
            if len(history) < 12:  # Need minimum encoder length
                logger.warning(f"Insufficient history for {group_id}: {len(history)} rows")
                return None
            
            # Prepare prediction data
            # For TFT, we need to create a proper dataset with future known values
            last_time_idx = int(history['time_idx'].max())
            
            # Create future time indices
            future_rows = []
            for i in range(1, prediction_length + 1):
                future_time_idx = last_time_idx + i
                
                # Copy last row and update time-varying known values
                future_row = history.iloc[-1].copy()
                future_row['time_idx'] = future_time_idx
                
                # Update month/quarter cyclical features
                future_month = (int(future_row.get('month', 1)) + i - 1) % 12 + 1
                future_row['month'] = future_month
                future_row['quarter'] = (future_month - 1) // 3 + 1
                future_row['month_sin'] = np.sin(2 * np.pi * future_month / 12)
                future_row['month_cos'] = np.cos(2 * np.pi * future_month / 12)
                
                # Known future covariates: use recent average as default (better than hard 0).
                if "units_completing" in history.columns:
                    uc = pd.to_numeric(history["units_completing"], errors="coerce").dropna()
                    future_row["units_completing"] = float(uc.tail(6).mean()) if len(uc) else 0.0
                else:
                    future_row["units_completing"] = 0.0

                # If months_to_handover exists, count it down each month for realism.
                if "months_to_handover" in history.columns:
                    mth = pd.to_numeric(history.iloc[-1].get("months_to_handover"), errors="coerce")
                    if pd.notna(mth):
                        future_row["months_to_handover"] = max(float(mth) - float(i), 0.0)

                # Lifecycle-aware handover features: keep them consistent for future rows.
                # If months_to_handover_signed exists (positive pre-handover, negative post),
                # decrement it each month; derive months_since_handover + handover_window_6m.
                if "months_to_handover_signed" in history.columns:
                    mths = pd.to_numeric(history.iloc[-1].get("months_to_handover_signed"), errors="coerce")
                    if pd.notna(mths):
                        signed = float(mths) - float(i)
                        future_row["months_to_handover_signed"] = signed
                        if "months_since_handover" in history.columns:
                            future_row["months_since_handover"] = max(-signed, 0.0)
                        if "handover_window_6m" in history.columns:
                            future_row["handover_window_6m"] = 1.0 if abs(signed) <= 6.0 else 0.0

                # If months_since_launch exists, increment each month for realism.
                if "months_since_launch" in history.columns:
                    msl = pd.to_numeric(history.iloc[-1].get("months_since_launch"), errors="coerce")
                    if pd.notna(msl):
                        future_row["months_since_launch"] = float(msl) + float(i)

                # Ensure features expected by the trained dataset exist for future rows.
                # Some checkpoints expect this flag as a (categorical/binary) feature.
                if "has_actual_rent" in history.columns:
                    future_row["has_actual_rent"] = history.iloc[-1].get("has_actual_rent")
                
                future_rows.append(future_row)
            
            # Combine history + future
            pred_data = pd.concat([history, pd.DataFrame(future_rows)], ignore_index=True)

            # --- IMPORTANT: Coerce categorical features to match training expectations ---
            # PyTorch Forecasting expects categorified string-like columns for categorical features.
            # If these are numeric (e.g., month=12), inference can fail with:
            #   "Data type of category month was found to be numeric - use a string type / categorified string"
            # Also: avoid producing the literal string "nan" (unknown category) which can crash with:
            #   "Unknown category 'nan' encountered..."
            # Determine which categorical columns the checkpoint expects and coerce them.
            expected_categoricals = set()
            try:
                dp = getattr(self.model, "dataset_parameters", {}) or {}
                for k in ["static_categoricals", "time_varying_known_categoricals"]:
                    v = dp.get(k)
                    if isinstance(v, list):
                        expected_categoricals.update(v)
            except Exception:
                expected_categoricals = set()

            if "month" in pred_data.columns and ("month" in expected_categoricals or not expected_categoricals):
                m = pd.to_numeric(pred_data["month"], errors="coerce")
                # Fill missing months from nearby values; final fallback to 1
                m = m.ffill().bfill().fillna(1).astype(int)
                pred_data["month"] = m.astype(str)
            if "quarter" in pred_data.columns and ("quarter" in expected_categoricals or not expected_categoricals):
                # Prefer deriving from month to keep month/quarter consistent
                if "month" in pred_data.columns:
                    q = ((pd.to_numeric(pred_data["month"], errors="coerce").fillna(1).astype(int) - 1) // 3 + 1).astype(int)
                    pred_data["quarter"] = q.astype(str)
                else:
                    q = pd.to_numeric(pred_data["quarter"], errors="coerce").ffill().bfill().fillna(1).astype(int)
                    pred_data["quarter"] = q.astype(str)

            # Coerce remaining expected categoricals (e.g., developer_brand, developer_registered_name, reg_type_dld).
            # Use a stable placeholder for missing values ("nan") so it can match encoder classes if present.
            for col in sorted(expected_categoricals):
                if col in {"month", "quarter"}:
                    continue
                if col in pred_data.columns:
                    pred_data[col] = pred_data[col].fillna("nan").astype(str)
            if "time_idx" in pred_data.columns:
                pred_data["time_idx"] = pd.to_numeric(pred_data["time_idx"], errors="coerce").fillna(0).astype(int)
            if "units_completing" in pred_data.columns:
                pred_data["units_completing"] = pd.to_numeric(pred_data["units_completing"], errors="coerce").fillna(0).astype(float)

            # Some models expect has_actual_rent; derive it if missing.
            # Treat as string/categorical to be safe with dataset categorification.
            if "has_actual_rent" not in pred_data.columns:
                if "median_rent" in pred_data.columns:
                    pred_data["has_actual_rent"] = (pd.to_numeric(pred_data["median_rent"], errors="coerce").fillna(0) > 0).astype(int).astype(str)
                else:
                    pred_data["has_actual_rent"] = "0"
            else:
                # normalize existing column
                pred_data["has_actual_rent"] = pred_data["has_actual_rent"].astype(str)

            # Ensure common numeric model features are present and non-null.
            # The trained dataset disallows NA values for real-valued covariates.
            for must_have in ["months_since_launch", "median_rent_sqft"]:
                if must_have not in pred_data.columns:
                    pred_data[must_have] = 0.0

            # Fill remaining numeric columns with safe defaults.
            # Keep string/categorical columns as-is; for everything else, coerce to numeric and fill NaNs.
            skip = {"group_id", "year_month"} | {"area_name", "property_type", "bedroom", "reg_type", "developer_name", "month", "quarter", "has_actual_rent"}
            for col in pred_data.columns:
                if col in skip:
                    continue
                if pred_data[col].dtype == object:
                    continue
                pred_data[col] = pd.to_numeric(pred_data[col], errors="coerce").fillna(0)
            
            # Make prediction using raw mode for quantiles.
            # pytorch-forecasting/lightning versions can return (pred, x) or (pred, x, index/decoder) etc.
            res = self.model.predict(
                pred_data,
                mode="raw",
                return_x=True
            )
            if isinstance(res, tuple):
                raw_predictions = res[0]
                x = res[1] if len(res) > 1 else None
            else:
                raw_predictions = res
                x = None
            
            # Extract quantile predictions.
            # Depending on pytorch-forecasting version, raw output may be exposed as:
            # - Output.output
            # - Output.prediction
            # or be a tensor directly.
            output = getattr(raw_predictions, "output", None)
            if output is None:
                output = getattr(raw_predictions, "prediction", None)
            if output is None:
                output = raw_predictions
            
            # Get indices for quantiles we care about
            # Default quantiles: [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]
            results: Dict[str, Any] = {}
            
            # Output forms:
            # 1) list[target] -> Tensor[batch, horizon, quantiles]  (common for multi-target)
            # 2) Tensor[batch, horizon, targets, quantiles]
            if isinstance(output, list) and len(output) >= 2:
                price_t = output[0]  # [batch, horizon, quantiles]
                rent_t = output[1]
                results["median_price"] = {
                    "q10": float(price_t[0, -1, QUANTILE_10].item()),
                    "q50": float(price_t[0, -1, QUANTILE_50].item()),
                    "q90": float(price_t[0, -1, QUANTILE_90].item()),
                }
                results["median_rent"] = {
                    "q10": float(rent_t[0, -1, QUANTILE_10].item()),
                    "q50": float(rent_t[0, -1, QUANTILE_50].item()),
                    "q90": float(rent_t[0, -1, QUANTILE_90].item()),
                }
            else:
                try:
                    if len(output.shape) == 4:
                        price_preds = output[0, :, 0, :].cpu().numpy()
                        rent_preds = output[0, :, 1, :].cpu().numpy()
                        results["median_price"] = {
                            "q10": float(price_preds[-1, QUANTILE_10]),
                            "q50": float(price_preds[-1, QUANTILE_50]),
                            "q90": float(price_preds[-1, QUANTILE_90]),
                        }
                        results["median_rent"] = {
                            "q10": float(rent_preds[-1, QUANTILE_10]),
                            "q50": float(rent_preds[-1, QUANTILE_50]),
                            "q90": float(rent_preds[-1, QUANTILE_90]),
                        }
                    else:
                        logger.warning(f"Unexpected raw prediction output shape: {getattr(output, 'shape', None)}")
                        return None
                except Exception as e:
                    logger.warning(f"Unexpected raw prediction output type: {type(output)} ({e})")
                    return None

            # --- Model attribution (non-causal) ---
            # Use TFT's interpretability output if available, and extract a compact top-driver list.
            try:
                interpretation = self.model.interpret_output(raw_predictions, reduction="sum")
                drivers: List[ModelDriver] = []

                # In pytorch-forecasting 1.5.x, interpret_output returns tensors:
                # - static_variables: len(static_categoricals)+len(static_reals)
                # - encoder_variables: len(time_varying_categoricals_encoder)+len(time_varying_reals_encoder)
                # - decoder_variables: len(time_varying_categoricals_decoder)+len(time_varying_reals_decoder)
                static_names = list(self.model.hparams.get("static_categoricals", [])) + list(self.model.hparams.get("static_reals", []))
                encoder_names = list(self.model.hparams.get("time_varying_categoricals_encoder", [])) + list(self.model.hparams.get("time_varying_reals_encoder", []))
                decoder_names = list(self.model.hparams.get("time_varying_categoricals_decoder", [])) + list(self.model.hparams.get("time_varying_reals_decoder", []))

                def _tensor_to_items(names: List[str], t) -> List[tuple[str, float]]:
                    try:
                        # reduction="sum" returns 1D tensor of importances
                        vals = t.detach().cpu().float().numpy().tolist()
                    except Exception:
                        return []
                    if not isinstance(vals, list):
                        return []
                    if len(vals) != len(names):
                        # Fallback: best-effort zip
                        n = min(len(vals), len(names))
                        vals = vals[:n]
                        names = names[:n]
                    total = float(sum(vals)) if sum(vals) else 0.0
                    out = []
                    for n, v in zip(names, vals):
                        vv = float(v)
                        # Normalize to share of total to be comparable
                        imp = (vv / total) if total > 0 else 0.0
                        out.append((str(n), imp))
                    out.sort(key=lambda x: x[1], reverse=True)
                    return out

                for feat, imp in _tensor_to_items(static_names, interpretation.get("static_variables"))[:5]:
                    drivers.append(ModelDriver(feature=feat, importance=imp, target="overall"))
                for feat, imp in _tensor_to_items(encoder_names, interpretation.get("encoder_variables"))[:8]:
                    drivers.append(ModelDriver(feature=feat, importance=imp, target="overall"))
                for feat, imp in _tensor_to_items(decoder_names, interpretation.get("decoder_variables"))[:5]:
                    drivers.append(ModelDriver(feature=feat, importance=imp, target="overall"))

                # Deduplicate by feature, keeping the max importance
                best: Dict[str, float] = {}
                for d in drivers:
                    best[d.feature] = max(best.get(d.feature, 0.0), float(d.importance))
                top = sorted(best.items(), key=lambda x: x[1], reverse=True)

                # Filter out purely technical / non-investor-facing features.
                # These can dominate interpretability (e.g., seasonality encodings) but aren't helpful for agents/investors.
                blocked = {
                    "time_idx", "relative_time_idx",
                    "month", "quarter", "month_sin", "month_cos",
                    "area_name", "developer_name", "property_type", "bedroom", "reg_type",
                    "median_price_scale", "median_rent_scale",
                    # Not investor-facing drivers (implementation / data-availability flags)
                    "has_actual_rent", "rent_count",
                }
                filtered = []
                for f, v in top:
                    if f in blocked:
                        continue
                    if str(f).startswith("dev_"):
                        continue
                    if str(f).endswith("_scale"):
                        continue
                    filtered.append((f, v))
                # Keep top 8 investor-relevant drivers
                # Attach current value + recent change for explainability (best-effort).
                def _current_and_change(col: str) -> tuple[Optional[float], Optional[float]]:
                    if col not in history.columns:
                        return None, None
                    s = pd.to_numeric(history[col], errors="coerce").dropna()
                    if len(s) == 0:
                        return None, None
                    cur = float(s.iloc[-1])
                    ch = None
                    if len(s) >= 7:
                        ch = float(cur - float(s.iloc[-7]))
                    return cur, ch

                top_drivers = []
                for f, v in filtered[:8]:
                    cur, ch = _current_and_change(str(f))
                    top_drivers.append(ModelDriver(feature=str(f), importance=float(v), target="overall", current_value=cur, change_6m=ch))

                # --- What-if impacts (local sensitivity) ---
                # Re-run the model with small controlled tweaks to a few top drivers to approximate "when X changes, model output moves by Y".
                def _extract_price_q50_sqm(pred_data_cf: pd.DataFrame) -> Optional[float]:
                    try:
                        res_cf = self.model.predict(pred_data_cf, mode="raw", return_x=False)
                        out_cf = getattr(res_cf, "output", None)
                        if out_cf is None:
                            out_cf = getattr(res_cf, "prediction", None)
                        if out_cf is None:
                            out_cf = res_cf
                        if isinstance(out_cf, list) and len(out_cf) >= 1:
                            price_t_cf = out_cf[0]
                            return float(price_t_cf[0, -1, QUANTILE_50].item())
                        if hasattr(out_cf, "shape") and len(out_cf.shape) == 4:
                            return float(out_cf[0, -1, 0, QUANTILE_50].item())
                    except Exception:
                        return None
                    return None

                baseline_q50_sqm = results.get("median_price", {}).get("q50")
                what_if: List[Dict[str, Any]] = []
                if baseline_q50_sqm and top_drivers:
                    # Supported perturbations (applied on future rows only)
                    perturb_specs: Dict[str, Dict[str, Any]] = {
                        # Use larger but still plausible shocks so output isn't rounded to 0.0%.
                        "market_median_price": {"kind": "mult", "value": 1.05, "desc": "up 5% (assumed over next 6 months)"},
                        "market_transactions": {"kind": "mult", "value": 1.25, "desc": "up 25% (assumed over next 6 months)"},
                        "units_completing": {"kind": "mult", "value": 1.25, "desc": "up 25% (assumed over next 6 months)"},
                        "eibor_3m": {"kind": "add", "value": 1.00, "desc": "up +1.00 percentage point (assumed over next 6 months)"},
                        "months_to_handover": {"kind": "add", "value": -6.0, "desc": "6 months sooner (faster delivery assumption)"},
                    }
                    # Only do this for a few top drivers to keep latency manageable
                    pick = [d.feature for d in top_drivers if d.feature in perturb_specs][:3]
                    if pick:
                        last_idx = int(history["time_idx"].max())
                        for feat in pick:
                            spec = perturb_specs[feat]
                            try:
                                cf = pred_data.copy(deep=True)
                                # Apply shock to the near term (last 3 observed months + future) to reflect a regime shift.
                                mask_future = cf["time_idx"] > (last_idx - 3)
                                if feat not in cf.columns:
                                    continue
                                s = pd.to_numeric(cf.loc[mask_future, feat], errors="coerce").fillna(0.0)
                                if spec["kind"] == "mult":
                                    cf.loc[mask_future, feat] = (s * float(spec["value"])).astype(float)
                                elif spec["kind"] == "add":
                                    cf.loc[mask_future, feat] = (s + float(spec["value"])).astype(float)
                                # Ensure numeric sanitation post-change
                                cf.loc[mask_future, feat] = pd.to_numeric(cf.loc[mask_future, feat], errors="coerce").fillna(0.0)
                                q50_cf = _extract_price_q50_sqm(cf)
                                if q50_cf is None:
                                    continue
                                delta_pct = ((q50_cf / float(baseline_q50_sqm)) - 1) * 100 if baseline_q50_sqm else None
                                what_if.append({
                                    "feature": feat,
                                    "assumption": spec["desc"],
                                    "baseline_q50_sqm": float(baseline_q50_sqm),
                                    "counterfactual_q50_sqm": float(q50_cf),
                                    "delta_percent_vs_baseline": float(delta_pct) if delta_pct is not None else None,
                                })
                            except Exception:
                                continue

                if top_drivers:
                    results["_model_attribution"] = ModelAttribution(top_drivers=top_drivers, what_if_impacts=what_if)
            except Exception:
                # Attribution is optional; do not fail inference if interpretability isn't available.
                pass
            
            return results
            
        except Exception as e:
            logger.error(f"Model inference error: {e}")
            return None
    
    def predict(
        self,
        area: str,
        property_type: str = "Unit",
        bedroom: str = "2BR",
        reg_type: str = "OffPlan",
        developer: str = "Unknown",
        price: Optional[float] = None,
        handover_months: Optional[int] = None,
        unit_sqft: Optional[float] = None
    ) -> TFTPrediction:
        """
        Make price and rent predictions.
        
        Args:
            area: Area name (e.g., "Business Bay")
            property_type: "Unit" or "Villa"
            bedroom: e.g., "2BR", "Studio"
            reg_type: "OffPlan" or "Ready"
            developer: Developer name (Arabic or English)
            price: Optional purchase price for context
            handover_months: Months until handover (for off-plan)
            unit_sqft: Optional unit size in square feet (for total value calculation)
            
        Returns:
            TFTPrediction with forecasts
        """
        # Find matching group
        matched_id, match_type, confidence = self.find_matching_group(
            area, property_type, bedroom, reg_type, developer
        )
        
        # Get current values
        current = self.get_current_values(matched_id) if matched_id else {}
        
        # Note: median_price from training data is in AED/sqm, convert to AED/sqft
        current_sqm = current.get("median_price")
        current_sqft = current_sqm / SQM_TO_SQFT if current_sqm else None
        current_rent = current.get("median_rent")
        has_rent = current.get("has_actual_rent", False)
        
        # Default horizon: prefer the model's trained max horizon (checkpoint-limited).
        # If unknown (e.g., model not loaded), fall back to 12m since that's investor-relevant.
        model_max_horizon = None
        try:
            model_max_horizon = int((getattr(self.model, "dataset_parameters", {}) or {}).get("max_prediction_length") or 0) or None
        except Exception:
            model_max_horizon = None
        requested_horizon = int(handover_months) if handover_months else int(model_max_horizon or 12)
        forecast_horizon = requested_horizon
        
        # Try model inference if available (note: TFT checkpoint has a fixed max_prediction_length)
        model_results: Optional[Dict[str, Any]] = None
        if self.is_loaded and matched_id:
            horizon_for_model = min(forecast_horizon, model_max_horizon) if model_max_horizon else forecast_horizon
            model_results = self._run_model_inference(matched_id, horizon_for_model)
        
        # Build forecasts from model results or fallback to estimation
        model_attribution: Optional[ModelAttribution] = None
        if model_results:
            # Use actual model predictions
            price_preds = model_results.get("median_price", {})
            rent_preds = model_results.get("median_rent", {})
            model_attribution = model_results.get("_model_attribution")
            
            # Model outputs AED/sqm, convert to AED/sqft
            # Fallback values are already in reasonable AED/sqft
            q10_sqm = price_preds.get("q10")
            q50_sqm = price_preds.get("q50")
            q90_sqm = price_preds.get("q90")
            
            forecast_sqft_low = q10_sqm / SQM_TO_SQFT if q10_sqm else (current_sqft * 1.05 if current_sqft else 1350)
            forecast_sqft_median = q50_sqm / SQM_TO_SQFT if q50_sqm else (current_sqft * 1.12 if current_sqft else 1500)
            forecast_sqft_high = q90_sqm / SQM_TO_SQFT if q90_sqm else (current_sqft * 1.20 if current_sqft else 1725)
            
            forecast_rent_low = rent_preds.get("q10", current_rent * 0.95 if current_rent else 72000)
            forecast_rent_median = rent_preds.get("q50", current_rent * 1.05 if current_rent else 80000)
            forecast_rent_high = rent_preds.get("q90", current_rent * 1.10 if current_rent else 88000)
            
            # Increase confidence for actual model predictions
            confidence = min(confidence + 10, 100)
            # Ensure we report the actual horizon the model can forecast (checkpoint-limited).
            if model_max_horizon:
                forecast_horizon = min(forecast_horizon, model_max_horizon)
        else:
            # Fallback: estimate based on current values and typical appreciation
            annual_appreciation = 0.12
            monthly_appreciation = annual_appreciation / 12
            total_appreciation = (1 + monthly_appreciation) ** forecast_horizon - 1
            
            if current_sqft:
                forecast_sqft_median = current_sqft * (1 + total_appreciation)
                forecast_sqft_low = forecast_sqft_median * 0.9
                forecast_sqft_high = forecast_sqft_median * 1.15
            else:
                forecast_sqft_median = 1500
                forecast_sqft_low = 1350
                forecast_sqft_high = 1725
                current_sqft = 1350
            
            # Rent estimation
            if current_rent and has_rent:
                rent_appreciation = 0.05
                rent_monthly = rent_appreciation / 12
                rent_total = (1 + rent_monthly) ** forecast_horizon - 1
                
                forecast_rent_median = current_rent * (1 + rent_total)
                forecast_rent_low = forecast_rent_median * 0.92
                forecast_rent_high = forecast_rent_median * 1.08
            else:
                if price:
                    estimated_rent = price * 0.06
                    forecast_rent_median = estimated_rent
                    forecast_rent_low = estimated_rent * 0.9
                    forecast_rent_high = estimated_rent * 1.1
                else:
                    forecast_rent_median = 80000
                    forecast_rent_low = 72000
                    forecast_rent_high = 88000
                current_rent = None
                has_rent = False
        
        # Calculate appreciation
        appreciation_percent = None
        if current_sqft and forecast_sqft_median:
            appreciation_percent = ((forecast_sqft_median / current_sqft) - 1) * 100

        # Longer horizon projections (12m/24m) derived from model-implied monthly growth over the model horizon.
        forecast_sqft_low_12m = forecast_sqft_median_12m = forecast_sqft_high_12m = None
        forecast_sqft_low_24m = forecast_sqft_median_24m = forecast_sqft_high_24m = None
        appreciation_percent_12m = appreciation_percent_24m = None
        long_horizon_method = None
        try:
            if model_results and current_sqft and forecast_horizon and forecast_horizon > 0:
                def _extrapolate(cur: float, end: float, base_h: int, target_h: int) -> float:
                    if cur <= 0 or base_h <= 0:
                        return end
                    g = (end / cur) ** (1.0 / base_h) - 1.0
                    return cur * ((1.0 + g) ** target_h)

                if forecast_sqft_low and forecast_sqft_median and forecast_sqft_high:
                    forecast_sqft_low_12m = _extrapolate(current_sqft, forecast_sqft_low, forecast_horizon, 12)
                    forecast_sqft_median_12m = _extrapolate(current_sqft, forecast_sqft_median, forecast_horizon, 12)
                    forecast_sqft_high_12m = _extrapolate(current_sqft, forecast_sqft_high, forecast_horizon, 12)
                    forecast_sqft_low_24m = _extrapolate(current_sqft, forecast_sqft_low, forecast_horizon, 24)
                    forecast_sqft_median_24m = _extrapolate(current_sqft, forecast_sqft_median, forecast_horizon, 24)
                    forecast_sqft_high_24m = _extrapolate(current_sqft, forecast_sqft_high, forecast_horizon, 24)
                    appreciation_percent_12m = ((forecast_sqft_median_12m / current_sqft) - 1) * 100 if forecast_sqft_median_12m else None
                    appreciation_percent_24m = ((forecast_sqft_median_24m / current_sqft) - 1) * 100 if forecast_sqft_median_24m else None
                    long_horizon_method = "extrapolated_from_model_implied_monthly_cagr"
        except Exception:
            pass
        
        # Calculate yield
        estimated_yield = None
        if price and forecast_rent_median:
            estimated_yield = (forecast_rent_median / price) * 100
        
        # Calculate total values if unit_sqft provided
        current_total = None
        forecast_total_low = None
        forecast_total_median = None
        forecast_total_high = None
        forecast_total_low_12m = None
        forecast_total_median_12m = None
        forecast_total_high_12m = None
        
        if unit_sqft and unit_sqft > 0:
            if current_sqft:
                current_total = round(current_sqft * unit_sqft, 0)
            forecast_total_low = round(forecast_sqft_low * unit_sqft, 0)
            forecast_total_median = round(forecast_sqft_median * unit_sqft, 0)
            forecast_total_high = round(forecast_sqft_high * unit_sqft, 0)
            # Calculate 12m total values from extrapolated sqft values
            if forecast_sqft_low_12m is not None:
                forecast_total_low_12m = round(forecast_sqft_low_12m * unit_sqft, 0)
            if forecast_sqft_median_12m is not None:
                forecast_total_median_12m = round(forecast_sqft_median_12m * unit_sqft, 0)
            if forecast_sqft_high_12m is not None:
                forecast_total_high_12m = round(forecast_sqft_high_12m * unit_sqft, 0)
        
        price_forecast = PriceForecast(
            current_sqft=round(current_sqft, 2) if current_sqft else None,
            forecast_sqft_low=round(forecast_sqft_low, 2),
            forecast_sqft_median=round(forecast_sqft_median, 2),
            forecast_sqft_high=round(forecast_sqft_high, 2),
            forecast_horizon_months=forecast_horizon,
            appreciation_percent=round(appreciation_percent, 1) if appreciation_percent else None,
            forecast_sqft_low_12m=round(forecast_sqft_low_12m, 2) if forecast_sqft_low_12m is not None else None,
            forecast_sqft_median_12m=round(forecast_sqft_median_12m, 2) if forecast_sqft_median_12m is not None else None,
            forecast_sqft_high_12m=round(forecast_sqft_high_12m, 2) if forecast_sqft_high_12m is not None else None,
            appreciation_percent_12m=round(appreciation_percent_12m, 1) if appreciation_percent_12m is not None else None,
            forecast_sqft_low_24m=round(forecast_sqft_low_24m, 2) if forecast_sqft_low_24m is not None else None,
            forecast_sqft_median_24m=round(forecast_sqft_median_24m, 2) if forecast_sqft_median_24m is not None else None,
            forecast_sqft_high_24m=round(forecast_sqft_high_24m, 2) if forecast_sqft_high_24m is not None else None,
            appreciation_percent_24m=round(appreciation_percent_24m, 1) if appreciation_percent_24m is not None else None,
            long_horizon_method=long_horizon_method,
            unit_sqft=unit_sqft,
            current_total_value=current_total,
            forecast_total_value_low=forecast_total_low,
            forecast_total_value_median=forecast_total_median,
            forecast_total_value_high=forecast_total_high,
            forecast_total_value_low_12m=forecast_total_low_12m,
            forecast_total_value_median_12m=forecast_total_median_12m,
            forecast_total_value_high_12m=forecast_total_high_12m
        )
        
        rent_forecast = RentForecast(
            current_annual=round(current_rent, 0) if current_rent else None,
            forecast_annual_low=round(forecast_rent_low, 0),
            forecast_annual_median=round(forecast_rent_median, 0),
            forecast_annual_high=round(forecast_rent_high, 0),
            has_actual_rent=has_rent,
            estimated_yield_percent=round(estimated_yield, 2) if estimated_yield else None
        )
        
        # Compute trend insights from historical data
        trend_insights = None
        if matched_id:
            raw_insights = self.compute_trend_insights(matched_id)
            if raw_insights:
                trend_insights = TrendInsights(**raw_insights)
        
        return TFTPrediction(
            price_forecast=price_forecast,
            rent_forecast=rent_forecast,
            trend_insights=trend_insights,
            model_attribution=model_attribution,
            match_type="model" if model_results else match_type,
            matched_group_id=matched_id,
            confidence=confidence
        )


# Singleton instance
_service: Optional[TFTInferenceService] = None


def get_tft_service() -> TFTInferenceService:
    """Get singleton TFTInferenceService instance."""
    global _service
    if _service is None:
        _service = TFTInferenceService()
    return _service


async def predict(
    area: str,
    property_type: str = "Unit",
    bedroom: str = "2BR",
    reg_type: str = "OffPlan",
    developer: str = "Unknown",
    price: Optional[float] = None,
    handover_months: Optional[int] = None,
    unit_sqft: Optional[float] = None
) -> TFTPrediction:
    """
    Convenience function for making predictions.
    
    Args:
        area: Area name (DLD name like "Al Barsha South Fourth")
        property_type: "Unit" or "Villa"
        bedroom: e.g., "2BR", "Studio"
        reg_type: "OffPlan" or "Ready"
        developer: Developer name (Arabic or English)
        price: Optional purchase price for context
        handover_months: Months until handover (for off-plan)
        unit_sqft: Optional unit size in square feet (for total value calculation)
    """
    service = get_tft_service()
    return service.predict(
        area=area,
        property_type=property_type,
        bedroom=bedroom,
        reg_type=reg_type,
        developer=developer,
        price=price,
        handover_months=handover_months,
        unit_sqft=unit_sqft
    )
