"""
Response Generator Service

Generates natural language responses from predictions and trend data.
Uses OpenAI for response synthesis with low temperature for consistency.
"""

import logging
from typing import Optional, Dict, Any
from pathlib import Path
from pydantic import BaseModel
from openai import OpenAI

from ..core.config import get_settings
from .tft_inference import TFTPrediction, TrendInsights
from .trend_lookup import DeveloperStats, AreaStats, RentBenchmark
from .entity_validator import ValidatedEntities

logger = logging.getLogger(__name__)

AREA_CONFIDENCE_THRESHOLD = 80.0
DEVELOPER_CONFIDENCE_THRESHOLD = 70.0
MODEL_FORECAST_CONFIDENCE_THRESHOLD = 70.0
MODEL_FORECAST_MATCH_TYPES = {"exact", "exact_all_developers", "partial_area_bedroom", "model"}

class GeneratedResponse(BaseModel):
    """Generated analysis response."""
    content: str
    summary: str
    report_data: Dict[str, Any]


def format_currency(value: float, suffix: str = "") -> str:
    """Format number as currency string."""
    if value >= 1_000_000:
        return f"AED {value/1_000_000:.2f}M{suffix}"
    elif value >= 1_000:
        return f"AED {value/1_000:.0f}K{suffix}"
    else:
        return f"AED {value:.0f}{suffix}"


def build_context_prompt(
    query: str,
    entities: ValidatedEntities,
    prediction: TFTPrediction,
    trends: Dict[str, Any],
    price: Optional[float] = None,
    handover_months: Optional[int] = None,
    unit_sqft: Optional[float] = None
) -> str:
    """Build context prompt for response generation."""
    
    sections = []
    
    # Property details - use display names for user-friendly output
    prop_details = []
    
    # Developer - use building developer name if available, otherwise English name
    display_developer = entities.building_developer_name or entities.developer_english
    if display_developer:
        prop_details.append(f"Developer: {display_developer}")
    
    # Area - use display name (e.g., "JVC") instead of DLD name (e.g., "Al Barsha South Fourth")
    display_area = entities.area_display_name or entities.area_name
    if display_area:
        prop_details.append(f"Area: {display_area}")
    
    if entities.bedroom:
        prop_details.append(f"Bedrooms: {entities.bedroom}")
    if entities.reg_type:
        prop_details.append(f"Type: {entities.reg_type}")
    if price:
        prop_details.append(f"Price: {format_currency(price)}")
    if unit_sqft:
        prop_details.append(f"Size: {unit_sqft:.0f} sqft")
    if handover_months and (entities.reg_type or "").lower() == "offplan":
        prop_details.append(f"Handover: in {int(handover_months)} months")
    
    if prop_details:
        sections.append("**Property Details:**\n" + "\n".join(f"- {d}" for d in prop_details))

    # Resolved entities + confidence (always include; factual)
    resolved_section = "**Resolved Entities:**\n"
    resolved_section += f"- Area (DLD): {entities.area_name or 'Not provided'} (method: {entities.area_resolution_method or 'n/a'}, confidence: {entities.area_confidence:.0f}%)\n"
    resolved_section += f"- Area (display): {display_area or 'Not available'}\n"
    if entities.developer_english:
        resolved_section += f"- Developer (input): {entities.developer_english}\n"
    resolved_section += f"- Developer (Arabic for model): {entities.developer_arabic or 'Unknown'} (method: {entities.developer_resolution_method or 'n/a'}, confidence: {entities.developer_confidence:.0f}%)\n"
    resolved_section += f"- Segment: {entities.property_type} | {entities.bedroom} | {entities.reg_type}\n"
    sections.append(resolved_section)
    
    # Get forecast data for analysis
    pf = prediction.price_forecast
    area_stats: Optional[AreaStats] = trends.get("area_stats")

    # ============================
    # Off‑Plan Investment Snapshot
    # ============================
    # For OffPlan units, we return an investor-oriented snapshot (deal summary → handover value → post-handover rent/yield).
    if (entities.reg_type or "").lower() == "offplan":
        display_area = entities.area_display_name or entities.area_name
        display_developer = entities.building_developer_name or entities.developer_english

        # Required unit inputs
        required_missing = []
        if not unit_sqft:
            required_missing.append("Unit size (sqft)")
        if not price:
            required_missing.append("Purchase price (AED)")

        # Derived purchase price per sqft
        purchase_price_sqft = (float(price) / float(unit_sqft)) if (price and unit_sqft and unit_sqft > 0) else None

        # Handover horizon (months)
        h = int(handover_months) if handover_months else None

        # Base horizon used by the checkpoint for the forecast end-values
        base_h = int(pf.forecast_horizon_months or 0) or None

        def _extrapolate(cur: Optional[float], end: Optional[float], base_horizon: Optional[int], target_horizon: int) -> Optional[float]:
            if cur is None or end is None or not base_horizon or base_horizon <= 0 or target_horizon <= 0:
                return None
            if cur <= 0:
                return None
            g = (end / cur) ** (1.0 / float(base_horizon)) - 1.0
            return cur * ((1.0 + g) ** float(target_horizon))

        # Target horizons: at handover and +12m post-handover
        target_h = h or (base_h or 12)
        target_h_plus_12 = target_h + 12

        cur_sqft = pf.current_sqft
        # Per-sqft projections at handover
        handover_sqft_med = _extrapolate(cur_sqft, pf.forecast_sqft_median, base_h, target_h) or pf.forecast_sqft_median
        handover_sqft_low = _extrapolate(cur_sqft, pf.forecast_sqft_low, base_h, target_h) or pf.forecast_sqft_low
        handover_sqft_high = _extrapolate(cur_sqft, pf.forecast_sqft_high, base_h, target_h) or pf.forecast_sqft_high

        # Per-sqft projections at +12m post-handover
        post12_sqft_med = _extrapolate(cur_sqft, pf.forecast_sqft_median, base_h, target_h_plus_12)
        post12_sqft_low = _extrapolate(cur_sqft, pf.forecast_sqft_low, base_h, target_h_plus_12)
        post12_sqft_high = _extrapolate(cur_sqft, pf.forecast_sqft_high, base_h, target_h_plus_12)

        # Totals (AED)
        handover_total_med = (handover_sqft_med * unit_sqft) if (handover_sqft_med and unit_sqft) else None
        handover_total_low = (handover_sqft_low * unit_sqft) if (handover_sqft_low and unit_sqft) else None
        handover_total_high = (handover_sqft_high * unit_sqft) if (handover_sqft_high and unit_sqft) else None

        post12_total_med = (post12_sqft_med * unit_sqft) if (post12_sqft_med and unit_sqft) else None
        post12_total_low = (post12_sqft_low * unit_sqft) if (post12_sqft_low and unit_sqft) else None
        post12_total_high = (post12_sqft_high * unit_sqft) if (post12_sqft_high and unit_sqft) else None

        # Uplifts vs purchase price
        uplift_handover_aed = (handover_total_med - price) if (handover_total_med and price) else None
        uplift_handover_pct = ((handover_total_med / price) - 1) * 100 if (handover_total_med and price) else None
        uplift_post12_aed = (post12_total_med - price) if (post12_total_med and price) else None
        uplift_post12_pct = ((post12_total_med / price) - 1) * 100 if (post12_total_med and price) else None

        # Rent + yield (post-handover)
        rf = prediction.rent_forecast
        cur_rent = rf.current_annual if rf.current_annual else None
        rent_handover_med = _extrapolate(cur_rent, rf.forecast_annual_median, base_h, target_h) or rf.forecast_annual_median
        rent_handover_low = _extrapolate(cur_rent, rf.forecast_annual_low, base_h, target_h) or rf.forecast_annual_low
        rent_handover_high = _extrapolate(cur_rent, rf.forecast_annual_high, base_h, target_h) or rf.forecast_annual_high

        yield_med = (rent_handover_med / price) * 100 if (rent_handover_med and price) else None
        yield_low = (rent_handover_low / price) * 100 if (rent_handover_low and price) else None
        yield_high = (rent_handover_high / price) * 100 if (rent_handover_high and price) else None

        # Drivers (sales-script labels; no technical feature names)
        driver_labels = {
            "months_to_handover": "Lifecycle timing to handover",
            "months_to_handover_signed": "Lifecycle timing to handover",
            "handover_window_6m": "Handover window (near-term delivery)",
            "units_completing": "Scheduled completions (supply landing soon)",
            "supply_units": "Supply pipeline (upcoming inventory)",
            "active_projects": "Project pipeline activity",
            "transaction_count": "Liquidity (segment transactions)",
            "market_transactions": "Liquidity (wider market)",
            "market_median_price": "Area momentum (price context)",
            "govt_valuation_median": "Government valuation benchmark",
            "eibor_3m": "Rates (EIBOR)",
            "eibor_6m": "Rates (EIBOR)",
            "eibor_12m": "Rates (EIBOR)",
            "visitors_total": "Demand proxy (tourism)",
        }
        drivers: list[str] = []
        if prediction.model_attribution and prediction.model_attribution.top_drivers:
            for d in sorted(prediction.model_attribution.top_drivers, key=lambda x: x.importance, reverse=True):
                label = driver_labels.get(d.feature)
                if not label:
                    continue
                if label not in drivers:
                    drivers.append(label)
                if len(drivers) >= 5:
                    break

        # Report copy (investor-facing)
        title = "Off‑Plan Investment Snapshot"
        from datetime import datetime
        date_generated = datetime.now().strftime("%B %d, %Y")

        # Gating: if the matched series is low quality, we should not present model-derived ranges as “estimates”.
        model_ok = (prediction.confidence or 0) >= MODEL_FORECAST_CONFIDENCE_THRESHOLD and prediction.match_type in MODEL_FORECAST_MATCH_TYPES

        notice_lines = [
            "**IMPORTANT NOTICE**",
            "",
            "This report is generated using an independent, data-driven forecasting model.",
            "It is provided for informational purposes only and does not constitute investment advice.",
        ]

        # 1) Property overview
        unit_config = entities.bedroom or "Not provided"
        property_type = entities.property_type or "Unit"
        if property_type.lower() == "unit":
            property_type = "Apartment"
        property_lines = [
            "**Property Overview**",
            f"- Developer: {display_developer or 'Not provided'}",
            f"- Location / Community: {display_area or 'Not provided'}",
            f"- Property Type: {property_type}",
            f"- Unit Configuration: {unit_config}",
            f"- Unit Size: {unit_sqft:.0f} sqft" if unit_sqft else "- Unit Size: Not provided",
            "- Stage: Off‑Plan",
            f"- Estimated Handover: {'In ~' + str(h) + ' months' if h else 'Not provided'}",
            f"- Report Date: {date_generated}",
        ]

        # 2) Purchase summary
        purchase_lines = [
            "**Purchase Summary**",
            f"- Purchase Price: {format_currency(price)}" if price else "- Purchase Price: Not provided",
            f"- Price per sqft: AED {purchase_price_sqft:,.0f}/sqft" if purchase_price_sqft else "- Price per sqft: Not available",
        ]

        # 3) Projected value outlook
        value_lines = ["**Projected Value Outlook: AI‑Model Forecast**"]
        if required_missing:
            value_lines.append(f"- Missing required inputs: {', '.join(required_missing)}")
        elif not model_ok:
            value_lines.append("- Not available (model match quality too low for a reliable estimate)")
        else:
            value_lines.append("- Estimated Market Value at Handover")
            value_lines.append(f"  - Estimated: {format_currency(handover_total_med)}")
            value_lines.append(f"  - Forecast Range: {format_currency(handover_total_low)} – {format_currency(handover_total_high)}")
            value_lines.append("")
            value_lines.append("- Estimated Market Value 12 Months Post‑Handover")
            value_lines.append(f"  - Estimated: {format_currency(post12_total_med) if post12_total_med else 'Not available'}")
            value_lines.append(
                f"  - Forecast Range: {format_currency(post12_total_low) if post12_total_low else 'Not available'} – {format_currency(post12_total_high) if post12_total_high else 'Not available'}"
            )
            value_lines.append("")
            value_lines.append("- Projected Capital Uplift")
            if uplift_handover_pct is not None:
                value_lines.append(f"  - By Handover: ~{uplift_handover_pct:.1f}%")
            else:
                value_lines.append("  - By Handover: Not available")
            if uplift_post12_pct is not None:
                value_lines.append(f"  - By +12 Months: ~{uplift_post12_pct:.1f}%")
            else:
                value_lines.append("  - By +12 Months: Not available")
            value_lines.append("")
            value_lines.append("These projections are generated using a predictive model trained on historical Dubai market data and comparable unit performance.")

        # 4) Rent & yield (post-handover)
        rent_lines = ["**Rental & Yield Outlook | Post‑Handover**"]
        if required_missing:
            rent_lines.append("- Not available (missing required unit inputs)")
        elif not model_ok:
            rent_lines.append("- Not available (model match quality too low for a reliable estimate)")
            rent_lines.append("- Estimated annual rent after handover (AED/yr): Not available")
            rent_lines.append("- Estimated gross yield after handover (%): Not available")
        else:
            rent_lines.append("- Estimated Annual Rent")
            rent_lines.append(
                f"  - Range: {format_currency(rent_handover_low) if rent_handover_low else 'Not available'} – {format_currency(rent_handover_high) if rent_handover_high else 'Not available'}"
            )
            rent_lines.append(f"  - Median Estimate: {format_currency(rent_handover_med) if rent_handover_med else 'Not available'}")
            rent_lines.append("")
            rent_lines.append("- Estimated Gross Yield")
            if yield_med is not None and yield_low is not None and yield_high is not None:
                rent_lines.append(f"  - Range: {yield_low:.1f}% – {yield_high:.1f}%")
            else:
                rent_lines.append("  - Not available")

            rent_bench: Optional[RentBenchmark] = trends.get("rent_benchmark")
            if rent_bench and rent_bench.median_annual_rent:
                rent_lines.append(f"")
                rent_lines.append(f"- Current Area Benchmark Rent (Median): {format_currency(rent_bench.median_annual_rent)}")

        # 5) Area market context (numbers only)
        area_lines = ["**Area Market Context | Independent Market Signals**"]
        if area_stats and area_stats.current_median_sqft:
            area_lines.append(f"- Current Area Median Price: AED {area_stats.current_median_sqft:,.0f} per sqft")
            area_lines.append("")
            area_lines.append("- Price Performance")
            if area_stats.price_change_12m is not None:
                area_lines.append(f"  - 12‑month change: {area_stats.price_change_12m:+.1f}%")
            if area_stats.price_change_36m is not None:
                area_lines.append(f"  - 36‑month change: {area_stats.price_change_36m:+.1f}%")
            area_lines.append("")
            area_lines.append(f"- Transaction Activity: Transactions (last 12 months): {area_stats.transaction_count_12m:,}")
            area_lines.append(f"- Upcoming Supply Pipeline: Scheduled new units: {area_stats.supply_pipeline:,}" if area_stats.supply_pipeline else "- Upcoming Supply Pipeline: Scheduled new units: Not available")
        else:
            area_lines.append("- Not available")

        # 6) Why the forecast landed here
        explain_lines = ["**Why the Forecast Landed Here**"]
        ti = prediction.trend_insights
        explain_lines.append("The model’s forecast is primarily influenced by:")
        if drivers:
            explain_lines.extend([f"- {d}" for d in drivers[:5]])
        else:
            explain_lines.append("- Not available")
        explain_lines.append("")
        if ti:
            if ti.price_change_3m is not None:
                explain_lines.append(f"- Recent segment price (last ~3m): {ti.price_change_3m:+.1f}%")
            if ti.price_change_12m is not None:
                explain_lines.append(f"- Recent segment price (last ~12m): {ti.price_change_12m:+.1f}%")
            if ti.area_transaction_volume_trend is not None:
                explain_lines.append(f"- Area liquidity (last 3 full months vs same period last year): {ti.area_transaction_volume_trend:+.1f}%")
        explain_lines.append("")
        explain_lines.append("Recent pricing and liquidity trends have been factored into the forecast to moderate assumptions and reflect current market conditions.")

        # Factual training coverage (use only verified counts/ranges from repo stats).
        # Source: Data/tft/build_stats.json + Data/cleaned/cleaning_stats.json
        coverage_lines = [
            "**Data Coverage & Model Credibility**",
            "- Sales transactions (raw): ~1.61M",
            "- Rental contracts (raw): ~9.52M",
            "- Projects/units/buildings/valuations used as supporting context: ~3,039 / ~2.34M / ~239K / ~87K",
            "- Model training table (monthly aggregates): 72,205 rows across 1,745 segment series (2003–2025)",
            "- The model evaluates each unit within its specific market segment, rather than relying on generic averages.",
        ]

        # 8) CTA
        cta = "**Press Generate Report to export the investor PDF.**"

        return "\n".join(
            notice_lines
            + [""] + property_lines
            + [""] + purchase_lines
            + [""] + value_lines
            + [""] + rent_lines
            + [""] + area_lines
            + [""] + explain_lines
            + [""] + coverage_lines
            + [""] + [cta]
        )
    
    # Get market price - prefer area_stats (current market data) over pf.current_sqft (TFT model)
    market_price_sqft = None
    if area_stats and area_stats.current_median_sqft:
        market_price_sqft = area_stats.current_median_sqft
    elif pf.current_sqft:
        market_price_sqft = pf.current_sqft
    
    # Price vs Market Analysis (if price and sqft provided)
    if price and market_price_sqft:
        # Use actual sqft if provided, otherwise estimate
        if pf.unit_sqft:
            actual_sqft = pf.unit_sqft
        else:
            # Estimate typical sqft for bedroom type
            bedroom_sizes = {"Studio": 450, "1BR": 750, "2BR": 1100, "3BR": 1600, "4BR": 2200, "5BR": 3000}
            actual_sqft = bedroom_sizes.get(entities.bedroom, 1100)
        
        user_price_sqft = price / actual_sqft
        
        price_section = "**Price Analysis:**\n"
        if pf.unit_sqft:
            price_section += f"- Your property: {format_currency(price)} for {actual_sqft:.0f} sqft\n"
        else:
            price_section += f"- Your property: {format_currency(price)} (sqft not provided, estimated {actual_sqft:.0f} sqft for {entities.bedroom})\n"
        price_section += f"- Your price per sqft: AED {user_price_sqft:,.0f}/sqft\n"
        price_section += f"- Area market average: AED {market_price_sqft:,.0f}/sqft\n"
        
        premium_pct = ((user_price_sqft / market_price_sqft) - 1) * 100
        price_section += f"- Difference: {premium_pct:+.0f}% {'above' if premium_pct > 0 else 'below'} market average\n"
        
        sections.append(price_section)
    
    # Building developer caveat - important to be transparent
    if entities.is_building_developer and entities.developer_data_caveat:
        sections.append(f"**Note:** {entities.developer_data_caveat}")
    
    # === MODEL TREND INSIGHTS (Primary analysis source) ===
    ti = prediction.trend_insights
    if ti:
        trend_section = "**Segment Trend (matched series):**\n"
        if prediction.matched_group_id:
            trend_section += f"- Matched group_id: {prediction.matched_group_id}\n"
        
        # Price momentum
        if ti.price_change_3m is not None:
            trend_section += f"- Price change (3 months, vs ~3 months ago): {ti.price_change_3m:+.1f}%\n"
        if ti.price_change_6m is not None:
            trend_section += f"- Price change (6 months, vs ~6 months ago): {ti.price_change_6m:+.1f}%\n"
        if ti.price_change_12m is not None:
            trend_section += f"- Price change (12 months, vs ~12 months ago): {ti.price_change_12m:+.1f}%\n"
        
        # Trend direction
        if ti.price_trend_direction:
            trend_section += f"- Trend direction: {ti.price_trend_direction.upper()}\n"

        # Clarity note: momentum vs forecast
        if pf.forecast_sqft_median is not None and pf.current_sqft is not None:
            trend_section += "- Note: trend direction is backward-looking momentum; the model forecast is forward-looking and can differ.\n"
        
        # Transaction volume
        if ti.transaction_volume_trend is not None:
            # Segment-level volume (matched series)
            if ti.transaction_volume_recent_avg is not None and ti.transaction_volume_year_ago_avg is not None:
                trend_section += (
                    f"- Segment transaction volume (avg/month, last 3 full months vs same 3 months 1y ago): "
                    f"{ti.transaction_volume_trend:+.1f}% "
                    f"(recent={ti.transaction_volume_recent_avg:.1f}, year_ago={ti.transaction_volume_year_ago_avg:.1f})\n"
                )
            else:
                trend_section += f"- Segment transaction volume (last 3 full months vs same 3 months 1y ago): {ti.transaction_volume_trend:+.1f}%\n"

        # Area-wide volume (sum across all segments in the DLD area)
        if ti.area_transaction_volume_trend is not None:
            if ti.area_transaction_volume_recent_avg is not None and ti.area_transaction_volume_year_ago_avg is not None:
                trend_section += (
                    f"- Area-wide transaction volume (avg/month, last 3 full months vs same 3 months 1y ago): "
                    f"{ti.area_transaction_volume_trend:+.1f}% "
                    f"(recent={ti.area_transaction_volume_recent_avg:.0f}, year_ago={ti.area_transaction_volume_year_ago_avg:.0f})\n"
                )
            else:
                trend_section += f"- Area-wide transaction volume (last 3 full months vs same 3 months 1y ago): {ti.area_transaction_volume_trend:+.1f}%\n"
        
        # Data recency
        if ti.data_as_of:
            trend_section += f"- Data as of: {ti.data_as_of}\n"
        
        sections.append(trend_section)
        
        # Supply dynamics
        if ti.supply_pipeline or ti.units_completing_6m or ti.active_projects:
            supply_section = "**Supply Dynamics:**\n"
            if ti.supply_pipeline:
                supply_section += f"- Supply pipeline: {ti.supply_pipeline:,} units\n"
            if ti.units_completing_6m:
                supply_section += f"- Units completing (last 6m, from historical schedule): {ti.units_completing_6m:,}\n"
            if ti.active_projects:
                supply_section += f"- Active projects in area: {ti.active_projects}\n"
            sections.append(supply_section)
        
        # Developer from model data (only when we have a real, developer-specific series)
        # If developer is Unknown / building developer, these fields reflect the registered/master entity and can mislead users.
        if (
            ti.developer_projects
            and entities.developer_arabic
            and entities.developer_arabic != "Unknown"
            and not entities.is_building_developer
        ):
            dev_model_section = "**Developer Execution (matched series entity):**\n"
            dev_model_section += f"- Total projects: {ti.developer_projects}\n"
            dev_model_section += f"- Completed projects: {ti.developer_completed or 0}\n"
            if ti.developer_avg_completion:
                dev_model_section += f"- Avg completion rate: {ti.developer_avg_completion:.0f}%\n"
            sections.append(dev_model_section)
        
        # Market context
        market_context = []
        if ti.current_eibor_3m is not None:
            eibor_note = f"Current 3M EIBOR: {ti.current_eibor_3m:.2f}%"
            if ti.eibor_change_6m is not None:
                eibor_note += f" ({ti.eibor_change_6m:+.2f}% vs 6m ago)"
            market_context.append(eibor_note)
        
        if ti.tourism_visitors:
            tourism_note = f"Monthly visitors: {ti.tourism_visitors:,}"
            if ti.tourism_change_12m is not None:
                tourism_note += f" ({ti.tourism_change_12m:+.1f}% YoY)"
            market_context.append(tourism_note)
        
        if market_context:
            sections.append("**Market Context:**\n- " + "\n- ".join(market_context))
    
    # Price forecast (gated by match quality to avoid wrong-series output)
    model_ok = (prediction.confidence or 0) >= MODEL_FORECAST_CONFIDENCE_THRESHOLD and prediction.match_type in MODEL_FORECAST_MATCH_TYPES
    if model_ok and (pf.current_sqft or pf.forecast_sqft_median):
        forecast_section = "**Capital Appreciation (model + extrapolated horizons, per sqft):**\n"
        if pf.current_sqft:
            forecast_section += f"- Current (matched series): AED {pf.current_sqft:.0f}/sqft\n"
        forecast_section += f"- 6-month forecast (model, median/P50): AED {pf.forecast_sqft_median:.0f}/sqft\n"
        forecast_section += f"- 6-month range (model, P10–P90): AED {pf.forecast_sqft_low:.0f} - {pf.forecast_sqft_high:.0f}/sqft\n"
        if pf.appreciation_percent is not None:
            forecast_section += f"- 6-month implied appreciation: {pf.appreciation_percent:.1f}%\n"

        # 12/24 month projections (derived)
        if pf.forecast_sqft_median_12m is not None and pf.forecast_sqft_low_12m is not None and pf.forecast_sqft_high_12m is not None:
            forecast_section += f"- 12-month forecast (extrapolated): AED {pf.forecast_sqft_median_12m:.0f}/sqft\n"
            forecast_section += f"- 12-month range (extrapolated, P10–P90): AED {pf.forecast_sqft_low_12m:.0f} - {pf.forecast_sqft_high_12m:.0f}/sqft\n"
            if pf.appreciation_percent_12m is not None:
                forecast_section += f"- 12-month implied appreciation: {pf.appreciation_percent_12m:.1f}%\n"
        if pf.forecast_sqft_median_24m is not None and pf.forecast_sqft_low_24m is not None and pf.forecast_sqft_high_24m is not None:
            forecast_section += f"- 24-month forecast (extrapolated): AED {pf.forecast_sqft_median_24m:.0f}/sqft\n"
            forecast_section += f"- 24-month range (extrapolated, P10–P90): AED {pf.forecast_sqft_low_24m:.0f} - {pf.forecast_sqft_high_24m:.0f}/sqft\n"
            if pf.appreciation_percent_24m is not None:
                forecast_section += f"- 24-month implied appreciation: {pf.appreciation_percent_24m:.1f}%\n"

        if pf.long_horizon_method:
            forecast_section += f"- Method note: 12/24m are derived by compounding the model-implied monthly growth from the 6-month model forecast ({pf.long_horizon_method})."
        sections.append(forecast_section)
    elif pf.forecast_sqft_median:
        sections.append("**Segment Forecast (model):**\n- Not shown due to low match quality (match_type/confidence below threshold).")
        
        # Total value section (if unit size provided)
        if pf.unit_sqft and pf.forecast_total_value_median:
            total_section = f"**Total Value Forecast (for {pf.unit_sqft:.0f} sqft unit):**\n"
            if pf.current_total_value:
                total_section += f"- Current estimated value: {format_currency(pf.current_total_value)}\n"
            total_section += f"- {pf.forecast_horizon_months}-month forecast (median): {format_currency(pf.forecast_total_value_median)}\n"
            total_section += f"- Forecast range: {format_currency(pf.forecast_total_value_low)} - {format_currency(pf.forecast_total_value_high)}"
            sections.append(total_section)
    
    # Rent forecast (gated)
    rf = prediction.rent_forecast
    if model_ok:
        rent_section = "**Rental Forecast (model):**\n"
        if rf.current_annual and rf.has_actual_rent:
            rent_section += f"- Current annual rent (model series): {format_currency(rf.current_annual)}\n"
        rent_section += f"- Forecast annual rent (median): {format_currency(rf.forecast_annual_median)}\n"
        rent_section += f"- Forecast range (P10–P90): {format_currency(rf.forecast_annual_low)} - {format_currency(rf.forecast_annual_high)}\n"
        if rf.estimated_yield_percent is not None:
            rent_section += f"- Implied yield (forecast median / price): {rf.estimated_yield_percent:.1f}%"
        if not rf.has_actual_rent:
            rent_section += "\n- Note: Model rent estimated from comparable properties"
        sections.append(rent_section)
    else:
        sections.append("**Rental Forecast (model):**\n- Not shown due to low match quality (match_type/confidence below threshold).")

    # Model-driven links (attribution; non-causal)
    if model_ok and prediction.model_attribution and prediction.model_attribution.top_drivers:
        # Translate to investor-friendly labels and hide internal-only features
        label_map = {
            "units_completing": "Scheduled completions (units)",
            "supply_units": "Supply pipeline (units)",
            "active_projects": "Active projects (count)",
            "units_registered": "Units registered (count)",
            "market_median_price": "Wider market median price (context)",
            "market_transactions": "Wider market transactions (context)",
            "transaction_count": "Segment transaction count (liquidity)",
            "govt_valuation_median": "Govt valuation median (context)",
            "valuation_count": "Valuation sample size (context)",
            "months_to_handover": "Months to handover (time-to-delivery)",
            "eibor_3m": "EIBOR 3M (rates)",
            "eibor_6m": "EIBOR 6M (rates)",
            "eibor_12m": "EIBOR 12M (rates)",
            "visitors_total": "Tourism visitors (demand proxy)",
        }
        allow = set(label_map.keys())
        drivers = [d for d in prediction.model_attribution.top_drivers if d.feature in allow]
        drivers_sorted = sorted(drivers, key=lambda d: d.importance, reverse=True)[:5]
        if drivers_sorted:
            lines = ["**Why the model is forecasting this (simple explanation):**"]
            lines.append("- This forecast is the model’s 6‑month view of the *next* 6 months. Recent 3/6/12m changes above are *past* momentum.")
            if pf.long_horizon_method:
                lines.append("- 12/24m numbers are extrapolations from the 6m model-implied monthly growth (not directly predicted by the checkpoint).")

            lines.append("- What the model is “looking at” right now (and how it moved over the last 6 months):")
            for d in drivers_sorted:
                label = label_map.get(d.feature, d.feature)
                if d.current_value is None:
                    lines.append(f"  - {label}: (current value not available) [weight {d.importance:.2f}]")
                    continue
                cur = float(d.current_value)
                if d.change_6m is None:
                    lines.append(f"  - {label}: current {cur:,.2f} [weight {d.importance:.2f}]")
                    continue
                ch = float(d.change_6m)
                prev = cur - ch
                pct = None
                if prev != 0:
                    pct = (ch / prev) * 100
                if pct is not None and abs(pct) < 1000:
                    lines.append(f"  - {label}: {pct:+.1f}% over 6m (current {cur:,.2f}) [weight {d.importance:.2f}]")
                else:
                    lines.append(f"  - {label}: Δ6m {ch:+,.2f} (current {cur:,.2f}) [weight {d.importance:.2f}]")

            if prediction.model_attribution.what_if_impacts:
                lines.append("- If these drivers move, how does the model’s 6‑month price forecast respond? (model sensitivity, holding other inputs constant):")
                for wi in prediction.model_attribution.what_if_impacts[:3]:
                    feat = wi.get("feature")
                    if not feat:
                        continue
                    label = label_map.get(feat, feat)
                    assumption = wi.get("assumption", "")
                    dp = wi.get("delta_percent_vs_baseline")
                    if dp is None:
                        continue
                    dp = float(dp)
                    if abs(dp) < 0.05:
                        lines.append(f"  - If {label} is {assumption} ⇒ change is <0.1% (very small in this case)")
                    else:
                        sign = "+" if dp >= 0 else ""
                        lines.append(f"  - If {label} is {assumption} ⇒ {sign}{dp:.2f}% change in the 6‑month forecast")

            lines.append("- Important: this is how the trained model behaves on this data (non‑causal).")
            sections.append("\n".join(lines))
    
    # Developer stats (already confidence-gated upstream in chat_service; keep factual)
    dev_stats: Optional[DeveloperStats] = trends.get("developer_stats")
    if dev_stats:
        dev_label = display_developer or entities.developer_english or "Developer"
        dev_section = f"**Developer Execution ({dev_label}):**\n"

        # Lead with what matters most for off-plan execution risk
        if dev_stats.avg_duration_months:
            dev_section += f"- Average time to complete (historical): {dev_stats.avg_duration_months:.0f} months\n"
        # Only show delay if we actually have signal (current lookup generation defaults to 0)
        if dev_stats.avg_delay_months is not None and dev_stats.avg_delay_months > 0:
            dev_section += f"- Average delay (historical): {dev_stats.avg_delay_months:.0f} months\n"

        # Only include the section if we have at least one meaningful execution metric.
        if "Average time to complete" in dev_section or "Average delay" in dev_section:
            sections.append(dev_section)
    
    # Area stats - already confidence-gated upstream in chat_service; keep factual and label as area-wide
    area_stats: Optional[AreaStats] = trends.get("area_stats")
    if area_stats:
        display_area = entities.area_display_name or entities.area_name
        area_section = f"**Area Market (area-wide, {display_area}):**\n"
        if area_stats.current_median_sqft:
            area_section += f"- Current median: AED {area_stats.current_median_sqft:.0f}/sqft\n"
        if area_stats.price_change_12m is not None:
            direction = "+" if area_stats.price_change_12m >= 0 else ""
            area_section += f"- 12-month change: {direction}{area_stats.price_change_12m:.1f}%\n"
        if area_stats.price_change_36m is not None:
            direction = "+" if area_stats.price_change_36m >= 0 else ""
            area_section += f"- 36-month change: {direction}{area_stats.price_change_36m:.1f}%\n"
        area_section += f"- Transactions (12m): {area_stats.transaction_count_12m:,}\n"
        
        # Supply/Demand Analysis
        if area_stats.supply_pipeline and area_stats.transaction_count_12m:
            area_section += f"- Upcoming supply: {area_stats.supply_pipeline:,} units\n"
            
            # Calculate absorption rate and years of supply
            monthly_absorption = area_stats.transaction_count_12m / 12
            if monthly_absorption > 0:
                years_of_supply = area_stats.supply_pipeline / (monthly_absorption * 12)
                area_section += f"- **Supply/Demand Ratio:** {years_of_supply:.1f} years of inventory at current absorption rate\n"
                
        elif area_stats.supply_pipeline:
            area_section += f"- Upcoming supply: {area_stats.supply_pipeline:,} units"
        
        sections.append(area_section)
    
    # Rent benchmark
    rent_bench: Optional[RentBenchmark] = trends.get("rent_benchmark")
    if rent_bench:
        rent_section = f"**Rent Benchmark ({rent_bench.area_name}, {rent_bench.bedrooms}BR):**\n"
        if rent_bench.median_annual_rent:
            rent_section += f"- Median annual rent: {format_currency(rent_bench.median_annual_rent)}\n"
        rent_section += f"- Contract count (12m): {rent_bench.rent_count:,}"
        sections.append(rent_section)

    # Lookup audit (factual)
    lookup_audit = trends.get("lookup_audit") or {}
    if lookup_audit:
        audit_lines = ["**Lookup Audit:**"]
        dev_a = lookup_audit.get("developer")
        if dev_a:
            audit_lines.append(
                f"- Developer lookup: input='{dev_a.get('input')}', mapped_to_arabic='{dev_a.get('mapped_to_arabic')}', matched='{dev_a.get('matched')}', method={dev_a.get('method')}, score={dev_a.get('score')}"
            )
        area_a = lookup_audit.get("area")
        if area_a:
            audit_lines.append(
                f"- Area lookup: input='{area_a.get('input')}', matched='{area_a.get('matched')}', method={area_a.get('method')}, score={area_a.get('score')}"
            )
        rent_a = lookup_audit.get("rent")
        if rent_a:
            audit_lines.append(
                f"- Rent lookup: area_input='{rent_a.get('area_input')}', matched_area='{rent_a.get('matched_area')}', area_method={rent_a.get('area_method')}, area_score={rent_a.get('area_score')}, bedroom_input='{rent_a.get('bedroom_input')}', matched_bedroom='{rent_a.get('matched_bedroom')}'"
            )
        sections.append("\n".join(audit_lines))
    
    # Confidence note
    confidence_note = f"\n**Data Confidence:**\n- Match type: {prediction.match_type}\n- Overall confidence: {prediction.confidence:.0f}%"
    sections.append(confidence_note)

    # Short glossary to reduce jargon confusion (keep factual)
    glossary = (
        "\n**Definitions (quick):**\n"
        "- Model / matched series: the historical time series for the matched segment (area × bedrooms × type × reg_type × developer where available).\n"
        "- P10–P90: 10th–90th percentile range of model outcomes (an uncertainty band, not a guarantee).\n"
        "- P50: the median forecast.\n"
        "- 3/6/12 month change: percent change between the latest value and the value ~3/6/12 months earlier in the same matched series.\n"
    )
    sections.append(glossary)
    
    return "\n\n".join(sections)


async def generate_response(
    query: str,
    entities: ValidatedEntities,
    prediction: TFTPrediction,
    trends: Dict[str, Any],
    price: Optional[float] = None,
    handover_months: Optional[int] = None,
    unit_sqft: Optional[float] = None
) -> GeneratedResponse:
    """
    Generate a natural language response based on predictions and trends.
    
    Args:
        query: Original user query
        entities: Validated entity names
        prediction: TFT model predictions
        trends: Trend lookup data
        price: Optional price from query
        
    Returns:
        GeneratedResponse with content, summary, and structured report data
    """
    settings = get_settings()
    
    # Build context
    context = build_context_prompt(query, entities, prediction, trends, price, handover_months, unit_sqft)

    # For Off-Plan, we return the deterministic investor snapshot directly.
    # This avoids "stuck thinking" delays and prevents the formatter model from changing the structure.
    if (entities.reg_type or "").lower() == "offplan":
        return GeneratedResponse(
            content=context,
            summary=f"Off‑Plan Snapshot — {entities.area_display_name or entities.area_name or 'area'}",
            report_data=_build_report_data(entities, prediction, trends, price, handover_months, unit_sqft),
        )

    # Load output contract (if available) to standardize output
    contract_text: Optional[str] = None
    try:
        # .../Properly/backend/app/services/response_generator.py -> parents[4] = Properly/
        project_root = Path(__file__).resolve().parents[4]
        contract_path = project_root / "Docs" / "frontend" / "LLM" / "OUTPUT_CONTRACT.md"
        if contract_path.exists():
            contract_text = contract_path.read_text(encoding="utf-8")
    except Exception:
        contract_text = None
    
    # If no OpenAI key, return formatted context directly
    if not settings.openai_api_key:
        return GeneratedResponse(
            content=context,
            summary=f"Analysis for {entities.area_name or 'property'} {entities.bedroom or ''} - {prediction.match_type} match",
            report_data=_build_report_data(entities, prediction, trends, price, handover_months, unit_sqft)
        )
    
    # Generate response with OpenAI
    client = OpenAI(api_key=settings.openai_api_key)
    
    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {
                    "role": "system",
                    "content": f"""You are a Dubai real estate data analyst. Your job is to REFORMAT the pre-calculated data provided.

CRITICAL RULES:
1. DO NOT recalculate any numbers - use EXACTLY the values provided in the data
2. The price per sqft has already been calculated correctly - just copy it
3. DO NOT assume typical unit sizes - the actual size is provided
4. Present facts only - no opinions or recommendations
5. If a section says a model forecast is NOT shown due to low match quality, do not output any model price/rent forecast values.
6. NEVER tell the user to buy/sell/hold. If the user asks for advice (e.g., "Should I buy?"), respond with factual metrics only and include this exact line in Data Quality + Coverage:
   "This report is factual and is not investment advice."

## Output Format (MUST FOLLOW)

Follow the Output Contract below as the report structure. If a value is not present in the provided data, write "Not available" and keep the label.

{contract_text or "Output Contract file not found. Use: Resolved Entities, Segment Definition, Market Prices (Current), Capital Appreciation (Model), Rental Levels + Rental Change, Yield, Supply, Developer Delivery Record, Data Quality + Coverage."}
"""
                },
                {
                    "role": "user",
                    "content": f"""Reformat this pre-calculated data into a clean report. DO NOT recalculate any values:

{context}"""
                }
            ],
            temperature=0.1,  # Very low temperature to avoid creativity
            max_tokens=1400
        )
        
        content = response.choices[0].message.content

        # Guardrail: if the formatter omits critical pre-computed sections, fall back to the deterministic context.
        # This prevents "missing explanation" issues when the LLM truncates or ignores parts of the provided data.
        try:
            if content:
                must_have = ["Resolved Entities", "Data Confidence"]
                for s in must_have:
                    if s not in content:
                        content = context
                        break
                # If we have model attribution available + forecasts shown, require the explanation header.
                if (
                    content != context
                    and prediction.model_attribution
                    and prediction.model_attribution.top_drivers
                    and prediction.match_type in MODEL_FORECAST_MATCH_TYPES
                    and (prediction.confidence or 0) >= MODEL_FORECAST_CONFIDENCE_THRESHOLD
                ):
                    if "Why the model forecast looks like this" not in content:
                        content = context
        except Exception:
            content = context
        
        # Generate short summary
        summary = f"{entities.area_name or 'Property'} {entities.bedroom or ''}"
        if entities.developer_english:
            summary = f"{entities.developer_english} - {summary}"
        if prediction.price_forecast.appreciation_percent:
            summary += f" | {prediction.price_forecast.appreciation_percent:.0f}% forecast"
        
        return GeneratedResponse(
            content=content,
            summary=summary,
            report_data=_build_report_data(entities, prediction, trends, price, handover_months, unit_sqft)
        )
        
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        # Fallback to formatted context
        return GeneratedResponse(
            content=context,
            summary=f"Analysis for {entities.area_name or 'property'}",
            report_data=_build_report_data(entities, prediction, trends, price, handover_months, unit_sqft)
        )


def _build_report_data(
    entities: ValidatedEntities,
    prediction: TFTPrediction,
    trends: Dict[str, Any],
    price: Optional[float],
    handover_months: Optional[int],
    unit_sqft: Optional[float]
) -> Dict[str, Any]:
    """Build structured report data for PDF generation."""
    
    # Calculate handover-extrapolated values (same logic as text response)
    pf = prediction.price_forecast
    h = int(handover_months) if handover_months else None
    base_h = int(pf.forecast_horizon_months or 0) or None
    # Anchor used for extrapolation growth-rate. Prefer model series "current", then independent area median,
    # and only as a last resort use purchase price per sqft.
    area_stats = trends.get("area_stats")
    area_cur_sqft = getattr(area_stats, "current_median_sqft", None) if area_stats else None
    cur_sqft = pf.current_sqft or area_cur_sqft
    if cur_sqft is None and price and unit_sqft and unit_sqft > 0:
        try:
            cur_sqft = float(price) / float(unit_sqft)
        except Exception:
            cur_sqft = None
    
    def _extrapolate(cur: Optional[float], end: Optional[float], base_horizon: Optional[int], target_horizon: int) -> Optional[float]:
        if cur is None or end is None or not base_horizon or base_horizon <= 0 or target_horizon <= 0:
            return end
        if cur <= 0:
            return end
        g = (end / cur) ** (1.0 / float(base_horizon)) - 1.0
        return cur * ((1.0 + g) ** float(target_horizon))
    
    # Target horizons: at handover and +12m post-handover
    target_h = h or (base_h or 12)
    target_h_plus_12 = target_h + 12
    
    # Per-sqft projections at handover (extrapolated if needed)
    handover_sqft_med = _extrapolate(cur_sqft, pf.forecast_sqft_median, base_h, target_h) or pf.forecast_sqft_median
    handover_sqft_low = _extrapolate(cur_sqft, pf.forecast_sqft_low, base_h, target_h) or pf.forecast_sqft_low
    handover_sqft_high = _extrapolate(cur_sqft, pf.forecast_sqft_high, base_h, target_h) or pf.forecast_sqft_high
    
    # Per-sqft projections at +12m post-handover
    post12_sqft_med = _extrapolate(cur_sqft, pf.forecast_sqft_median, base_h, target_h_plus_12)
    post12_sqft_low = _extrapolate(cur_sqft, pf.forecast_sqft_low, base_h, target_h_plus_12)
    post12_sqft_high = _extrapolate(cur_sqft, pf.forecast_sqft_high, base_h, target_h_plus_12)
    
    # Calculate total values for handover and +12m (for PDF)
    handover_total_med = round(handover_sqft_med * unit_sqft, 0) if handover_sqft_med and unit_sqft else None
    handover_total_low = round(handover_sqft_low * unit_sqft, 0) if handover_sqft_low and unit_sqft else None
    handover_total_high = round(handover_sqft_high * unit_sqft, 0) if handover_sqft_high and unit_sqft else None
    
    post12_total_med = round(post12_sqft_med * unit_sqft, 0) if post12_sqft_med and unit_sqft else None
    post12_total_low = round(post12_sqft_low * unit_sqft, 0) if post12_sqft_low and unit_sqft else None
    post12_total_high = round(post12_sqft_high * unit_sqft, 0) if post12_sqft_high and unit_sqft else None
    
    # Calculate uplift values
    uplift_handover = round(handover_total_med - price, 0) if handover_total_med and price else None
    uplift_handover_pct = round((handover_total_med - price) / price * 100, 1) if handover_total_med and price and price > 0 else None
    uplift_plus12m = round(post12_total_med - price, 0) if post12_total_med and price else None
    uplift_plus12m_pct = round((post12_total_med - price) / price * 100, 1) if post12_total_med and price and price > 0 else None
    
    # Calculate yield range (from rent forecast and price)
    rf = prediction.rent_forecast
    rent_low = rf.forecast_annual_low
    rent_med = rf.forecast_annual_median
    rent_high = rf.forecast_annual_high
    yield_med = rf.estimated_yield_percent
    yield_low = round((rent_low / price) * 100, 1) if rent_low and price and price > 0 else None
    yield_high = round((rent_high / price) * 100, 1) if rent_high and price and price > 0 else None
    
    # Get base price_forecast and add handover-specific values
    price_forecast_data = prediction.price_forecast.model_dump()
    price_forecast_data["handover_total_value_median"] = handover_total_med
    price_forecast_data["handover_total_value_low"] = handover_total_low
    price_forecast_data["handover_total_value_high"] = handover_total_high
    price_forecast_data["post12_total_value_median"] = post12_total_med
    price_forecast_data["post12_total_value_low"] = post12_total_low
    price_forecast_data["post12_total_value_high"] = post12_total_high
    price_forecast_data["handover_months_target"] = target_h
    
    return {
        "property": {
            "developer": entities.building_developer_name or entities.developer_english,
            "developer_english": entities.developer_english,
            "developer_arabic": entities.developer_arabic,
            "is_building_developer": entities.is_building_developer,
            "area": entities.area_name,
            "area_display": entities.area_display_name or entities.area_name,
            "bedroom": entities.bedroom,
            "property_type": entities.property_type,
            "reg_type": entities.reg_type,
            "price": price,
            "unit_sqft": unit_sqft,
            "handover_months": handover_months,
            "area_confidence": entities.area_confidence,
            "area_resolution_method": entities.area_resolution_method,
            "developer_confidence": entities.developer_confidence,
            "developer_resolution_method": entities.developer_resolution_method,
        },
        "price_forecast": price_forecast_data,
        "rent_forecast": prediction.rent_forecast.model_dump(),
        "trend_insights": prediction.trend_insights.model_dump() if prediction.trend_insights else None,
        "model_attribution": prediction.model_attribution.model_dump() if prediction.model_attribution else None,
        "developer_stats": trends.get("developer_stats").model_dump() if trends.get("developer_stats") else None,
        "area_stats": trends.get("area_stats").model_dump() if trends.get("area_stats") else None,
        "rent_benchmark": trends.get("rent_benchmark").model_dump() if trends.get("rent_benchmark") else None,
        "lookup_audit": trends.get("lookup_audit"),
        "match_info": {
            "type": prediction.match_type,
            "group_id": prediction.matched_group_id,
            "confidence": prediction.confidence,
        },
        "gating": {
            "area_confidence_threshold": AREA_CONFIDENCE_THRESHOLD,
            "developer_confidence_threshold": DEVELOPER_CONFIDENCE_THRESHOLD,
            "model_forecast_confidence_threshold": MODEL_FORECAST_CONFIDENCE_THRESHOLD,
            "model_forecast_match_types": sorted(list(MODEL_FORECAST_MATCH_TYPES)),
        },
        "caveats": {
            "developer_caveat": entities.developer_data_caveat if entities.is_building_developer else None,
        },
        # Pre-computed values for PDF (no calculations needed in frontend)
        "handover_total_value_median": handover_total_med,
        "handover_total_value_low": handover_total_low,
        "handover_total_value_high": handover_total_high,
        "plus12m_total_value_median": post12_total_med,
        "plus12m_total_value_low": post12_total_low,
        "plus12m_total_value_high": post12_total_high,
        "uplift_handover": uplift_handover,
        "uplift_handover_percent": uplift_handover_pct,
        "uplift_plus12m": uplift_plus12m,
        "uplift_plus12m_percent": uplift_plus12m_pct,
        "yield_low": yield_low,
        "yield_high": yield_high,
        "investor_calc_debug": {
            "base_horizon_months": base_h,
            "handover_months_target": target_h,
            "handover_months_plus_12_target": target_h_plus_12,
            "unit_sqft": unit_sqft,
            "purchase_price_aed": price,
            "current_sqft_anchor": cur_sqft,
            "current_sqft_source": (
                "price_forecast.current_sqft"
                if pf.current_sqft
                else ("area_stats.current_median_sqft" if area_cur_sqft is not None else ("purchase_price_aed / unit_sqft" if (price and unit_sqft) else "none"))
            ),
            "forecast_sqft_median_base_horizon": pf.forecast_sqft_median,
            "forecast_sqft_low_base_horizon": pf.forecast_sqft_low,
            "forecast_sqft_high_base_horizon": pf.forecast_sqft_high,
        },
    }

