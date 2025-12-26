"""
Report API Routes

Handles PDF report generation.
"""

from fastapi import APIRouter, Depends, HTTPException
from fastapi import UploadFile, File
from pydantic import BaseModel
from typing import Dict, Any, Optional
import json
import math

from ..core.auth import get_current_user, User
from ..core.supabase import get_supabase_admin

router = APIRouter()

def _extrapolate_sqft(cur: Optional[float], end: Optional[float], base_horizon: Optional[int], target_horizon: int) -> Optional[float]:
    """
    Extrapolate per-sqft forecast to a different horizon using a constant monthly growth rate
    implied by (cur -> end) over base_horizon months.
    """
    if cur is None or end is None or not base_horizon or base_horizon <= 0 or target_horizon <= 0:
        return end
    if cur <= 0:
        return end
    try:
        g = (end / cur) ** (1.0 / float(base_horizon)) - 1.0
        return cur * ((1.0 + g) ** float(target_horizon))
    except Exception:
        return end

def _ensure_investor_fields(report_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensure investor-facing totals exist on report_data so the PDF can be a pure rendering layer.
    This is intentionally server-side so we never compute these values in the PDF.
    """
    rd = report_data or {}
    prop = rd.get("property") or {}
    pf = rd.get("price_forecast") or {}
    rf = rd.get("rent_forecast") or {}
    area_stats = rd.get("area_stats") or {}

    # Required inputs
    unit_sqft = prop.get("unit_sqft") or pf.get("unit_sqft")
    price = prop.get("price")

    # Horizons
    base_h = pf.get("forecast_horizon_months") or 12
    try:
        base_h = int(base_h)
    except Exception:
        base_h = 12

    h = prop.get("handover_months")
    target_h = None
    try:
        target_h = int(h) if h is not None else None
    except Exception:
        target_h = None
    if not target_h or target_h <= 0:
        target_h = base_h or 12
    target_h_plus_12 = target_h + 12

    # Per-sqft anchors
    cur_sqft = pf.get("current_sqft")
    cur_sqft_source = "price_forecast.current_sqft"
    # If the model didn't provide a current_sqft anchor, prefer an independent market anchor (area median)
    # before ever falling back to the user's purchase price.
    if cur_sqft is None:
        area_cur = None
        try:
            # area_stats.current_median_sqft is already AED/sqft in our trend lookup output
            area_cur = area_stats.get("current_median_sqft") if isinstance(area_stats, dict) else None
        except Exception:
            area_cur = None
        if area_cur is not None:
            cur_sqft = area_cur
            cur_sqft_source = "area_stats.current_median_sqft"

    if (cur_sqft is None) and price and unit_sqft and unit_sqft > 0:
        # Last-resort anchor: purchase price per sqft. This can skew growth rates (premium/discount),
        # so we only use it if no better anchor exists.
        try:
            cur_sqft = float(price) / float(unit_sqft)
            cur_sqft_source = "purchase_price_aed / unit_sqft"
        except Exception:
            cur_sqft = None
            cur_sqft_source = "none"

    med_end = pf.get("forecast_sqft_median")
    low_end = pf.get("forecast_sqft_low")
    high_end = pf.get("forecast_sqft_high")

    # If we don't even have a forecast, we can't compute investor totals.
    if med_end is None or unit_sqft is None:
        return rd

    handover_sqft_med = _extrapolate_sqft(cur_sqft, med_end, base_h, target_h) or med_end
    handover_sqft_low = _extrapolate_sqft(cur_sqft, low_end, base_h, target_h) or low_end
    handover_sqft_high = _extrapolate_sqft(cur_sqft, high_end, base_h, target_h) or high_end

    post12_sqft_med = _extrapolate_sqft(cur_sqft, med_end, base_h, target_h_plus_12) or med_end
    post12_sqft_low = _extrapolate_sqft(cur_sqft, low_end, base_h, target_h_plus_12) or low_end
    post12_sqft_high = _extrapolate_sqft(cur_sqft, high_end, base_h, target_h_plus_12) or high_end

    # Totals (AED)
    try:
        u = float(unit_sqft)
    except Exception:
        return rd

    handover_total_med = round(handover_sqft_med * u, 0) if handover_sqft_med is not None else None
    handover_total_low = round(handover_sqft_low * u, 0) if handover_sqft_low is not None else None
    handover_total_high = round(handover_sqft_high * u, 0) if handover_sqft_high is not None else None

    plus12_total_med = round(post12_sqft_med * u, 0) if post12_sqft_med is not None else None
    plus12_total_low = round(post12_sqft_low * u, 0) if post12_sqft_low is not None else None
    plus12_total_high = round(post12_sqft_high * u, 0) if post12_sqft_high is not None else None

    # Uplift
    uplift_handover = None
    uplift_handover_pct = None
    uplift_plus12m = None
    uplift_plus12m_pct = None
    if price and price > 0:
        try:
            p = float(price)
            if handover_total_med is not None:
                uplift_handover = round(handover_total_med - p, 0)
                uplift_handover_pct = round(((handover_total_med - p) / p) * 100.0, 1)
            if plus12_total_med is not None:
                uplift_plus12m = round(plus12_total_med - p, 0)
                uplift_plus12m_pct = round(((plus12_total_med - p) / p) * 100.0, 1)
        except Exception:
            pass

    # Yield range (from rent forecast and price)
    yield_low = None
    yield_high = None
    if price and price > 0:
        try:
            p = float(price)
            r_low = rf.get("forecast_annual_low")
            r_high = rf.get("forecast_annual_high")
            if r_low is not None:
                yield_low = round((float(r_low) / p) * 100.0, 1)
            if r_high is not None:
                yield_high = round((float(r_high) / p) * 100.0, 1)
        except Exception:
            pass

    # Write onto top-level report_data (what the PDF reads)
    rd["handover_total_value_median"] = handover_total_med
    rd["handover_total_value_low"] = handover_total_low
    rd["handover_total_value_high"] = handover_total_high
    rd["plus12m_total_value_median"] = plus12_total_med
    rd["plus12m_total_value_low"] = plus12_total_low
    rd["plus12m_total_value_high"] = plus12_total_high
    rd["uplift_handover"] = uplift_handover
    rd["uplift_handover_percent"] = uplift_handover_pct
    rd["uplift_plus12m"] = uplift_plus12m
    rd["uplift_plus12m_percent"] = uplift_plus12m_pct
    rd["yield_low"] = yield_low
    rd["yield_high"] = yield_high

    # Debug payload to explain calculation inputs and prevent "same inputs, different outputs" confusion.
    # Safe to include in report_data (not rendered by default).
    rd["investor_calc_debug"] = {
        "base_horizon_months": base_h,
        "handover_months_target": target_h,
        "handover_months_plus_12_target": target_h_plus_12,
        "unit_sqft": unit_sqft,
        "purchase_price_aed": price,
        "current_sqft_anchor": cur_sqft,
        "current_sqft_source": cur_sqft_source,
        "forecast_sqft_median_base_horizon": med_end,
        "forecast_sqft_low_base_horizon": low_end,
        "forecast_sqft_high_base_horizon": high_end,
    }

    # Also stash on price_forecast for compatibility with older clients/debugging
    try:
        pf["handover_total_value_median"] = handover_total_med
        pf["handover_total_value_low"] = handover_total_low
        pf["handover_total_value_high"] = handover_total_high
        pf["post12_total_value_median"] = plus12_total_med
        pf["post12_total_value_low"] = plus12_total_low
        pf["post12_total_value_high"] = plus12_total_high
        pf["handover_months_target"] = target_h
        rd["price_forecast"] = pf
    except Exception:
        pass

    return rd


class ReportRequest(BaseModel):
    """Request body for report generation."""
    message_id: str
    agent_settings: Optional[Dict[str, Any]] = None
    unit_sqft: Optional[float] = None
    purchase_price_aed: Optional[float] = None


class ReportResponse(BaseModel):
    """Response with report data."""
    report_id: str
    report_data: Dict[str, Any]
    agent_settings: Dict[str, Any]


@router.post("/report/generate", response_model=ReportResponse)
async def generate_report(
    request: ReportRequest,
    user: User = Depends(get_current_user)
) -> ReportResponse:
    """
    Generate report data for PDF creation.
    
    The actual PDF is generated client-side using @react-pdf/renderer.
    This endpoint returns the structured data needed.
    
    Args:
        request: Report request with message_id
        user: Authenticated user
        
    Returns:
        Report data and agent settings for PDF generation
    """
    client = get_supabase_admin()
    
    # Get message with report data
    result = client.table("chat_messages")\
        .select("*, chat_sessions!inner(user_id)")\
        .eq("id", request.message_id)\
        .execute()
    
    if not result.data or len(result.data) == 0:
        raise HTTPException(status_code=404, detail="Message not found")
    
    message = result.data[0]
    
    # Verify message belongs to user's session
    if message["chat_sessions"]["user_id"] != user.id:
        raise HTTPException(status_code=403, detail="Access denied")
    
    if not message.get("report_data"):
        raise HTTPException(status_code=400, detail="No report data available for this message")
    
    # Get agent settings
    user_result = client.table("users")\
        .select("name, company_name, phone, logo_url, primary_color, secondary_color, report_footer, show_contact_info")\
        .eq("id", user.id)\
        .execute()
    
    agent_settings = request.agent_settings or {}
    if user_result.data and len(user_result.data) > 0:
        stored = user_result.data[0]
        # Use stored settings as base, override with request settings
        agent_settings = {
            "name": agent_settings.get("name", stored.get("name")),
            "company_name": agent_settings.get("company_name", stored.get("company_name")),
            "phone": agent_settings.get("phone", stored.get("phone")),
            "logo_url": agent_settings.get("logo_url", stored.get("logo_url")),
            "primary_color": agent_settings.get("primary_color", stored.get("primary_color", "#0f766e")),
            "secondary_color": agent_settings.get("secondary_color", stored.get("secondary_color", "#10b981")),
            "report_footer": agent_settings.get("report_footer", stored.get("report_footer")),
            "show_contact_info": agent_settings.get("show_contact_info", stored.get("show_contact_info", True)),
        }
    
    # Reuse existing report record for this message (overwrite behavior)
    report_id = ""
    existing = client.table("reports")\
        .select("id")\
        .eq("message_id", request.message_id)\
        .eq("user_id", user.id)\
        .execute()
    if existing.data and len(existing.data) > 0:
        report_id = existing.data[0]["id"]
    else:
        report_result = client.table("reports").insert({
            "message_id": request.message_id,
            "user_id": user.id,
        }).execute()
        report_id = report_result.data[0]["id"] if report_result.data else ""
    
    # Hydrate inputs (if provided) and ensure investor fields exist for PDF rendering
    rd = message["report_data"]
    if not isinstance(rd, dict):
        try:
            rd = json.loads(rd)
        except Exception:
            rd = {}

    prop = rd.get("property") or {}
    pf = rd.get("price_forecast") or {}

    if request.unit_sqft is not None:
        prop["unit_sqft"] = request.unit_sqft
        pf["unit_sqft"] = request.unit_sqft
    if request.purchase_price_aed is not None:
        prop["price"] = request.purchase_price_aed

    rd["property"] = prop
    rd["price_forecast"] = pf

    rd = _ensure_investor_fields(rd)

    # Persist hydrated report_data back onto the message so future PDFs (and history) are consistent.
    try:
        client.table("chat_messages").update({"report_data": rd}).eq("id", request.message_id).execute()
    except Exception:
        pass

    return ReportResponse(
        report_id=report_id,
        report_data=rd,
        agent_settings=agent_settings
    )


@router.get("/report/{report_id}")
async def get_report(
    report_id: str,
    user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get existing report data.
    
    Args:
        report_id: Report UUID
        user: Authenticated user
        
    Returns:
        Report data
    """
    client = get_supabase_admin()
    
    result = client.table("reports")\
        .select("*, chat_messages(*)")\
        .eq("id", report_id)\
        .eq("user_id", user.id)\
        .single()\
        .execute()
    
    if not result.data:
        raise HTTPException(status_code=404, detail="Report not found")
    
    report = result.data
    message = report.get("chat_messages", {})
    
    return {
        "report_id": report["id"],
        "created_at": report["created_at"],
        "pdf_url": report.get("pdf_url"),
        "report_data": message.get("report_data", {}),
    }


@router.post("/report/{report_id}/pdf-url")
async def save_pdf_url(
    report_id: str,
    pdf_url: str,
    user: User = Depends(get_current_user)
) -> Dict[str, str]:
    """
    Save PDF URL after client-side generation and upload.
    
    Args:
        report_id: Report UUID
        pdf_url: URL of uploaded PDF
        user: Authenticated user
        
    Returns:
        Success message
    """
    client = get_supabase_admin()
    
    # Verify report belongs to user
    result = client.table("reports")\
        .select("id")\
        .eq("id", report_id)\
        .eq("user_id", user.id)\
        .single()\
        .execute()
    
    if not result.data:
        raise HTTPException(status_code=404, detail="Report not found")
    
    # Update PDF URL
    client.table("reports")\
        .update({"pdf_url": pdf_url})\
        .eq("id", report_id)\
        .execute()
    
    return {"status": "updated", "pdf_url": pdf_url}


@router.post("/report/{report_id}/upload")
async def upload_report_pdf(
    report_id: str,
    file: UploadFile = File(...),
    user: User = Depends(get_current_user)
) -> Dict[str, str]:
    """
    Upload a generated PDF to Supabase Storage and persist the storage path on the report.

    Overwrite behavior: uploading again for the same report_id replaces the existing file.
    """
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Invalid file type. Only application/pdf is allowed.")

    client = get_supabase_admin()

    # Verify report belongs to user and fetch message_id
    result = client.table("reports")\
        .select("id, message_id")\
        .eq("id", report_id)\
        .eq("user_id", user.id)\
        .single()\
        .execute()
    if not result.data:
        raise HTTPException(status_code=404, detail="Report not found")

    message_id = result.data.get("message_id")
    if not message_id:
        raise HTTPException(status_code=400, detail="Report is missing message_id")

    # Read file content
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty file")

    # Store under stable path (overwrite)
    path = f"{user.id}/{message_id}.pdf"
    try:
        client.storage.from_("reports").upload(
            path,
            content,
            {
                "content-type": "application/pdf",
                "x-upsert": "true",
                "cache-control": "3600",
            },
        )
    except Exception:
        # Some storage configurations require update() for overwrite; try PUT as fallback.
        try:
            client.storage.from_("reports").update(
                path,
                content,
                {
                    "content-type": "application/pdf",
                    "x-upsert": "true",
                    "cache-control": "3600",
                },
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

    # Persist storage path (bucket is private; download uses signed URLs)
    client.table("reports")\
        .update({"pdf_storage_path": path, "pdf_url": None})\
        .eq("id", report_id)\
        .eq("user_id", user.id)\
        .execute()

    return {"status": "uploaded", "pdf_storage_path": path}


@router.get("/report/{report_id}/download-url")
async def get_report_download_url(
    report_id: str,
    user: User = Depends(get_current_user)
) -> Dict[str, str]:
    """
    Return a short-lived signed URL to download a report PDF from private storage.
    """
    client = get_supabase_admin()

    # Verify ownership and get storage path
    result = client.table("reports")\
        .select("id, pdf_storage_path")\
        .eq("id", report_id)\
        .eq("user_id", user.id)\
        .single()\
        .execute()

    if not result.data:
        raise HTTPException(status_code=404, detail="Report not found")

    storage_path = result.data.get("pdf_storage_path")
    if not storage_path:
        raise HTTPException(status_code=404, detail="No PDF saved for this report")

    try:
        signed = client.storage.from_("reports").create_signed_url(storage_path, 600)  # 10 minutes
        signed_url = signed.get("signedURL") or signed.get("signedUrl")
        if not signed_url:
            raise Exception("Missing signed URL")
        return {"signed_url": signed_url}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create signed URL: {str(e)}")

