"""
Smoke test for the TFT checkpoint loading + inference.

Runs inference via the backend service code (no server required).

Usage (from repo root):
  backend/venv/bin/python scripts/smoke_test_tft_checkpoint.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path


async def main() -> None:
    # Import inside async main so this script fails fast if backend deps are missing.
    repo_root = Path(__file__).resolve().parents[1]
    backend_dir = repo_root / "backend"
    # backend is not a python package, but backend/app is. Add backend/ to sys.path so "app" is importable.
    sys.path.insert(0, str(backend_dir))

    from app.services.tft_inference import get_tft_service, parse_group_id, predict as tft_predict

    svc = get_tft_service()
    print("model_loaded:", bool(svc.is_loaded and svc.model is not None))
    print("tft_model_path:", svc.settings.tft_model_path)
    print("tft_data_path:", svc.settings.tft_data_path)
    print("groups_loaded:", len(svc.groups))

    if not (svc.is_loaded and svc.model is not None):
        raise RuntimeError("TFT model did not load. Check backend/models/tft_final.ckpt and backend deps.")
    if not svc.groups:
        raise RuntimeError("No groups loaded from training data. Check Data/tft/tft_training_data.csv path.")

    # Pick a deterministic group from the loaded training data and round-trip through predict()
    gid = svc.groups[0]
    c = parse_group_id(gid)
    if not c.get("area"):
        raise RuntimeError(f"Could not parse group_id: {gid}")

    print("sample_group_id:", gid)
    print("parsed_components:", c)

    pred = await tft_predict(
        area=c["area"],
        property_type=c["property_type"] or "Unit",
        bedroom=c["bedroom"] or "2BR",
        reg_type=c["reg_type"] or "OffPlan",
        developer=c["developer"] or "ALL_DEVELOPERS",
        price=None,
        handover_months=None,  # default should become model_max_horizon (checkpoint-limited) or 12
        unit_sqft=1000.0,
    )

    pf = pred.price_forecast
    rf = pred.rent_forecast

    print("match_type:", pred.match_type)
    print("confidence:", pred.confidence)
    print("forecast_horizon_months:", pf.forecast_horizon_months)
    print("current_sqft:", pf.current_sqft)
    print("forecast_sqft_median:", pf.forecast_sqft_median)
    print("forecast_sqft_range:", (pf.forecast_sqft_low, pf.forecast_sqft_high))
    print("appreciation_percent:", pf.appreciation_percent)
    print("appreciation_percent_12m:", pf.appreciation_percent_12m)
    print("appreciation_percent_24m:", pf.appreciation_percent_24m)
    print("rent_forecast_median:", rf.forecast_annual_median)

    if not pf.forecast_horizon_months or pf.forecast_horizon_months < 6:
        raise AssertionError("Forecast horizon looks wrong (expected >= 6, typically 12).")
    if pf.forecast_sqft_median is None:
        raise AssertionError("No median price forecast returned.")


if __name__ == "__main__":
    asyncio.run(main())


