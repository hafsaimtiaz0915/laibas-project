#!/usr/bin/env python3
"""
Diagnostics for median_rent / median_rent_sqft missingness in TFT V2 training data.

This script is intentionally NON-MUTATING (no imputation). It:
- Quantifies NaN/Inf rates overall and by group_id
- Outputs top group_ids contributing most missing rent values
- Checks correlation with rent_count==0, transaction_count==0, property_type/bedroom/reg_type, and time ranges
- Confirms whether NaNs are parse artifacts vs true missing values
- Confirms whether gaps are missing rows vs missing values within existing rows
- Verifies whether missing keys are absent in raw rent contracts (vs a pipeline join/mapping issue)

Usage:
  python3 scripts/diagnose_rent_missingness_v2.py \
    --tft_csv Data/tft/latest/tft_training_data_v2.csv \
    --out_dir Data/tft/latest/diagnostics_rent_missingness_v2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


def _safe_int_str(x: object) -> str | None:
    try:
        if x is None:
            return None
        if isinstance(x, float) and np.isnan(x):
            return None
        s = str(x).strip()
        if not s or s.lower() in {"nan", "none", "null"}:
            return None
        return str(int(float(s)))
    except Exception:
        return None


def build_area_name_to_id_map_from_transactions(transactions_csv: Path, chunk_size: int = 700_000) -> Dict[str, str]:
    counts: Dict[Tuple[str, str], int] = {}
    for chunk in pd.read_csv(transactions_csv, chunksize=chunk_size, usecols=["area_name_en", "area_id"], low_memory=False):
        chunk["area_name_en"] = chunk["area_name_en"].fillna("").astype(str).str.strip()
        chunk["area_id"] = chunk["area_id"].apply(_safe_int_str)
        chunk = chunk[(chunk["area_name_en"] != "") & chunk["area_id"].notna()]
        vc = chunk.groupby(["area_name_en", "area_id"]).size()
        for (name, aid), n in vc.items():
            counts[(name, aid)] = counts.get((name, aid), 0) + int(n)
    best: Dict[str, Tuple[str, int]] = {}
    for (name, aid), n in counts.items():
        cur = best.get(name)
        if cur is None or n > cur[1]:
            best[name] = (aid, n)
    return {name: aid for name, (aid, _) in best.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tft_csv", default="Data/tft/latest/tft_training_data_v2.csv")
    ap.add_argument("--rents_csv", default="Data/cleaned/Rent_Contracts_Cleaned.csv")
    ap.add_argument("--transactions_csv", default="Data/cleaned/Transactions_Cleaned.csv")
    ap.add_argument("--out_dir", default="Data/tft/latest/diagnostics_rent_missingness_v2")
    args = ap.parse_args()

    tft_csv = Path(args.tft_csv)
    rents_csv = Path(args.rents_csv)
    tx_csv = Path(args.transactions_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(tft_csv, low_memory=False)
    N = len(df)

    # numeric coercion + Inf checks
    overall = {"rows": int(N)}
    for c in ["median_rent", "median_rent_sqft"]:
        if c not in df.columns:
            overall[c] = {"present": False}
            continue
        raw = df[c]
        num = pd.to_numeric(raw, errors="coerce")
        overall[c] = {
            "present": True,
            "raw_dtype": str(raw.dtype),
            "nan_rate": float(num.isna().mean()) if N else 0.0,
            "inf_rate": float(np.isinf(num.to_numpy()).mean()) if N else 0.0,
            "min": float(np.nanmin(num.to_numpy())) if N else None,
            "max": float(np.nanmax(num.to_numpy())) if N else None,
        }

    # missing flags
    miss_rent = df["median_rent"].isna() if "median_rent" in df.columns else pd.Series(False, index=df.index)
    miss_sqft = df["median_rent_sqft"].isna() if "median_rent_sqft" in df.columns else pd.Series(False, index=df.index)

    # correlation with rent_count / transaction_count
    if "rent_count" in df.columns:
        rc = pd.to_numeric(df["rent_count"], errors="coerce").fillna(0).astype(int)
        overall["crosstab_median_rent_missing_vs_rent_count_zero"] = pd.crosstab(
            miss_rent, rc == 0, rownames=["median_rent_missing"], colnames=["rent_count==0"]
        ).to_dict()
    if "transaction_count" in df.columns:
        tc = pd.to_numeric(df["transaction_count"], errors="coerce").fillna(0).astype(int)
        overall["crosstab_median_rent_missing_vs_transaction_count_zero"] = pd.crosstab(
            miss_rent, tc == 0, rownames=["median_rent_missing"], colnames=["transaction_count==0"]
        ).to_dict()

    # by group_id + top50
    if "group_id" in df.columns:
        grp = df.groupby("group_id", dropna=False).agg(
            n=("group_id", "size"),
            miss_rent=("median_rent", lambda s: int(s.isna().sum())) if "median_rent" in df.columns else ("group_id", "size"),
            miss_sqft=("median_rent_sqft", lambda s: int(s.isna().sum())) if "median_rent_sqft" in df.columns else ("group_id", "size"),
            rent_count_zero=("rent_count", lambda s: int((pd.to_numeric(s, errors="coerce").fillna(0).astype(int) == 0).sum()))
            if "rent_count" in df.columns
            else ("group_id", "size"),
        )
        grp["miss_rent_rate"] = grp["miss_rent"] / grp["n"]
        grp["miss_sqft_rate"] = grp["miss_sqft"] / grp["n"]
        top50 = grp.sort_values(["miss_rent", "n"], ascending=[False, False]).head(50).reset_index()
        top50_path = out_dir / "top50_group_ids_missing_median_rent.csv"
        top50.to_csv(top50_path, index=False)
        overall["top50_group_ids_missing_median_rent_csv"] = str(top50_path)

    # slices by categorical
    slices = {}
    for col in ["property_type", "bedroom", "reg_type"]:
        if col in df.columns and "median_rent" in df.columns:
            s = (
                df.groupby(col, dropna=False)
                .agg(n=("median_rent", "size"), miss_rate=("median_rent", lambda x: float(x.isna().mean())))
                .sort_values(["miss_rate", "n"], ascending=[False, False])
            )
            slices[f"missing_rate_by_{col}"] = s.reset_index().to_dict(orient="records")

    # slices by time ranges
    if "year_month" in df.columns and "median_rent" in df.columns:
        ym = (
            df.groupby("year_month")
            .agg(n=("median_rent", "size"), miss_rate=("median_rent", lambda x: float(x.isna().mean())))
            .sort_values("miss_rate", ascending=False)
        )
        slices["worst_15_months_by_missing_rate"] = ym.head(15).reset_index().to_dict(orient="records")
        slices["best_15_months_by_missing_rate"] = ym.tail(15).sort_values("miss_rate").reset_index().to_dict(orient="records")

    if "time_idx" in df.columns and "median_rent" in df.columns:
        ti = df.groupby("time_idx").agg(n=("median_rent", "size"), miss_rate=("median_rent", lambda x: float(x.isna().mean())))
        slices["time_idx_missing_rate_head_15"] = ti.head(15).reset_index().to_dict(orient="records")
        slices["time_idx_missing_rate_tail_15"] = ti.tail(15).reset_index().to_dict(orient="records")

    # parsing artifacts: check whether any non-numeric tokens exist (should be 0 since dtype float in CSV)
    parse = {}
    for c in ["median_rent", "median_rent_sqft"]:
        if c not in df.columns:
            continue
        s = pd.read_csv(tft_csv, usecols=[c], dtype=str, low_memory=False)[c].fillna("").astype(str).str.strip()
        nonempty = s != ""
        num = pd.to_numeric(s.replace({"": np.nan}), errors="coerce")
        bad = nonempty & num.isna()
        parse[c] = {
            "nonempty_bad_parse_rate": float((bad.sum() / max(int(nonempty.sum()), 1))),
            "examples_bad_parse": s.loc[bad].head(25).tolist(),
        }

    # missing rows vs missing values: check for time_idx gaps (should be 0)
    gaps = {"groups_with_time_idx_gaps": None}
    if {"group_id", "time_idx"} <= set(df.columns):
        bad = 0
        for _, g in df.groupby("group_id"):
            t = pd.to_numeric(g["time_idx"], errors="coerce").dropna().astype(int).tolist()
            if not t:
                continue
            if len(t) != (max(t) + 1):
                bad += 1
        gaps["groups_with_time_idx_gaps"] = int(bad)

    # Verify whether missing keys exist in raw rents (absence vs join artifact)
    key_check = {"missing_rows_with_key_present_in_raw_rents": None, "missing_rows_with_key_absent_in_raw_rents": None}
    if {"year_month", "area_id", "bedroom", "median_rent"} <= set(df.columns) and rents_csv.exists() and tx_csv.exists():
        area_map = build_area_name_to_id_map_from_transactions(tx_csv)
        # build raw rent key set (fast-ish)
        keys = set()
        usecols = ["contract_start_date_parsed", "annual_amount", "area_name_en", "bedrooms", "property_usage_en"]
        for chunk in pd.read_csv(rents_csv, chunksize=700_000, usecols=usecols, low_memory=False):
            chunk = chunk[chunk["property_usage_en"] == "Residential"]
            dt = pd.to_datetime(chunk["contract_start_date_parsed"], errors="coerce")
            chunk = chunk[dt.notna()].copy()
            chunk["year_month"] = dt[dt.notna()].dt.strftime("%Y-%m").values
            chunk["area_name_en"] = chunk["area_name_en"].fillna("").astype(str).str.strip()
            chunk["area_id"] = chunk["area_name_en"].map(area_map).apply(_safe_int_str)
            chunk = chunk[chunk["area_id"].notna()]
            amt = pd.to_numeric(chunk["annual_amount"], errors="coerce")
            chunk = chunk[amt.notna() & (amt > 0)]
            br = chunk["bedrooms"].astype(str).str.strip().tolist()
            for ym, aid, b in zip(chunk["year_month"].tolist(), chunk["area_id"].tolist(), br):
                keys.add((ym, aid, b))
        miss = df["median_rent"].isna()
        exists = []
        for ym, aid, br in zip(df["year_month"].astype(str), df["area_id"].astype(str), df["bedroom"].astype(str)):
            exists.append((ym, aid, br) in keys)
        exists = pd.Series(exists, index=df.index)
        key_check["missing_rows_with_key_present_in_raw_rents"] = int((miss & exists).sum())
        key_check["missing_rows_with_key_absent_in_raw_rents"] = int((miss & ~exists).sum())

    report = {
        "tft_csv": str(tft_csv),
        "overall": overall,
        "slices": slices,
        "parse_artifacts": parse,
        "row_gaps": gaps,
        "raw_rents_key_presence_check": key_check,
    }
    (out_dir / "rent_missingness_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report["overall"], indent=2))
    print(f"Wrote diagnostics to: {out_dir}")


if __name__ == "__main__":
    main()


