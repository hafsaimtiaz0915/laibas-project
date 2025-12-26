#!/usr/bin/env python3
"""
Build TFT-compatible training data (V2) with:

- area_id as the hub key (no string joins in the modeling layer)
- developer_id as the legal truth for OffPlan (via Transactions.project_number -> Projects.project_number)
- developer_brand as a second layer compiled to IDs (developer_id -> brand)
- deterministic project deduplication
- transaction group filtering (Sales) derived from lookup (no hardcoded IDs)
- optional geography enrichment from Community.kml via Lkp_Areas.municipality_number
- macro regime enrichment from Residential_Sale_Index.csv

Outputs (separate from V1):
- Data/tft/tft_training_data_v2.csv
- Data/tft/build_stats_v2.json

Usage:
    python scripts/build_tft_data_v2.py
"""

from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from xml.etree import ElementTree as ET

from brand_resolver import BrandResolver


# -----------------------------
# Logging
# -----------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# -----------------------------
# Paths
# -----------------------------
CLEANED_DIR = Path("Data/cleaned")
RAW_NEW_DIR = Path("Data/raw_data/new_raw_data")
LOOKUPS_DIR = Path("Data/lookups")
OUTPUT_DIR = Path("Data/tft")  # legacy/default; overwritten per-run in main()
RUNS_DIR = OUTPUT_DIR / "runs"
LATEST_DIR = OUTPUT_DIR / "latest"


# -----------------------------
# Constants (mostly copied from V1 to keep behaviour consistent)
# -----------------------------
BEDROOM_MAP = {
    "1 B/R": "1BR",
    "2 B/R": "2BR",
    "3 B/R": "3BR",
    "4 B/R": "4BR",
    "5 B/R": "5BR",
    "6 B/R": "6BR+",
    "7 B/R": "6BR+",
    "8 B/R": "6BR+",
    "9 B/R": "6BR+",
    "10 B/R": "6BR+",
    "Studio": "Studio",
    "PENTHOUSE": "Penthouse",
    "Single Room": "Room",
    # Exclude non-residential
    "Office": None,
    "Shop": None,
    "Store": None,
    "GYM": None,
}

VALID_PROPERTY_TYPES = ["Unit", "Villa"]

REG_TYPE_MAP = {
    "Off-Plan Properties": "OffPlan",
    "Existing Properties": "Ready",
}


# -----------------------------
# Config
# -----------------------------
@dataclass(frozen=True)
class V2Config:
    # If True, restrict pricing label to transaction groups whose name_en contains "sale".
    filter_sales_only: bool = True
    # Modeling grain: default series key should exclude developer to reduce sparsity (brand/umbrella are features).
    include_developer_in_group_id: bool = False
    # KML match threshold (warn below, hard-fail below hard_fail_kml_match_rate).
    warn_kml_match_rate: float = 0.95
    # KML coverage varies by administrative layer; do not hard-fail by default.
    hard_fail_kml_match_rate: float = 0.0

    # Brand resolution: must be 100% (hard fail if any unresolved after overrides).
    # No UNKNOWN/OTHER buckets are permitted.
    require_full_brand_coverage: bool = True

    # -----------------------------
    # Rent imputation policy (for multi-target training: price + rent)
    # -----------------------------
    # Strategy for handling missing rent values:
    #   "tiered" = ffill+bfill within group → area-level median → global median (recommended for multi-target)
    #   "drop"   = drop rows where rent is missing (loses price data too — use only for rent-only model)
    #   "none"   = leave NaNs as-is (training script must handle)
    rent_imputation_strategy: str = "tiered"
    # Minimum number of rent contracts required for a (year_month, area_id, bedroom) rent point to be considered "observed".
    # Below this threshold, rent is treated as missing and imputation is applied.
    min_rent_count: int = 1
    # Always write a rent-missingness audit artifact (JSON + CSV top contributors).
    write_rent_missingness_audit: bool = True
    # If True, also write the unfiltered dataset (before imputation/drop) for audit purposes.
    write_unfiltered_dataset_copy: bool = True

    # Curated brand universe (canonical public-facing labels)
    public_brands_path: str = "Data/lookups/public_brands.json"
    public_brand_aliases_path: str = "Data/lookups/public_brand_aliases.json"
    blocked_brand_labels_path: str = "Data/lookups/blocked_brand_labels.json"
    developer_id_overrides_path: str = "Data/lookups/brand_overrides_developer_id.csv"

    # Corporate suffix patterns that must never appear in developer_brand
    corporate_suffix_regex: str = r"(?i)\b(?:L\.L\.C|LLC|P\.?J\.?S\.?C|PJSC|LTD|LIMITED|LLP|S\.A\.R\.L|SARL|INC|FZCO|FZ\-LLC|FZE|O\.F)\b"

    # Top-50 reporting list (UI only)
    top50_2025_path: str = "Data/lookups/top50_developers_2025.json"

    # Umbrella mapping (explicit parent group)
    umbrella_map_path: str = "Data/lookups/umbrella_map.json"
    # SPV tripwire: if any top-N SPV candidates are not umbrella-mapped, emit alert (and optionally fail)
    spv_tripwire_top_n: int = 50
    # Hard-fail conditions
    spv_tripwire_fail_value_aed: float = 1_000_000_000.0  # AED 1B
    spv_tripwire_fail_top_n_by_value: int = 10
    spv_tripwire_fail_if_high_conf_unapplied: bool = True

    # High-impact SPV outliers detector (non-top50)
    spv_outliers_value_aed: float = 1_000_000_000.0
    spv_outliers_tx_12m: int = 500
    spv_outliers_units: int = 5000
    spv_outliers_concentration_share: float = 0.70
    spv_outliers_fail_value_aed: float = 5_000_000_000.0  # AED 5B

    # Baseline snapshot / delta reporting
    # NOTE: Baseline is promoted to Data/tft/latest only after a successful run.
    baseline_snapshot_path: str = "Data/tft/latest/baseline_snapshot_v2.json"

    # Closure pipeline thresholds for learned mapping
    learned_min_tx: int = 30
    learned_min_share: float = 0.85


def _norm_upper_spaces(x: object) -> str:
    s = _norm_ws(x).upper()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _boundary_phrase_pattern(phrase: str) -> Optional[re.Pattern]:
    p = _norm_upper_spaces(phrase)
    if not p:
        return None
    toks = [t for t in p.split() if t]
    if not toks:
        return None
    body = r"\s+".join([re.escape(t) for t in toks])
    return re.compile(rf"(?<![A-Z0-9_]){body}(?![A-Z0-9_])")


def _tokenize_preserve(x: object) -> List[str]:
    """
    Tokenize after punctuation->space normalization, preserving original token casing where possible.
    """
    s = _clean_str(x)
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    if not s:
        return []
    return s.split()


def normalize_legal_name_to_label(name: object, *, lang: str = "en") -> str:
    """
    Conservative label normalization:
    - Remove only corporate/legal suffix tokens (handles spaced variants like 'L L C')
    - Do NOT strip identity words like 'Properties', 'Development', 'Real Estate'
    - Preserve remaining tokens/casing
    """
    tokens = _tokenize_preserve(name)
    if not tokens:
        return ""

    # junk-only names
    junk = {"-", "—", "UNKNOWN", "N/A", "NA", "NONE", "NULL", "NAN", "0", "00"}
    if len(tokens) == 1 and tokens[0].strip().upper() in junk:
        return ""

    # corporate suffix token set (uppercased)
    suffix = {
        "LLC", "L.L.C", "L.L.C.", "LTD", "LIMITED", "LLP", "INC",
        # free-zone / legal suffixes
        "FZ", "FZC", "FZCO", "FZE", "FZ-LLC", "FZLLC", "OF",
        "PJSC", "P.J.S.C", "P.J.S.C.", "PSC", "P.S.C",
        "CO", "CO.", "COMPANY", "COMPANIES",
        # holdings
        "HOLDING", "HOLDINGS",
        # DMCC can appear as a legal suffix; treat as suffix for legal-normalized labels
        "DMCC",
    }
    # Arabic corporate sequences (very conservative)
    # After punctuation removal, ش.ذ.م.م often becomes tokens: ش ذ م م
    seq_drop = []
    if lang == "ar":
        seq_drop.extend([
            ["ش", "ذ", "م", "م"],
            ["ذ", "م", "م"],
            ["ش", "م", "ع"],
        ])

    up = [t.upper() for t in tokens]
    keep = [True] * len(tokens)

    # drop single-token suffixes
    for i, u in enumerate(up):
        if u in suffix:
            keep[i] = False

    # drop spaced variants: L L C / P J S C (post punctuation->space)
    def drop_seq(seq: List[str]):
        m = len(seq)
        for i in range(0, len(up) - m + 1):
            if all(up[i + j] == seq[j] for j in range(m)):
                for j in range(m):
                    keep[i + j] = False

    drop_seq(["L", "L", "C"])
    drop_seq(["P", "J", "S", "C"])
    drop_seq(["P", "S", "C"])
    # free zone variants split into tokens
    drop_seq(["F", "Z"])
    drop_seq(["F", "Z", "C"])
    drop_seq(["F", "Z", "C", "O"])
    drop_seq(["F", "Z", "E"])
    for seq in seq_drop:
        drop_seq([t.upper() for t in seq])

    out = [t for t, k in zip(tokens, keep) if k]
    s = " ".join(out).strip()
    # collapse junk again
    if s.upper() in junk:
        return ""
    return s


def _norm_label_any(x: object) -> str:
    """
    Unicode-safe label normalization used for cross-language matching:
    - whitespace collapse
    - punctuation -> space
    - collapse whitespace again
    """
    s = _norm_ws(x)
    s = re.sub(r"[^\w\s]", " ", s, flags=re.UNICODE)
    s = re.sub(r"\s+", " ", s, flags=re.UNICODE).strip()
    return s


def _label_is_authority_or_freezone(label: str) -> bool:
    """
    True only for authority/freezone-like entities that should stay as DEVELOPER_ID_*.
    Keep this narrow and explicit.
    """
    if not label:
        return False
    nl = _norm_label_any(label)
    toks = [t.upper() for t in _tokenize_preserve(label)]
    # explicit: DMCC and a small set of known authorities/freezones (EN + common abbreviations)
    if "DMCC" in toks:
        return True
    if any(t in {"DAFZA", "JAFZA", "TRAKHEES"} for t in toks):
        return True
    if nl in {
        _norm_label_any("DUBAI MULTI COMMODITIES CENTER"),
        _norm_label_any("DUBAI INTEGRATED ECONOMIC ZONES AUTHORITY"),
        _norm_label_any("DUBAI AIRPORT FREEZONE"),
        _norm_label_any("JEBEL ALI FREEZONE"),
        _norm_label_any("TRAKHEES"),
        # Arabic spellings / equivalents (best-effort; keep conservative)
        _norm_label_any("سلطة مركز دبي للسلع المتعددة"),
        _norm_label_any("سلطة دبي للمناطق الاقتصادية المتكاملة"),
    }:
        return True
    return False


def _label_is_project_or_area(
    label: str,
    *,
    suspicious_label_tokens: set[str],
    suspicious_area_name_norms: set[str],
    blocked_norms: set[str],
) -> bool:
    """
    Heuristic: label looks like a place/community/project/building rather than a developer brand.

    IMPORTANT: this is used to prevent project/area strings from surviving as training labels.
    """
    if not label:
        return False
    nl = _norm_label_any(label)
    if nl in blocked_norms:
        return True
    if suspicious_area_name_norms and nl in suspicious_area_name_norms:
        return True
    toks = [t.upper() for t in _tokenize_preserve(label)]
    if not toks:
        return False

    # Strong project/area tokens: one hit is enough
    strong = {
        # explicit project/community terms
        "PROJECT", "PROJECTS", "RESIDENTIAL", "VILLAGE", "SQUARE",
        # development phases / forms
        "PHASE", "ESTATE", "HILLS", "HARBOUR", "HARBOR", "CREEK", "LAGOON", "LAGOONS",
        "COURT", "COURTS", "RESIDENCE", "RESIDENCES", "TOWER", "TOWERS",
        "DISTRICT", "ISLAND", "ISLANDS", "PARK", "GARDENS", "HEIGHTS",
        "VIEW", "VIEWS",
        # single-token community names that often leak from legal entities
        "CITYWALK",
    }
    # Weak place tokens: require at least 2 weak hits to avoid false positives like "City Developments"
    weak = {"CITY", "BAY", "PALM", "MARINA", "DOWNTOWN", "JUMEIRAH"}

    hits_strong = any(t in strong for t in toks)
    if hits_strong:
        return True
    weak_hits = [t for t in toks if t in weak]
    if len(set(weak_hits)) >= 2:
        return True

    # Supplemental caller-supplied tokens (but never treat weak place tokens as single-hit triggers)
    extra = set([t.upper() for t in suspicious_label_tokens or set()]) - weak
    if extra and any(t in extra for t in toks):
        return True
    return False


CFG = V2Config()


# -----------------------------
# Utilities
# -----------------------------
def _clean_str(x: object) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    return str(x).strip()


def _norm_ws(x: object) -> str:
    """Whitespace-collapsed string for robust matching."""
    s = _clean_str(x)
    s = re.sub(r"\s+", " ", s)
    return s


def _safe_int_str(x: object) -> Optional[str]:
    """
    Convert an id-like value to a canonical digit-only string (no leading zeros),
    returning None if not parseable.
    """
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    # Handle numeric types safely (avoid "85.0" -> "850" bug)
    if isinstance(x, (int, np.integer)):
        s = str(int(x))
    elif isinstance(x, (float, np.floating)):
        if not np.isfinite(x):
            return None
        # Treat integer-like floats as ints
        if abs(x - round(x)) < 1e-9:
            s = str(int(round(x)))
        else:
            return None
    else:
        s = str(x).strip()
    if not s or s.lower() in {"nan", "none"}:
        return None
    # Common DLD export artefact: "85.0"
    m = re.match(r"^\s*(\d+)\s*(?:\.0+)?\s*$", s)
    if m:
        digits = m.group(1)
    else:
        # Extract FIRST digit run, do not concatenate all digits
        m2 = re.search(r"(\d+)", s)
        if not m2:
            return None
        digits = m2.group(1)
    # strip leading zeros (but keep single zero if all zeros)
    digits = digits.lstrip("0") or "0"
    return digits


def _parse_date_series(s: pd.Series) -> pd.Series:
    """Dates are in DD-MM-YYYY in the DLD extracts."""
    return pd.to_datetime(s, format="%d-%m-%Y", errors="coerce")


def _mode_or_nan(s: pd.Series) -> object:
    try:
        s2 = s.dropna()
        if len(s2) == 0:
            return np.nan
        m = s2.mode()
        return m.iat[0] if len(m) else s2.iat[0]
    except Exception:
        return np.nan


# -----------------------------
# Loaders
# -----------------------------
def load_transaction_groups_sales_ids() -> List[str]:
    """
    Determine which transaction group IDs represent "Sales".
    Never hardcode numeric IDs; derive from lookup content.
    """
    path = RAW_NEW_DIR / "Lkp_Transaction_Groups.csv"
    df = pd.read_csv(path, low_memory=False)
    # Normalize
    df["name_en"] = df["name_en"].astype(str).str.strip()
    df["group_id"] = df["group_id"].astype(str).str.strip()
    sales = df[df["name_en"].str.lower().str.contains("sale", na=False)]["group_id"].tolist()
    sales = sorted(set([gid for gid in sales if gid]))
    if not sales:
        raise RuntimeError("Could not determine Sales transaction groups from Lkp_Transaction_Groups.csv")
    logger.info(f"Sales groups (derived): {sales}")
    return sales


def load_developers_registry() -> pd.DataFrame:
    path = RAW_NEW_DIR / "Developers.csv"
    df = pd.read_csv(path, low_memory=False, usecols=["developer_id", "developer_name_en", "developer_name_ar"])
    df["developer_id"] = df["developer_id"].astype(str).str.strip()
    df["developer_name_en"] = df["developer_name_en"].astype(str).str.strip()
    df["developer_name_ar"] = df["developer_name_ar"].astype(str).str.strip()
    df = df[df["developer_id"].notna() & (df["developer_id"] != "")]
    df = df.drop_duplicates(subset=["developer_id"])
    return df


def load_projects_v2() -> pd.DataFrame:
    """
    Load Projects from new_raw_data, parse dates, and deterministically deduplicate by project_number.
    """
    usecols = [
        "project_id",
        "project_number",
        "developer_id",
        "developer_name",
        "master_developer_id",
        "master_developer_name",
        "project_start_date",
        "project_end_date",
        "completion_date",
        "cancellation_date",
        "project_status",
        "percent_completed",
        "area_id",
        "area_name_en",
        "master_project_en",
        "no_of_units",
        "no_of_buildings",
        "no_of_villas",
    ]
    path = RAW_NEW_DIR / "Projects.csv"
    df = pd.read_csv(path, low_memory=False, usecols=usecols)

    # Normalize ids
    for c in ["project_number", "developer_id", "master_developer_id", "area_id", "project_id"]:
        df[c] = df[c].apply(_safe_int_str)

    # Normalize strings
    for c in ["developer_name", "master_developer_name", "project_status", "area_name_en", "master_project_en"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()

    # Parse dates
    df["project_start_date_parsed"] = _parse_date_series(df["project_start_date"])
    df["project_end_date_parsed"] = _parse_date_series(df["project_end_date"])
    df["completion_date_parsed"] = _parse_date_series(df["completion_date"])
    df["cancellation_date_parsed"] = _parse_date_series(df["cancellation_date"])

    # Canonical handover date: completion else planned end
    df["handover_date"] = df["completion_date_parsed"].fillna(df["project_end_date_parsed"])

    # Deterministic dedup by project_number:
    # Prefer non-cancelled, then latest completion/end, then highest % complete.
    df["is_cancelled"] = df["cancellation_date_parsed"].notna()
    df["handover_sort"] = df["handover_date"]
    df["percent_completed_num"] = pd.to_numeric(df["percent_completed"], errors="coerce")

    df = df[df["project_number"].notna()]
    df = df.sort_values(
        by=["project_number", "is_cancelled", "handover_sort", "percent_completed_num"],
        ascending=[True, True, False, False],
        na_position="last",
    )
    before = len(df)
    df = df.drop_duplicates(subset=["project_number"], keep="first")
    after = len(df)
    if before != after:
        logger.info(f"Projects dedup: {before:,} -> {after:,} rows (by project_number)")

    # Precompute duration in months
    df["project_duration_months"] = (
        (df["handover_date"] - df["project_start_date_parsed"]).dt.days / 30.44
    ).clip(lower=0)

    return df


def compile_developer_brand_map_by_id(
    developers: pd.DataFrame,
    projects: pd.DataFrame,
    allowed_brands: Optional[List[str]] = None,
) -> Tuple[Dict[str, str], Dict[str, List[str]]]:
    """
    Compile developer_id -> brand from developer_brand_consolidation.json.

    The consolidation file is authored as Arabic registered entities. We resolve those names
    to developer_id using the Developers registry first, and then fall back to Projects
    (because Projects often contains the same registered strings).

    Returns:
      - dev_id_to_brand: developer_id -> brand
      - brand_debug_unmatched: brand -> list[str] of entities we couldn't resolve
    """
    path = LOOKUPS_DIR / "developer_brand_consolidation.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    brands_dict = data.get("brands") or {}
    # Hardening: also support legacy/root-level dict (if someone strips metadata wrapper)
    if not isinstance(brands_dict, dict):
        brands_dict = {}
    brands = brands_dict.items()
    allowed = set([b for b in (allowed_brands or []) if b])

    # Build fast lookup tables (normalized Arabic name -> developer_id)
    dev_ar = developers.copy()
    dev_ar["k"] = dev_ar["developer_name_ar"].map(_norm_ws)
    # Some Arabic registered names appear under multiple developer_ids in the registry.
    # For umbrella seeding, seed ALL matching developer_ids to avoid missing duplicates like Dubai Hills.
    ar_to_dev_ids: Dict[str, List[str]] = (
        dev_ar.dropna(subset=["k", "developer_id"])
        .assign(developer_id=lambda d: d["developer_id"].astype(str).str.strip())
        .groupby("k")["developer_id"]
        .apply(lambda s: sorted(set([x for x in s.tolist() if x])))
        .to_dict()
    )

    proj = projects.copy()
    proj["k_dev"] = proj["developer_name"].map(_norm_ws)
    proj["k_master"] = proj["master_developer_name"].map(_norm_ws)
    proj_dev_map = proj.dropna(subset=["k_dev", "developer_id"]).set_index("k_dev")["developer_id"].to_dict()
    proj_master_map = (
        proj.dropna(subset=["k_master", "master_developer_id"]).set_index("k_master")["master_developer_id"].to_dict()
    )

    dev_id_to_brand: Dict[str, str] = {}
    brand_debug_unmatched: Dict[str, List[str]] = {}

    for brand, info in brands:
        if allowed and str(brand) not in allowed:
            continue
        ents = info.get("registered_entities") or []
        unmatched: List[str] = []
        for e in ents:
            k = _norm_ws(e)
            if not k:
                continue

            # If multiple developer_ids share the same registered Arabic name, pick the first deterministically.
            dev_ids = ar_to_dev_ids.get(k) or []
            dev_id = dev_ids[0] if dev_ids else None
            if dev_id is None:
                # fallback via Projects registered strings (developer_name/master_developer_name)
                dev_id = proj_dev_map.get(k) or proj_master_map.get(k)

            if dev_id is None:
                unmatched.append(str(e))
                continue

            # If multiple brands map to same developer_id, last one wins (should not happen).
            dev_id_to_brand[str(dev_id)] = str(brand)

        if unmatched:
            brand_debug_unmatched[str(brand)] = unmatched

    # ------------------------------------------------------------
    # Expand mapping by matching brand aliases against Developers registry names.
    # This is an allowed mechanism: it maps developer_id -> curated brand using explicit alias tables,
    # without ever emitting legal entity names as brands.
    # ------------------------------------------------------------
    try:
        aliases_by_brand: Dict[str, List[str]] = {}
        for b, info in brands_dict.items():
            if allowed and str(b) not in allowed:
                continue
            aliases_by_brand.setdefault(str(b), [])
            aliases_by_brand[str(b)].extend(info.get("aliases") or [])
            # include Arabic registered entities as aliases too (helps match Arabic registry variants)
            aliases_by_brand[str(b)].extend(info.get("registered_entities") or [])
            # include brand label itself as an alias (boundary-safe)
            aliases_by_brand[str(b)].append(str(b))
        bd_path = LOOKUPS_DIR / "building_developers.json"
        if bd_path.exists():
            bd = json.loads(bd_path.read_text(encoding="utf-8"))
            raw = (bd.get("building_developers_without_own_data") or {})
            if isinstance(raw, dict):
                for b, info in raw.items():
                    if allowed and str(b) not in allowed:
                        continue
                    aliases_by_brand.setdefault(str(b), [])
                    aliases_by_brand[str(b)].extend((info or {}).get("aliases") or [])
                    aliases_by_brand[str(b)].append(str(b))

        def norm_text(x: object) -> str:
            s = _norm_ws(x).upper()
            s = re.sub(r"[^\w\s]", " ", s)
            s = re.sub(r"\s+", " ", s).strip()
            return s

        def alias_pattern(alias: str) -> Optional[Tuple[re.Pattern, int]]:
            a = norm_text(alias)
            if not a:
                return None
            toks = [t for t in a.split() if t]
            if not toks:
                return None
            # Avoid too-short aliases (high false-positive risk)
            if len(toks) == 1 and len(toks[0]) < 5:
                return None
            body = r"\s+".join([re.escape(t) for t in toks])
            pat = re.compile(rf"(?<![A-Z0-9_]){body}(?![A-Z0-9_])")
            score = 1000 * len(toks) + len(a)
            return pat, score

        dev_names = developers[["developer_id", "developer_name_en", "developer_name_ar"]].copy()
        dev_names["developer_id"] = dev_names["developer_id"].astype(str).str.strip()
        dev_names["en_norm"] = dev_names["developer_name_en"].map(norm_text)
        dev_names["ar_norm"] = dev_names["developer_name_ar"].map(norm_text)
        missing_ids = set(dev_names["developer_id"].tolist()) - set(dev_id_to_brand.keys())
        if missing_ids:
            sub = dev_names[dev_names["developer_id"].isin(missing_ids)].copy()
            best: Dict[str, Tuple[str, int, int]] = {}
            prec = list(aliases_by_brand.keys())
            prec_rank = {b: i for i, b in enumerate(prec)}

            def choose(cur: Optional[Tuple[str, int, int]], cand_brand: str, cand_score: int) -> Tuple[str, int, int]:
                pr = prec_rank.get(cand_brand, 10**9)
                if cur is None:
                    return (cand_brand, cand_score, pr)
                b0, s0, p0 = cur
                if cand_score > s0:
                    return (cand_brand, cand_score, pr)
                if cand_score < s0:
                    return cur
                if pr < p0:
                    return (cand_brand, cand_score, pr)
                return cur

            for b, als in aliases_by_brand.items():
                for a in als:
                    ap = alias_pattern(str(a))
                    if ap is None:
                        continue
                    pat, score = ap
                    m = sub["en_norm"].str.contains(pat, regex=True) | sub["ar_norm"].str.contains(pat, regex=True)
                    if not m.any():
                        continue
                    for dev_id in sub.loc[m, "developer_id"].tolist():
                        best[dev_id] = choose(best.get(dev_id), b, score)

            for dev_id, (b, _, _) in best.items():
                dev_id_to_brand[str(dev_id)] = str(b)
            logger.info(f"Expanded brand map via alias matching against Developers registry: +{len(best):,} developer_ids")
    except Exception as e:
        logger.warning(f"Could not expand brand map via aliases against Developers registry: {e}")

    # Apply explicit developer_id overrides (coverage closure)
    try:
        ovr_path = LOOKUPS_DIR / "brand_overrides_developer_id.csv"
        if ovr_path.exists():
            ovr = pd.read_csv(ovr_path, low_memory=False)
            if {"developer_id", "developer_brand"} <= set(ovr.columns):
                for _, r in ovr.iterrows():
                    did = _safe_int_str(r.get("developer_id"))
                    b = _norm_ws(r.get("developer_brand"))
                    if did and b:
                        dev_id_to_brand[str(did)] = str(b)
    except Exception as e:
        logger.warning(f"Could not apply developer_id overrides: {e}")

    logger.info(f"Compiled brand map for {len(dev_id_to_brand):,} developer_ids")
    if brand_debug_unmatched:
        logger.warning(
            "Some brand entities could not be resolved to developer_id. "
            "This is not fatal, but you should audit these strings.\n"
            + json.dumps({k: v[:10] for k, v in brand_debug_unmatched.items()}, ensure_ascii=False, indent=2)
        )
    return dev_id_to_brand, brand_debug_unmatched


def seed_umbrella_map_from_consolidation(
    developers: pd.DataFrame,
    projects: pd.DataFrame,
    consolidation_json: Path,
    out_path: Path,
    *,
    umbrella_label_aliases: Optional[Dict[str, str]] = None,
) -> Dict[str, object]:
    """
    Deterministically seed umbrella_map.json from developer_brand_consolidation.json registered_entities.
    Resolution channels:
      1) Developers registry (developer_name_ar exact after whitespace normalization)
      2) Projects developer_name/master_developer_name (exact after whitespace normalization)
    """
    umbrella_label_aliases = umbrella_label_aliases or {}
    data = json.loads(consolidation_json.read_text(encoding="utf-8"))
    brands_dict = data.get("brands") or {}
    if not isinstance(brands_dict, dict):
        brands_dict = {}

    dev_ar = developers.copy()
    dev_ar["k"] = dev_ar["developer_name_ar"].map(_norm_ws)
    ar_to_dev_ids: Dict[str, List[str]] = (
        dev_ar.dropna(subset=["k", "developer_id"])
        .assign(developer_id=lambda d: d["developer_id"].astype(str).str.strip())
        .groupby("k")["developer_id"]
        .apply(lambda s: sorted(set([x for x in s.tolist() if x])))
        .to_dict()
    )

    proj = projects.copy()
    proj["k_dev"] = proj["developer_name"].map(_norm_ws)
    proj["k_master"] = proj["master_developer_name"].map(_norm_ws)
    proj_dev_map = proj.dropna(subset=["k_dev", "developer_id"]).set_index("k_dev")["developer_id"].to_dict()
    proj_master_map = (
        proj.dropna(subset=["k_master", "master_developer_id"]).set_index("k_master")["master_developer_id"].to_dict()
    )

    out_map: Dict[str, str] = {}
    unmatched_entities: List[Dict[str, str]] = []
    matched_by_channel = {"Developers": 0, "Projects": 0}
    matched_by_umbrella: Dict[str, int] = {}
    total_entities = 0
    multi_match_entities: List[Dict[str, object]] = []

    for umbrella, info in brands_dict.items():
        ents = (info or {}).get("registered_entities") or []
        umbrella_label = str(umbrella_label_aliases.get(str(umbrella), str(umbrella))).strip()
        if not umbrella_label:
            continue
        matched_by_umbrella.setdefault(umbrella_label, 0)
        for e in ents:
            total_entities += 1
            k = _norm_ws(e)
            if not k:
                continue
            dev_ids = ar_to_dev_ids.get(k)
            channel = "Developers"
            if not dev_ids:
                dev_id_one = proj_dev_map.get(k) or proj_master_map.get(k)
                dev_ids = [str(dev_id_one)] if dev_id_one is not None else []
                channel = "Projects"
            dev_ids = [d for d in dev_ids if d and str(d).lower() not in {"nan", "none"}]
            if not dev_ids:
                unmatched_entities.append({"umbrella": umbrella_label, "entity": str(e), "channel": "UNMATCHED"})
                continue
            if len(dev_ids) > 1:
                multi_match_entities.append({"umbrella": umbrella_label, "entity": str(e), "developer_ids": dev_ids, "channel": channel})
            for dev_id in dev_ids:
                out_map[str(dev_id)] = umbrella_label
            matched_by_channel[channel] += 1
            matched_by_umbrella[umbrella_label] += 1

    # Merge into umbrella_map.json (never overwrite existing manual mappings)
    existing = {}
    if out_path.exists():
        try:
            existing = json.loads(out_path.read_text(encoding="utf-8"))
        except Exception:
            existing = {}
    existing_map = {}
    if isinstance(existing, dict):
        existing_map = dict(existing.get("map") or {})
    merged = dict(existing_map)
    # only fill missing keys from deterministic seed
    for did, umb in out_map.items():
        if str(did) not in merged:
            merged[str(did)] = str(umb)

    payload = {
        "_description": existing.get("_description", "Explicit umbrella mappings (developer_id -> umbrella label)."),
        "_version": existing.get("_version", "1.0"),
        "_last_updated": "2025-12-17",
        "map": dict(sorted(merged.items(), key=lambda kv: kv[0])),
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    # entity-level join metrics (not developer_id-level)
    matched_entity_count = int(total_entities - len(unmatched_entities))
    audit = {
        "total_registered_entities_in_consolidation": int(total_entities),
        "matched_entity_count": matched_entity_count,
        "unmatched_entity_count": int(len(unmatched_entities)),
        "seeded_developer_id_count": int(len(out_map)),
        "matched_by_channel": matched_by_channel,
        "matched_by_umbrella": dict(sorted(matched_by_umbrella.items(), key=lambda kv: kv[1], reverse=True)),
        "unmatched_entities_top": unmatched_entities[:50],
        "multi_match_entities_top": multi_match_entities[:25],
        "join_rate": {
            "matched_pct": float(matched_entity_count / max(total_entities, 1)),
            "unmatched_pct": float(len(unmatched_entities) / max(total_entities, 1)),
        },
        "merge_summary": {
            "existing_map_count": int(len(existing_map)),
            "merged_map_count": int(len(merged)),
            "new_keys_added_from_seed": int(max(len(merged) - len(existing_map), 0)),
        },
        "sample_seeded_rows": {
            # sample a few IDs per umbrella for quick sanity checks
            k: [did for did, umb in out_map.items() if umb == k][:10]
            for k in list(dict(sorted(matched_by_umbrella.items(), key=lambda kv: kv[1], reverse=True)).keys())[:10]
        },
    }
    audit_path = OUTPUT_DIR / "umbrella_seed_audit_v2.json"
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    return audit


def load_lkp_areas() -> pd.DataFrame:
    path = RAW_NEW_DIR / "Lkp_Areas.csv"
    # name_ar may be absent in some extracts; load if present.
    try:
        df = pd.read_csv(path, low_memory=False, usecols=["area_id", "name_en", "name_ar", "municipality_number"])
    except Exception:
        df = pd.read_csv(path, low_memory=False, usecols=["area_id", "name_en", "municipality_number"])
        df["name_ar"] = ""
    df["area_id"] = df["area_id"].apply(_safe_int_str)
    df["municipality_number"] = df["municipality_number"].apply(_safe_int_str)
    df["name_en"] = df["name_en"].astype(str).str.strip()
    df["name_ar"] = df["name_ar"].astype(str).str.strip()
    df = df.dropna(subset=["area_id"])
    df = df.drop_duplicates(subset=["area_id"])
    return df


def load_sale_index() -> pd.DataFrame:
    path = RAW_NEW_DIR / "Residential_Sale_Index.csv"
    df = pd.read_csv(path, low_memory=False)
    df["date"] = pd.to_datetime(df["first_date_of_month"], format="%d-%m-%Y", errors="coerce")
    df = df.dropna(subset=["date"])
    df["year_month"] = df["date"].dt.strftime("%Y-%m")
    # numeric
    for c in [
        "all_monthly_index",
        "flat_monthly_index",
        "villa_monthly_index",
        "all_monthly_price_index",
        "flat_monthly_price_index",
        "villa_monthly_price_index",
    ]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    keep = [
        "year_month",
        "all_monthly_index",
        "flat_monthly_index",
        "villa_monthly_index",
        "all_monthly_price_index",
        "flat_monthly_price_index",
        "villa_monthly_price_index",
    ]
    keep = [c for c in keep if c in df.columns]
    df = df[keep].drop_duplicates(subset=["year_month"]).sort_values("year_month")
    return df


# -----------------------------
# KML geo utilities
# -----------------------------
def _polygon_centroid_area(coords: List[Tuple[float, float]]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Compute centroid (lon, lat) and planar area using the shoelace formula on lon/lat degrees.
    This is an approximation suitable for features (not for precise geodesic area).
    """
    if len(coords) < 3:
        return None, None, None
    # ensure closed ring
    if coords[0] != coords[-1]:
        coords = coords + [coords[0]]
    x = np.array([p[0] for p in coords], dtype=float)
    y = np.array([p[1] for p in coords], dtype=float)
    a = 0.5 * np.sum(x[:-1] * y[1:] - x[1:] * y[:-1])
    if a == 0:
        return float(np.mean(x[:-1])), float(np.mean(y[:-1])), 0.0
    cx = (1.0 / (6.0 * a)) * np.sum((x[:-1] + x[1:]) * (x[:-1] * y[1:] - x[1:] * y[:-1]))
    cy = (1.0 / (6.0 * a)) * np.sum((y[:-1] + y[1:]) * (x[:-1] * y[1:] - x[1:] * y[:-1]))
    return float(cx), float(cy), float(abs(a))


def _norm_name_key(x: object) -> str:
    """
    Normalize place/area names for coarse joins:
    - uppercase
    - remove punctuation
    - collapse whitespace
    """
    s = _norm_ws(x).upper()
    # drop trailing code fragments like " - 614"
    s = re.sub(r"\s*-\s*\d+\s*$", "", s)
    s = re.sub(r"[^\w\s]", " ", s)  # keep letters/digits/underscore
    s = re.sub(r"\s+", " ", s).strip()
    return s


def build_area_geo_from_kml(lkp_areas: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """
    Build area_geo features keyed by area_id:
      - community_num (municipality/community number)
      - centroid_lon, centroid_lat
      - polygon_area_deg2 (approx)
    """
    kml_path = RAW_NEW_DIR / "Community.kml"
    if not kml_path.exists():
        logger.warning("Community.kml not found; skipping area_geo")
        return pd.DataFrame(columns=["area_id", "community_num", "centroid_lon", "centroid_lat", "polygon_area_deg2"]), {
            "kml_present": False
        }

    ns = {"kml": "http://www.opengis.net/kml/2.2"}
    tree = ET.parse(kml_path)
    root = tree.getroot()

    rows = []
    for pm in root.findall(".//kml:Placemark", ns):
        comm_num = None
        cname_e = None
        community_e = None
        for sd in pm.findall(".//kml:SimpleData", ns):
            if sd.attrib.get("name") == "COMM_NUM":
                comm_num = _safe_int_str(sd.text)
            elif sd.attrib.get("name") == "CNAME_E":
                cname_e = _clean_str(sd.text)
            elif sd.attrib.get("name") == "COMMUNITY_E":
                community_e = _clean_str(sd.text)
        # Prefer COMMUNITY_E (often includes "- 614"), else CNAME_E
        name_raw = community_e or cname_e
        name_norm = _norm_name_key(name_raw) if name_raw else ""
        if comm_num is None:
            continue
        coords_el = pm.find(".//kml:coordinates", ns)
        if coords_el is None or coords_el.text is None:
            continue
        coords_text = coords_el.text.strip()
        pts = []
        for token in coords_text.split():
            parts = token.split(",")
            if len(parts) < 2:
                continue
            try:
                lon = float(parts[0])
                lat = float(parts[1])
                pts.append((lon, lat))
            except Exception:
                continue
        cx, cy, area = _polygon_centroid_area(pts)
        rows.append(
            {
                "community_num": comm_num,
                "name_norm": name_norm,
                "centroid_lon": cx,
                "centroid_lat": cy,
                "polygon_area_deg2": area,
            }
        )

    geo = pd.DataFrame(rows)
    # dedup by community_num first
    geo = geo.drop_duplicates(subset=["community_num"])
    if geo.empty:
        logger.warning("Parsed 0 community polygons from KML; skipping area_geo")
        return pd.DataFrame(columns=["area_id", "community_num", "centroid_lon", "centroid_lat", "polygon_area_deg2"]), {
            "kml_present": True,
            "kml_polygons": 0,
        }

    # Join to area_id via municipality_number
    lkp = lkp_areas[["area_id", "municipality_number"]].copy()
    lkp = lkp.dropna(subset=["area_id", "municipality_number"])
    lkp = lkp.rename(columns={"municipality_number": "community_num"})
    merged_num = lkp.merge(geo, on="community_num", how="left")

    # Fallback join by normalized name for rows still missing geo
    # Build name index from geo (some names may be blank; drop blanks)
    geo_name = geo.copy()
    geo_name = geo_name[geo_name["name_norm"].astype(str).str.strip() != ""]
    geo_name = geo_name.drop_duplicates(subset=["name_norm"])
    lkp_name = lkp_areas[["area_id"]].copy()
    lkp_name["name_norm"] = lkp_areas["name_en"].map(_norm_name_key)

    merged = merged_num.merge(lkp_name, on="area_id", how="left", suffixes=("", "_lkp"))
    miss = merged["centroid_lon"].isna() & merged["name_norm_lkp"].notna() & (merged["name_norm_lkp"] != "")
    if miss.any() and not geo_name.empty:
        fill = merged.loc[miss, ["area_id", "name_norm_lkp"]].merge(
            geo_name[["name_norm", "centroid_lon", "centroid_lat", "polygon_area_deg2", "community_num"]],
            left_on="name_norm_lkp",
            right_on="name_norm",
            how="left",
        )
        # Apply fills
        idx = merged.index[miss]
        merged.loc[idx, "centroid_lon"] = fill["centroid_lon"].values
        merged.loc[idx, "centroid_lat"] = fill["centroid_lat"].values
        merged.loc[idx, "polygon_area_deg2"] = fill["polygon_area_deg2"].values
        # keep community_num from numeric join if present; else fill
        fill_comm = pd.Series(fill["community_num"].values, index=idx)
        merged.loc[idx, "community_num"] = merged.loc[idx, "community_num"].fillna(fill_comm)

    match_rate = float(merged["centroid_lon"].notna().mean()) if len(merged) else 0.0
    match_rate_num_only = float(merged_num["centroid_lon"].notna().mean()) if len(merged_num) else 0.0

    stats = {
        "kml_present": True,
        "kml_polygons": int(len(geo)),
        "areas_with_municipality_number": int(len(lkp)),
        "kml_match_rate_num_only": match_rate_num_only,
        "kml_match_rate_after_name_fallback": match_rate,
    }
    # Validate after fallback
    if match_rate < CFG.hard_fail_kml_match_rate:
        raise RuntimeError(
            f"KML match rate too low: {match_rate:.2%} "
            f"(hard fail threshold {CFG.hard_fail_kml_match_rate:.0%})"
        )
    if match_rate < CFG.warn_kml_match_rate:
        logger.warning(f"KML match rate below target: {match_rate:.2%} (warn threshold {CFG.warn_kml_match_rate:.0%})")
    else:
        logger.info(f"KML match rate: {match_rate:.2%}")

    out = merged[["area_id", "community_num", "centroid_lon", "centroid_lat", "polygon_area_deg2"]].copy()
    return out, stats


# -----------------------------
# Aggregations
# -----------------------------
def aggregate_transactions_v2(
    projects: pd.DataFrame,
    developers: pd.DataFrame,
    dev_id_to_brand: Dict[str, str],
    umbrella_by_id: Dict[str, str],
    sales_group_ids: List[str],
    brand_resolver: BrandResolver,
    safe_training_brands: Optional[set[str]] = None,
    id_brand_overrides: Optional[Dict[str, str]] = None,
    entity_owner_overrides: Optional[Dict[str, str]] = None,
    suspicious_label_tokens: Optional[set[str]] = None,
    suspicious_area_name_norms: Optional[set[str]] = None,
    blocked_brand_labels_norms: Optional[set[str]] = None,
    chunk_size: int = 200_000,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """
    Aggregate transactions to monthly by area_id/property_type/bedroom/reg_type/brand.
    Keep legal developer_id for audit.
    """
    tx_path = CLEANED_DIR / "Transactions_Cleaned.csv"
    if not tx_path.exists():
        raise FileNotFoundError(f"Missing cleaned transactions: {tx_path}")

    # project_number -> legal developer_id and timeline
    proj_info = projects[
        [
            "project_number",
            "developer_id",
            "master_developer_id",
            "project_start_date_parsed",
            "handover_date",
            "percent_completed_num",
            "project_duration_months",
        ]
    ].copy()
    proj_info = proj_info.dropna(subset=["project_number"])
    proj_info["project_number"] = proj_info["project_number"].astype(str)

    # master_project_en -> master_developer_id fallback (for missing project_number)
    mp = projects[["master_project_en", "master_developer_id"]].dropna().copy()
    mp["master_project_en"] = mp["master_project_en"].astype(str).str.strip()
    mp["master_developer_id"] = mp["master_developer_id"].astype(str)
    master_id_map = (
        mp.groupby(["master_project_en", "master_developer_id"])
        .size()
        .reset_index(name="n")
        .sort_values(["master_project_en", "n"], ascending=[True, False])
        .drop_duplicates("master_project_en")
        .set_index("master_project_en")["master_developer_id"]
        .to_dict()
    )

    devs = developers.set_index("developer_id")[["developer_name_en", "developer_name_ar"]]
    safe_training_brands = set(safe_training_brands or set())
    id_brand_overrides = dict(id_brand_overrides or {})
    entity_owner_overrides = dict(entity_owner_overrides or {})
    suspicious_label_tokens = set([t.upper() for t in (suspicious_label_tokens or set()) if str(t).strip()])
    suspicious_area_name_norms = set([_norm_label_any(x) for x in (suspicious_area_name_norms or set()) if _norm_label_any(x)])
    blocked_norms = set([_norm_label_any(x) for x in (blocked_brand_labels_norms or set()) if _norm_label_any(x)])

    def _is_holding_brand(x: object) -> bool:
        toks = [t.upper() for t in _tokenize_preserve(x)]
        return ("HOLDING" in toks) or ("HOLDINGS" in toks)

    audits = {
        "total_rows_seen": 0,
        "after_residential_property_filters": 0,
        "after_sales_filter": 0,
        "with_project_number_pct": None,
        "project_match_pct": None,
        "developer_registry_match_pct": None,
        "top_unmatched_project_numbers": [],
        "brand_resolution": {},
    }
    brand_source_counts: Dict[str, int] = {}
    # label coverage should be near-100% due to fallback chain; unresolved means label truly empty
    unresolved_examples: List[Dict[str, object]] = []
    unresolved_label_rows = 0
    unresolved_dev_counts: Dict[str, int] = {}
    unresolved_dev_names: Dict[str, str] = {}

    # run-level metrics for baseline/delta reporting
    seen_developer_ids: set[str] = set()
    seen_area_ids: set[str] = set()
    seen_project_numbers: set[str] = set()
    # developer_id -> per-month tx/value
    dev_monthly: Dict[str, Dict[str, Tuple[int, float]]] = {}
    max_year_month: Optional[str] = None

    # Evidence examples for reviewable suggestions:
    # developer_id -> list of up to 5 matched marketing strings (from TEXT evidence)
    dev_text_examples: Dict[str, Dict[str, int]] = {}
    # project_number -> TEXT brand counts (for linkage consistency)
    project_text_counts: Dict[str, Dict[str, int]] = {}
    # developer_id -> master_project_en counts and area_id counts (for outlier detection)
    dev_master_counts: Dict[str, Dict[str, int]] = {}
    dev_area_counts: Dict[str, Dict[str, int]] = {}

    # Label dispersion QA accumulators (label -> dev_id count; label -> area count; label -> samples)
    label_dev_counts: Dict[str, Dict[str, int]] = {}
    # Inverted view (developer_id -> label -> count) for assertions / owner override validation
    dev_label_counts: Dict[str, Dict[str, int]] = {}
    label_area_counts: Dict[str, Dict[str, int]] = {}
    label_master_samples: Dict[str, Dict[str, int]] = {}
    label_building_samples: Dict[str, Dict[str, int]] = {}
    label_source_counts: Dict[str, Dict[str, int]] = {}
    # Owner override self-check evidence accumulators:
    # developer_id -> (legal_name_en/ar/umbrella) -> tx_count
    dev_legal_en_counts: Dict[str, Dict[str, int]] = {}
    dev_legal_ar_counts: Dict[str, Dict[str, int]] = {}
    dev_umbrella_counts: Dict[str, Dict[str, int]] = {}
    # learned mapping counts from TEXT evidence only: developer_id -> brand -> tx_rows
    learned_text_counts: Dict[str, Dict[str, int]] = {}
    # developer_id -> distinct project_numbers seen in TEXT evidence (for confidence gating)
    dev_text_project_numbers: Dict[str, set[str]] = {}
    # False-positive trap accumulator: (alias, field) -> Counter-like dict of raw matched strings
    alias_traps: Dict[Tuple[str, str], Dict[str, int]] = {}


    unmatched_project_counts: Dict[str, int] = {}
    has_project = 0
    project_matched = 0
    dev_registry_matched = 0
    kept_rows = 0

    results = []

    usecols = [
        "transaction_id",
        "instance_date_parsed",
        "transaction_year",
        "transaction_month",
        "property_usage_en",
        "property_type_en",
        "reg_type_en",
        "rooms_en",
        "procedure_area",
        "actual_worth",
        "meter_sale_price",
        "project_number",
        "project_name_en",
        "master_project_en",
        "building_name_en",
        "area_id",
        "area_name_en",
        "trans_group_id",
    ]

    for chunk in pd.read_csv(tx_path, chunksize=chunk_size, usecols=usecols, low_memory=False):
        audits["total_rows_seen"] += len(chunk)

        # core filters
        chunk = chunk[chunk["property_usage_en"] == "Residential"]
        chunk = chunk[chunk["property_type_en"].isin(VALID_PROPERTY_TYPES)]
        chunk["bedroom"] = chunk["rooms_en"].map(BEDROOM_MAP)
        chunk = chunk.dropna(subset=["bedroom"])
        chunk["reg_type_dld"] = chunk["reg_type_en"].map(REG_TYPE_MAP)
        chunk = chunk.dropna(subset=["reg_type_dld"])
        chunk["reg_type"] = chunk["reg_type_dld"]

        # year_month
        chunk["year_month"] = (
            chunk["transaction_year"].astype(int).astype(str)
            + "-"
            + chunk["transaction_month"].astype(int).astype(str).str.zfill(2)
        )
        chunk["transaction_date"] = pd.to_datetime(chunk["instance_date_parsed"], errors="coerce")
        # track max month for last-12m anchoring
        if len(chunk):
            ym_max = chunk["year_month"].max()
            if isinstance(ym_max, str):
                if (max_year_month is None) or (ym_max > max_year_month):
                    max_year_month = ym_max

        # ------------------------------------------------------------
        # Price label (AED per sqft) - fix unit explicitly
        # DLD provides meter_sale_price (AED per sqm). We operate per sqft:
        #   price_sqft = price_sqm / 10.7639
        # If meter_sale_price missing, compute price_sqm = actual_worth / procedure_area (sqm) then convert.
        # ------------------------------------------------------------
        SQM_PER_SQFT = 0.09290304
        SQFT_PER_SQM = 1.0 / SQM_PER_SQFT  # 10.7639...

        chunk["meter_sale_price"] = pd.to_numeric(chunk["meter_sale_price"], errors="coerce")
        chunk["procedure_area"] = pd.to_numeric(chunk["procedure_area"], errors="coerce")
        chunk["actual_worth"] = pd.to_numeric(chunk["actual_worth"], errors="coerce")

        price_sqm = chunk["meter_sale_price"]
        missing_sqm = price_sqm.isna()
        if missing_sqm.any():
            # fallback compute price per sqm
            ok = missing_sqm & chunk["procedure_area"].notna() & (chunk["procedure_area"] > 0) & chunk["actual_worth"].notna() & (chunk["actual_worth"] > 0)
            price_sqm = price_sqm.copy()
            price_sqm.loc[ok] = chunk.loc[ok, "actual_worth"] / chunk.loc[ok, "procedure_area"]

        chunk["price_sqft"] = price_sqm / SQFT_PER_SQM
        chunk = chunk[chunk["price_sqft"].notna()]
        # Conservative sanity bounds in AED/sqft (same as V1 but now unit-correct)
        chunk = chunk[(chunk["price_sqft"] >= 100) & (chunk["price_sqft"] <= 50_000)]

        audits["after_residential_property_filters"] += len(chunk)

        # Sales filter (derived, not hardcoded)
        if CFG.filter_sales_only:
            chunk["trans_group_id"] = chunk["trans_group_id"].astype(str).str.strip()
            chunk = chunk[chunk["trans_group_id"].isin(sales_group_ids)]
        audits["after_sales_filter"] += len(chunk)

        if chunk.empty:
            continue

        # Normalize area_id
        chunk["area_id"] = chunk["area_id"].apply(_safe_int_str)
        chunk = chunk.dropna(subset=["area_id"])

        # Normalize project_number and track audits
        chunk["project_number"] = chunk["project_number"].apply(_safe_int_str)
        hp = chunk["project_number"].notna()
        has_project += int(hp.sum())

        # Join project info for legal developer_id and timeline (legal evidence channel)
        chunk = chunk.merge(proj_info, on="project_number", how="left")
        pm = chunk["developer_id"].notna()
        project_matched += int((hp & pm).sum())

        # Track unmatched project_numbers
        unmatch = hp & chunk["developer_id"].isna()
        if unmatch.any():
            for pn, n in chunk.loc[unmatch, "project_number"].value_counts().items():
                unmatched_project_counts[str(pn)] = unmatched_project_counts.get(str(pn), 0) + int(n)

        # Fallback legal developer_id using master_project_en -> master_developer_id (flagged)
        chunk["master_project_en"] = chunk["master_project_en"].astype(str).str.strip()
        chunk.loc[chunk["master_project_en"].str.lower().isin(["nan", "none", ""]), "master_project_en"] = np.nan
        chunk["developer_fallback_used"] = 0
        missing_dev = chunk["developer_id"].isna() & chunk["master_project_en"].notna()
        if missing_dev.any():
            chunk.loc[missing_dev, "developer_id"] = chunk.loc[missing_dev, "master_project_en"].map(master_id_map)
            chunk.loc[missing_dev & chunk["developer_id"].notna(), "developer_fallback_used"] = 1

        # Join developer registry (canonical)
        chunk["developer_id"] = chunk["developer_id"].astype(str)
        chunk = chunk.merge(devs, left_on="developer_id", right_index=True, how="left")
        dev_registry_matched += int(chunk["developer_name_en"].notna().sum())

        # ------------------------------------------------------------
        # Umbrella (explicit, nullable; used for owner-remap)
        # ------------------------------------------------------------
        did0 = chunk["developer_id"].astype(str).str.strip()
        did0 = did0.where((did0 != "") & (~did0.str.lower().isin(["nan", "none"])), other="")
        chunk["developer_umbrella"] = did0.map(umbrella_by_id).astype("object")
        chunk.loc[chunk["developer_umbrella"].astype(str).str.strip() == "", "developer_umbrella"] = pd.NA

        # ------------------------------------------------------------
        # Brand resolver (brand-first truth, must be 100% covered)
        # ------------------------------------------------------------
        # Fill missing master_project_en from Projects lookup (marketing-facing master project)
        if "master_project_en" in chunk.columns:
            chunk["master_project_en"] = chunk["master_project_en"].astype(str).str.strip()
            empty_mp = chunk["master_project_en"].str.lower().isin(["nan", "none", ""])
            if empty_mp.any():
                # project_number -> master_project_en from Projects
                mp_by_pn = projects[["project_number", "master_project_en"]].dropna(subset=["project_number"]).copy()
                mp_by_pn["project_number"] = mp_by_pn["project_number"].astype(str)
                mp_by_pn["master_project_en"] = mp_by_pn["master_project_en"].astype(str).str.strip()
                mp_map = mp_by_pn.dropna(subset=["master_project_en"]).set_index("project_number")["master_project_en"].to_dict()
                pn = chunk["project_number"].astype(str)
                fill = pn.map(mp_map)
                chunk.loc[empty_mp, "master_project_en"] = chunk.loc[empty_mp, "master_project_en"].where(~empty_mp, fill)
                chunk["master_project_en"] = chunk["master_project_en"].astype(str).str.strip()
                chunk.loc[chunk["master_project_en"].str.lower().isin(["nan", "none", ""]), "master_project_en"] = np.nan

        # Provide legal fields and text evidence into resolver.
        projects_by_pn = proj_info[["project_number", "developer_id"]].copy()
        # The resolver expects "project_number" as string column.
        projects_by_pn["project_number"] = projects_by_pn["project_number"].astype(str)
        resolved, brand_audit = brand_resolver.resolve_chunk(
            chunk[[
                "project_number",
                "project_name_en",
                "building_name_en",
                "master_project_en",
            ]].copy(),
            projects_by_project_number=projects_by_pn,
            # We enforce coverage at the pipeline level so we can write an audit file before failing.
            require_full_coverage=False,
        )
        # Attach brand fields back
        chunk["developer_brand"] = resolved["developer_brand"].values
        chunk["developer_brand_source"] = resolved["developer_brand_source"].values
        chunk["developer_brand_evidence"] = resolved["developer_brand_evidence"].values
        chunk["developer_brand_evidence_field"] = resolved["developer_brand_evidence_field"].values
        chunk["developer_brand_evidence_alias"] = resolved["developer_brand_evidence_alias"].values
        chunk["developer_brand_evidence_score"] = resolved["developer_brand_evidence_score"].values
        # Keep resolver output separate from final label
        chunk["developer_brand_public"] = chunk["developer_brand"].astype(str).str.strip()
        # accumulate coverage by source
        for k, v in (brand_audit.get("coverage_by_source") or {}).items():
            brand_source_counts[k] = brand_source_counts.get(k, 0) + int(v)

        # Learned mapping signals: for developer_ids that are NOT already explicitly mapped, collect TEXT-based brand outcomes
        if "developer_id" in chunk.columns:
            did = chunk["developer_id"].astype(str).str.strip()
            src = chunk["developer_brand_source"].astype(str).str.strip()
            text_mask2 = did.notna() & (did != "") & src.isin(["TEXT_STRONG", "TEXT_WEAK"])
            if text_mask2.any():
                sub = chunk.loc[text_mask2, ["developer_id", "developer_brand", "project_number"]].copy()
                sub["developer_id"] = sub["developer_id"].astype(str).str.strip()
                sub["developer_brand"] = sub["developer_brand"].astype(str).str.strip()
                sub["project_number"] = sub["project_number"].astype(str).str.strip()
                sub = sub[(sub["developer_id"] != "") & (sub["developer_brand"] != "")]
                # only collect for ids not explicitly mapped already
                sub = sub[~sub["developer_id"].isin(set(dev_id_to_brand.keys()))]
                if not sub.empty:
                    gb = sub.groupby(["developer_id", "developer_brand"]).size()
                    for (d, b), cnt in gb.items():
                        d = str(d)
                        b = str(b)
                        learned_text_counts.setdefault(d, {})
                        learned_text_counts[d][b] = learned_text_counts[d].get(b, 0) + int(cnt)
                    # distinct project_numbers observed under text evidence (avoid single-project skew)
                    sub_pn = sub[["developer_id", "project_number"]].copy()
                    sub_pn = sub_pn[(sub_pn["project_number"] != "") & (~sub_pn["project_number"].str.lower().isin(["nan", "none"]))]
                    if not sub_pn.empty:
                        for d, pns in sub_pn.groupby("developer_id")["project_number"]:
                            d = str(d)
                            dev_text_project_numbers.setdefault(d, set()).update(set([str(x) for x in pns.tolist() if str(x).strip()]))

        # False-positive trap sampling for TEXT matches only: collect top matched raw strings per alias.
        text_mask = chunk["developer_brand_source"].isin(["TEXT_STRONG", "TEXT_WEAK"])
        if text_mask.any():
            sample = chunk.loc[text_mask, ["developer_brand_evidence_alias", "developer_brand_evidence_field", "project_name_en", "building_name_en", "master_project_en"]].copy()
            # limit per chunk for speed
            sample = sample.head(50_000)
            # For each row, pick the field value that matched
            def pick_value(row):
                f = row["developer_brand_evidence_field"]
                if f == "project_name_en":
                    return row["project_name_en"]
                if f == "building_name_en":
                    return row["building_name_en"]
                if f == "master_project_en":
                    return row["master_project_en"]
                return ""
            sample["matched_value"] = sample.apply(pick_value, axis=1)
            sample["matched_value"] = sample["matched_value"].fillna("").astype(str).str.strip()
            sample = sample[(sample["developer_brand_evidence_alias"].astype(str).str.strip() != "") & (sample["matched_value"] != "")]
            if not sample.empty:
                # count occurrences
                grp = sample.groupby(["developer_brand_evidence_alias", "developer_brand_evidence_field"])["matched_value"].value_counts()
                # keep top 20 per (alias, field) for this chunk and merge into global dict
                for (alias, field, val), cnt in grp.groupby(level=[0, 1]).head(20).items():
                    key = (str(alias), str(field))
                    d = alias_traps.get(key)
                    if d is None:
                        d = {}
                        alias_traps[key] = d
                    d[str(val)] = d.get(str(val), 0) + int(cnt)

        # Per-developer evidence examples (TEXT only)
        if text_mask.any():
            cols = ["developer_id", "developer_brand_evidence_field", "project_name_en", "building_name_en", "master_project_en", "project_number", "developer_brand"]
            s2 = chunk.loc[text_mask, cols].copy()
            s2["developer_id"] = s2["developer_id"].astype(str).str.strip()
            s2["project_number"] = s2["project_number"].fillna("").astype(str).str.strip()

            def pick_val(row):
                f = row["developer_brand_evidence_field"]
                if f == "project_name_en":
                    return row["project_name_en"]
                if f == "building_name_en":
                    return row["building_name_en"]
                if f == "master_project_en":
                    return row["master_project_en"]
                # fallback: prefer project name then building then master
                return row.get("project_name_en") or row.get("building_name_en") or row.get("master_project_en") or ""

            s2["matched_value"] = s2.apply(pick_val, axis=1).fillna("").astype(str).str.strip()
            s2 = s2[(s2["developer_id"] != "") & (s2["matched_value"] != "")]
            if not s2.empty:
                # update per-dev examples
                gb = s2.groupby(["developer_id", "matched_value"]).size()
                for (did, mv), cnt in gb.items():
                    did = str(did)
                    mv = str(mv)
                    dev_text_examples.setdefault(did, {})
                    dev_text_examples[did][mv] = dev_text_examples[did].get(mv, 0) + int(cnt)
                # update per-project brand distribution for linkage consistency
                gp = s2.groupby(["project_number", "developer_brand"]).size()
                for (pn, b), cnt in gp.items():
                    pn = str(pn)
                    b = str(b)
                    if not pn or pn.lower() in {"nan", "none"}:
                        continue
                    project_text_counts.setdefault(pn, {})
                    project_text_counts[pn][b] = project_text_counts[pn].get(b, 0) + int(cnt)

        # developer_brand is intentionally "public brand only" or blank. Do not fill with legal names here.

        # Track seen ids for baseline/delta
        if "developer_id" in chunk.columns:
            seen_developer_ids.update([d for d in chunk["developer_id"].astype(str).str.strip().unique().tolist() if d and d.lower() not in {"nan", "none"}])
        if "area_id" in chunk.columns:
            seen_area_ids.update([a for a in chunk["area_id"].astype(str).str.strip().unique().tolist() if a and a.lower() not in {"nan", "none"}])
        if "project_number" in chunk.columns:
            pn_u = chunk["project_number"].dropna().astype(str).str.strip().unique().tolist()
            seen_project_numbers.update([p for p in pn_u if p and p.lower() not in {"nan", "none"}])

        # Per-dev per-month metrics (for baseline/delta & tripwires)
        chunk["actual_worth"] = pd.to_numeric(chunk["actual_worth"], errors="coerce").fillna(0.0).astype(float)
        m = chunk[["developer_id", "year_month", "transaction_id", "actual_worth"]].copy()
        m["developer_id"] = m["developer_id"].astype(str).str.strip()
        m["year_month"] = m["year_month"].astype(str)
        m = m[(m["developer_id"] != "") & (~m["developer_id"].str.lower().isin(["nan", "none"])) & (m["year_month"] != "")]
        if not m.empty:
            gb = m.groupby(["developer_id", "year_month"]).agg(tx=("transaction_id", "count"), value=("actual_worth", "sum"))
            for (did, ym), row in gb.iterrows():
                did = str(did)
                ym = str(ym)
                dev_monthly.setdefault(did, {})
                prev = dev_monthly[did].get(ym, (0, 0.0))
                dev_monthly[did][ym] = (int(prev[0]) + int(row["tx"]), float(prev[1]) + float(row["value"]))

        # Concentration stats inputs
        # master_project_en
        mp = chunk[["developer_id", "master_project_en"]].copy()
        mp["developer_id"] = mp["developer_id"].astype(str).str.strip()
        mp["master_project_en"] = mp["master_project_en"].fillna("").astype(str).str.strip()
        mp = mp[(mp["developer_id"] != "") & (~mp["developer_id"].str.lower().isin(["nan", "none"])) & (mp["master_project_en"] != "")]
        if not mp.empty:
            gbm = mp.groupby(["developer_id", "master_project_en"]).size()
            for (did, val), cnt in gbm.items():
                did = str(did)
                dev_master_counts.setdefault(did, {})
                dev_master_counts[did][str(val)] = dev_master_counts[did].get(str(val), 0) + int(cnt)
        # area_id
        ar = chunk[["developer_id", "area_id"]].copy()
        ar["developer_id"] = ar["developer_id"].astype(str).str.strip()
        ar["area_id"] = ar["area_id"].astype(str).str.strip()
        ar = ar[(ar["developer_id"] != "") & (~ar["developer_id"].str.lower().isin(["nan", "none"])) & (ar["area_id"] != "") & (~ar["area_id"].str.lower().isin(["nan", "none"]))]
        if not ar.empty:
            gba = ar.groupby(["developer_id", "area_id"]).size()
            for (did, val), cnt in gba.items():
                did = str(did)
                dev_area_counts.setdefault(did, {})
                dev_area_counts[did][str(val)] = dev_area_counts[did].get(str(val), 0) + int(cnt)

        # ------------------------------------------------------------
        # developer_brand_label + source (stable label; historic allowed)
        # ------------------------------------------------------------
        # 1) public brand if present
        chunk["developer_brand_label"] = chunk["developer_brand_public"].astype(str).str.strip()
        chunk["developer_brand_label_source"] = np.where(chunk["developer_brand_label"] != "", "PUBLIC_BRAND", "")

        # 2) Tier A: explicit developer_id -> canonical brand override (brand_overrides_developer_id.csv)
        miss = chunk["developer_brand_label"] == ""
        if miss.any() and id_brand_overrides:
            did = did0.copy()
            ovr = did.map(id_brand_overrides).fillna("").astype(str).str.strip()
            has = miss & (ovr != "")
            if has.any():
                chunk.loc[has, "developer_brand_label"] = ovr.loc[has]
                chunk.loc[has, "developer_brand_label_source"] = "ID_BRAND_OVERRIDE"

        # 3) Tier A: explicit entity owner overrides (place/project legal entities -> owning brand)
        miss = chunk["developer_brand_label"] == ""
        if miss.any() and entity_owner_overrides:
            did = did0.copy()
            ovr = did.map(entity_owner_overrides).fillna("").astype(str).str.strip()
            has = miss & (ovr != "")
            if has.any():
                chunk.loc[has, "developer_brand_label"] = ovr.loc[has]
                chunk.loc[has, "developer_brand_label_source"] = "OWNER_OVERRIDE_DEVELOPER_ID"

        # 4) Tier B/C: legal-normalized fallback with owner-aware suspicious handling
        miss = chunk["developer_brand_label"] == ""
        if miss.any():
            # candidate label: prefer EN then AR
            raw_en = chunk.loc[miss, "developer_name_en"].fillna("").astype(str)
            raw_ar = chunk.loc[miss, "developer_name_ar"].fillna("").astype(str)
            # authority detection must NOT depend on suffix-stripped candidates (e.g., DMCC may be stripped)
            is_auth_raw = raw_en.map(_label_is_authority_or_freezone) | raw_ar.map(_label_is_authority_or_freezone)

            en = raw_en.map(lambda x: normalize_legal_name_to_label(x, lang="en")).fillna("").astype(str).str.strip()
            ar = raw_ar.map(lambda x: normalize_legal_name_to_label(x, lang="ar")).fillna("").astype(str).str.strip()
            cand = en.copy()
            use_ar = cand == ""
            if use_ar.any():
                cand.loc[use_ar] = ar.loc[use_ar]

            # classify authority vs project/area
            is_auth = cand.map(_label_is_authority_or_freezone) | is_auth_raw
            is_proj = cand.map(
                lambda x: _label_is_project_or_area(
                    x,
                    suspicious_label_tokens=suspicious_label_tokens,
                    suspicious_area_name_norms=suspicious_area_name_norms,
                    blocked_norms=blocked_norms,
                )
            )

            # A) authorities/freezones: always DEVELOPER_ID_<id>
            did = did0.loc[miss]
            auth_has_id = is_auth & (did != "")
            if auth_has_id.any():
                idx = auth_has_id.index[auth_has_id]
                chunk.loc[idx, "developer_brand_label"] = "DEVELOPER_ID_" + did.loc[idx]
                chunk.loc[idx, "developer_brand_label_source"] = "ID_FALLBACK_AUTHORITY"

            # B) project/area labels: map to umbrella owner if safe, else DEVELOPER_ID_<id>
            still = (chunk["developer_brand_label"] == "") & miss
            if still.any():
                idx_still = still.index[still]
                cand2 = cand.loc[idx_still]
                did2 = did0.loc[idx_still]
                umb = chunk.loc[idx_still, "developer_umbrella"].fillna("").astype(str).str.strip()
                umb_is_auth = umb.map(_label_is_authority_or_freezone)
                # HOLDING policy: do not emit HOLDING/HOLDINGS labels from umbrella fallback (only allowed via PUBLIC_BRAND).
                umb_is_holding = umb.map(_is_holding_brand)
                umb_safe = (umb != "") & umb.isin(safe_training_brands) & (~umb_is_auth) & (~umb_is_holding)

                # determine which of the still-missing rows are project/area
                is_proj2 = cand2.map(
                    lambda x: _label_is_project_or_area(
                        x,
                        suspicious_label_tokens=suspicious_label_tokens,
                        suspicious_area_name_norms=suspicious_area_name_norms,
                        blocked_norms=blocked_norms,
                    )
                )
                # project/area + safe umbrella
                m_umb = (is_proj2 & umb_safe).fillna(False)
                idx_umb = m_umb.index[m_umb]
                if len(idx_umb):
                    chunk.loc[idx_umb, "developer_brand_label"] = umb.loc[idx_umb]
                    chunk.loc[idx_umb, "developer_brand_label_source"] = "UMBRELLA_OWNER_FALLBACK_SUSPICIOUS"

                # project/area without safe umbrella -> DEVELOPER_ID_<id>
                m_nuke = (is_proj2 & (~umb_safe) & (did2 != "")).fillna(False)
                idx_nuke = m_nuke.index[m_nuke]
                if len(idx_nuke):
                    chunk.loc[idx_nuke, "developer_brand_label"] = "DEVELOPER_ID_" + did2.loc[idx_nuke]
                    chunk.loc[idx_nuke, "developer_brand_label_source"] = "ID_FALLBACK_SUSPICIOUS"

                # C) non-project/area non-authority: allow legal-normalized label
                m_allow = ((~is_proj2) & (cand2 != "")).fillna(False)
                idx_allow = m_allow.index[m_allow]
                if len(idx_allow):
                    chunk.loc[idx_allow, "developer_brand_label"] = cand2.loc[idx_allow]
                    # source depends on which language was used (EN preferred, AR only when EN empty)
                    use_ar2 = use_ar.loc[idx_allow].fillna(False).astype(bool)
                    src = pd.Series(
                        np.where(use_ar2, "LEGAL_NORMALIZED_AR", "LEGAL_NORMALIZED_EN"),
                        index=idx_allow,
                        dtype="object",
                    )
                    chunk.loc[idx_allow, "developer_brand_label_source"] = src

        # 5) stable fallback: DEVELOPER_ID_<id>
        miss = chunk["developer_brand_label"] == ""
        if miss.any():
            did = did0.loc[miss]
            ok = did != ""
            if ok.any():
                idx = did.index[ok]
                chunk.loc[idx, "developer_brand_label"] = "DEVELOPER_ID_" + did.loc[idx]
                chunk.loc[idx, "developer_brand_label_source"] = "ID_FALLBACK"

        # 6) final stable fallback if developer_id is missing entirely (avoid unresolved)
        miss = chunk["developer_brand_label"] == ""
        if miss.any():
            chunk.loc[miss, "developer_brand_label"] = "DEVELOPER_ID_MISSING"
            chunk.loc[miss, "developer_brand_label_source"] = "MISSING_ID_FALLBACK"

        # ---- Label dispersion QA input collection (uses label + legal developer_id + area/master/building) ----
        # This catches fake labels like SKY COURTS / BUSINESS BAY becoming "developer labels" across many dev_ids.
        lbl = chunk["developer_brand_label"].astype(str).str.strip()
        did = chunk["developer_id"].astype(str).str.strip()
        aid = chunk["area_id"].astype(str).str.strip()
        mp = chunk["master_project_en"].fillna("").astype(str).str.strip()
        bn = chunk["building_name_en"].fillna("").astype(str).str.strip()
        # Count per label->developer_id and label->area_id
        gb = chunk.groupby([lbl, did]).size()
        for (l, d), cnt in gb.items():
            l = str(l); d = str(d)
            if not l or not d or d.lower() in {"nan","none"}:
                continue
            label_dev_counts.setdefault(l, {})
            label_dev_counts[l][d] = label_dev_counts[l].get(d, 0) + int(cnt)
            dev_label_counts.setdefault(d, {})
            dev_label_counts[d][l] = dev_label_counts[d].get(l, 0) + int(cnt)
        ga = chunk.groupby([lbl, aid]).size()
        for (l, a), cnt in ga.items():
            l = str(l); a = str(a)
            if not l or not a or a.lower() in {"nan","none"}:
                continue
            label_area_counts.setdefault(l, {})
            label_area_counts[l][a] = label_area_counts[l].get(a, 0) + int(cnt)
        # samples: master_project_en and building_name_en (top frequent)
        gm = chunk.groupby([lbl, mp]).size()
        for (l, v), cnt in gm.items():
            l = str(l); v = str(v).strip()
            if not l or not v or v.lower() in {"nan","none"}:
                continue
            label_master_samples.setdefault(l, {})
            label_master_samples[l][v] = label_master_samples[l].get(v, 0) + int(cnt)
        gn = chunk.groupby([lbl, bn]).size()
        for (l, v), cnt in gn.items():
            l = str(l); v = str(v).strip()
            if not l or not v or v.lower() in {"nan","none"}:
                continue
            label_building_samples.setdefault(l, {})
            label_building_samples[l][v] = label_building_samples[l].get(v, 0) + int(cnt)

        # label -> source distribution
        srcs = chunk[["developer_brand_label", "developer_brand_label_source"]].copy()
        srcs["developer_brand_label"] = srcs["developer_brand_label"].astype(str).str.strip()
        srcs["developer_brand_label_source"] = srcs["developer_brand_label_source"].astype(str).str.strip()
        gs = srcs.groupby(["developer_brand_label", "developer_brand_label_source"]).size()
        for (l, s), cnt in gs.items():
            l = str(l); s = str(s)
            if not l:
                continue
            label_source_counts.setdefault(l, {})
            label_source_counts[l][s] = label_source_counts[l].get(s, 0) + int(cnt)

        # developer_id -> source distribution (for assertions/proofs)
        dev_src = chunk[["developer_id", "developer_brand_label_source"]].copy()
        dev_src["developer_id"] = dev_src["developer_id"].astype(str).str.strip()
        dev_src["developer_brand_label_source"] = dev_src["developer_brand_label_source"].astype(str).str.strip()
        dev_src = dev_src[(dev_src["developer_id"] != "") & (~dev_src["developer_id"].str.lower().isin(["nan", "none"])) & (dev_src["developer_brand_label_source"] != "")]
        if not dev_src.empty:
            gds = dev_src.groupby(["developer_id", "developer_brand_label_source"]).size()
            for (d, s), cnt in gds.items():
                d = str(d); s = str(s)
                audits.setdefault("_dev_label_source_counts", {})
                mp = audits["_dev_label_source_counts"].setdefault(d, {})  # type: ignore[index]
                mp[s] = mp.get(s, 0) + int(cnt)

        # ---- Owner override evidence accumulators (top EN/AR legal names + umbrella) ----
        did_sc = chunk["developer_id"].astype(str).str.strip()
        did_sc = did_sc.where((did_sc != "") & (~did_sc.str.lower().isin(["nan", "none"])), other="")
        if (did_sc != "").any():
            en_sc = chunk["developer_name_en"].fillna("").astype(str).str.strip()
            ar_sc = chunk["developer_name_ar"].fillna("").astype(str).str.strip()
            umb_sc = chunk["developer_umbrella"].fillna("").astype(str).str.strip()

            tmp = pd.DataFrame({"developer_id": did_sc, "v": en_sc})
            tmp = tmp[(tmp["developer_id"] != "") & (tmp["v"] != "")]
            if not tmp.empty:
                g = tmp.groupby(["developer_id", "v"]).size()
                for (d, v), cnt in g.items():
                    d = str(d); v = str(v)
                    dev_legal_en_counts.setdefault(d, {})
                    dev_legal_en_counts[d][v] = dev_legal_en_counts[d].get(v, 0) + int(cnt)

            tmp = pd.DataFrame({"developer_id": did_sc, "v": ar_sc})
            tmp = tmp[(tmp["developer_id"] != "") & (tmp["v"] != "")]
            if not tmp.empty:
                g = tmp.groupby(["developer_id", "v"]).size()
                for (d, v), cnt in g.items():
                    d = str(d); v = str(v)
                    dev_legal_ar_counts.setdefault(d, {})
                    dev_legal_ar_counts[d][v] = dev_legal_ar_counts[d].get(v, 0) + int(cnt)

            tmp = pd.DataFrame({"developer_id": did_sc, "v": umb_sc})
            tmp = tmp[(tmp["developer_id"] != "") & (tmp["v"] != "")]
            if not tmp.empty:
                g = tmp.groupby(["developer_id", "v"]).size()
                for (d, v), cnt in g.items():
                    d = str(d); v = str(v)
                    dev_umbrella_counts.setdefault(d, {})
                    dev_umbrella_counts[d][v] = dev_umbrella_counts[d].get(v, 0) + int(cnt)

        # Label coverage check (should be extremely rare now)
        missing_label = chunk["developer_brand_label"].astype(str).str.strip() == ""
        if missing_label.any():
            unresolved_label_rows += int(missing_label.sum())
            # Count unresolved developer_ids for override closure
            did = chunk.loc[missing_label, "developer_id"].astype(str).str.strip()
            for d in did.tolist():
                if not d or str(d).lower() in {"nan", "none"}:
                    continue
                unresolved_dev_counts[str(d)] = unresolved_dev_counts.get(str(d), 0) + 1
            # Record a representative English legal name per developer_id (for the override report)
            nm = chunk.loc[missing_label, ["developer_id", "developer_name_en"]].copy()
            nm["developer_id"] = nm["developer_id"].astype(str).str.strip()
            nm["developer_name_en"] = nm["developer_name_en"].astype(str).str.strip()
            for _, r in nm.iterrows():
                d = r["developer_id"]
                if d and d not in unresolved_dev_names and r["developer_name_en"]:
                    unresolved_dev_names[d] = r["developer_name_en"]
            # capture small sample for audit report
            sample = chunk.loc[
                missing_label,
                ["transaction_id", "year_month", "area_id", "area_name_en", "project_number", "project_name_en", "building_name_en", "master_project_en"],
            ].head(200)
            unresolved_examples.extend(sample.to_dict(orient="records"))

        # No auto-buckets: if still missing, it will be handled by overrides or fail.

        # Phase features (same semantics as V1)
        chunk["months_since_launch"] = np.where(
            chunk["project_start_date_parsed"].notna() & chunk["transaction_date"].notna(),
            ((chunk["transaction_date"] - chunk["project_start_date_parsed"]).dt.days / 30.44).clip(lower=0),
            np.nan,
        )
        chunk["months_to_handover_signed"] = np.where(
            chunk["handover_date"].notna() & chunk["transaction_date"].notna(),
            ((chunk["handover_date"] - chunk["transaction_date"]).dt.days / 30.44),
            np.nan,
        )
        chunk["dld_offplan_after_handover"] = 0
        has_h = chunk["handover_date"].notna() & chunk["transaction_date"].notna()
        after = has_h & (chunk["transaction_date"] >= chunk["handover_date"])
        dld_offplan = chunk["reg_type_dld"].astype(str) == "OffPlan"
        flip = dld_offplan & after
        if flip.any():
            chunk.loc[flip, "reg_type"] = "Ready"
            chunk.loc[flip, "dld_offplan_after_handover"] = 1
        chunk["months_to_handover"] = np.where(
            np.isfinite(chunk["months_to_handover_signed"]),
            np.maximum(chunk["months_to_handover_signed"], 0),
            np.nan,
        )
        chunk["months_since_handover"] = np.where(
            np.isfinite(chunk["months_to_handover_signed"]),
            np.maximum(-chunk["months_to_handover_signed"], 0),
            np.nan,
        )
        chunk["handover_window_6m"] = np.where(
            np.isfinite(chunk["months_to_handover_signed"]),
            (np.abs(chunk["months_to_handover_signed"]) <= 6).astype(int),
            0,
        )
        chunk["project_percent_complete"] = chunk["percent_completed_num"]
        chunk["phase_ratio"] = np.where(
            chunk["project_duration_months"].notna() & (chunk["project_duration_months"] > 0),
            (chunk["months_since_launch"] / chunk["project_duration_months"]).clip(0, 1),
            np.nan,
        )

        kept_rows += len(chunk)
        results.append(chunk)

    if not results:
        raise RuntimeError("No transactions after filters; check inputs/filters.")

    df = pd.concat(results, ignore_index=True)

    # Aggregate (developer_brand is the modeling key; retain legal developer_id + legal name for audit)
    group_cols = ["year_month", "area_id", "property_type_en", "bedroom", "reg_type", "developer_brand_label"]
    agg = (
        df.groupby(group_cols)
        .agg(
            median_price=("price_sqft", "median"),
            transaction_count=("transaction_id", "count"),
            months_since_launch=("months_since_launch", "median"),
            months_to_handover=("months_to_handover", "median"),
            months_to_handover_signed=("months_to_handover_signed", "median"),
            months_since_handover=("months_since_handover", "median"),
            handover_window_6m=("handover_window_6m", "median"),
            dld_offplan_after_handover=("dld_offplan_after_handover", "max"),
            project_percent_complete=("project_percent_complete", "median"),
            project_duration_months=("project_duration_months", "median"),
            phase_ratio=("phase_ratio", "median"),
            # audit fields
            developer_id=("developer_id", _mode_or_nan),
            developer_legal_name_en=("developer_name_en", _mode_or_nan),
            developer_brand_public=("developer_brand_public", _mode_or_nan),
            developer_brand_source=("developer_brand_source", _mode_or_nan),
            developer_brand_evidence=("developer_brand_evidence", _mode_or_nan),
            developer_brand_evidence_field=("developer_brand_evidence_field", _mode_or_nan),
            developer_brand_evidence_alias=("developer_brand_evidence_alias", _mode_or_nan),
            developer_brand_evidence_score=("developer_brand_evidence_score", "median"),
            developer_brand_label_source=("developer_brand_label_source", _mode_or_nan),
            developer_umbrella=("developer_umbrella", _mode_or_nan),
            developer_fallback_used=("developer_fallback_used", "max"),
        )
        .reset_index()
    )
    agg = agg.rename(columns={"property_type_en": "property_type"})
    # Back-compat: keep `developer_brand` as the final stable label used for modeling features.
    agg["developer_brand"] = agg["developer_brand_label"]

    # Compute audit summary
    audits["with_project_number_pct"] = float(has_project / max(audits["after_sales_filter"], 1))
    audits["project_match_pct"] = float(project_matched / max(has_project, 1))
    audits["developer_registry_match_pct"] = float(dev_registry_matched / max(kept_rows, 1))
    top_unmatched = sorted(unmatched_project_counts.items(), key=lambda kv: kv[1], reverse=True)[:50]
    audits["top_unmatched_project_numbers"] = [{"project_number": k, "count": v} for k, v in top_unmatched]

    logger.info(
        "Off-plan/dev audits: "
        f"project_number present={audits['with_project_number_pct']:.1%}, "
        f"project match={audits['project_match_pct']:.1%}, "
        f"developer registry match={audits['developer_registry_match_pct']:.1%}"
    )

    audits["brand_resolution"] = {
        "coverage_by_source": brand_source_counts,
        "unresolved_label_rows": int(unresolved_label_rows),
        "secondary_market_brand": None,
        "brand_required_reg_types": ["ALL"],
        "unresolved_examples_sample": unresolved_examples[:200],
        "false_positive_trap_sample": {
            f"{alias}::{field}": dict(sorted(vals.items(), key=lambda kv: kv[1], reverse=True)[:20])
            for (alias, field), vals in list(alias_traps.items())[:500]
        },
    }

    # Persist run-level sets/metrics for baseline/delta even on failure
    audits["run_sets"] = {
        "developer_ids_seen_count": int(len(seen_developer_ids)),
        "area_ids_seen_count": int(len(seen_area_ids)),
        "project_numbers_seen_count": int(len(seen_project_numbers)),
    }
    audits["_run_sets_raw"] = {
        "developer_ids_seen": sorted(list(seen_developer_ids))[:5000],
        "area_ids_seen": sorted(list(seen_area_ids))[:5000],
        "project_numbers_seen": sorted(list(seen_project_numbers))[:5000],
    }

    # Only hard-fail if brand_label truly cannot be produced (should be extremely rare)
    if CFG.require_full_brand_coverage and unresolved_label_rows > 0:
        # ------------------------------------------------------------
        # Closure pipeline artifacts (written before failing)
        # ------------------------------------------------------------
        # Coverage curve
        top_sorted = sorted(unresolved_dev_counts.items(), key=lambda kv: kv[1], reverse=True)
        total_unres = int(sum(unresolved_dev_counts.values()))
        curve_points = [10, 25, 50, 100, 200, 500, 1000]
        curve = []
        cum = 0
        for i, (did, cnt) in enumerate(top_sorted, start=1):
            cum += int(cnt)
            if i in curve_points:
                curve.append({"top_n_developer_ids": i, "rows_covered": cum, "pct_covered": float(cum / max(total_unres, 1))})

        # Load umbrella map (schema: {"map": {...}})
        umbrella_path = Path(CFG.umbrella_map_path)
        umbrella = {}
        if umbrella_path.exists():
            try:
                umbrella = json.loads(umbrella_path.read_text(encoding="utf-8"))
            except Exception:
                umbrella = {}
        umbrella_by_id = (umbrella.get("map") or {}) if isinstance(umbrella, dict) else {}

        # Suggestion helpers
        approved = set(getattr(brand_resolver, "approved_brands", []) or [])
        aliases_by_brand = getattr(brand_resolver, "brand_aliases", {}) or {}
        # Precompile alias patterns
        compiled: List[Tuple[str, str, re.Pattern, int]] = []  # (brand, alias, pat, score)
        for b, als in aliases_by_brand.items():
            if b not in approved:
                continue
            for a in (als or []):
                pat = _boundary_phrase_pattern(str(a))
                if pat is None:
                    continue
                toks = _norm_upper_spaces(a).split()
                # drop too-short single-token aliases (false positive risk)
                if len(toks) == 1 and len(toks[0]) < 5:
                    continue
                score = 1000 * len(toks) + len(_norm_upper_spaces(a))
                compiled.append((str(b), str(a), pat, score))

        def suggest_from_legal_name(dev_id: str, legal_name: str) -> Tuple[str, str, str]:
            # returns (brand, confidence, reason)
            if dev_id in umbrella_by_id and str(umbrella_by_id[dev_id]).strip():
                b = str(umbrella_by_id[dev_id]).strip()
                if b in approved:
                    return b, "HIGH", "umbrella_map developer_id"
            nm = _norm_upper_spaces(legal_name)
            if not nm:
                return "", "LOW", "missing legal name"

            # exact brand label token/phrase match
            best = ("", -1, "", "")
            for b in approved:
                pat = _boundary_phrase_pattern(b)
                if pat and pat.search(nm):
                    score = 2000 + len(_norm_upper_spaces(b))
                    if score > best[1]:
                        best = (b, score, "HIGH", "legal name contains brand token/phrase")

            # alias match
            for b, a, pat, score in compiled:
                if pat.search(nm):
                    conf = "HIGH" if score >= 2000 else "MED"
                    if score > best[1]:
                        best = (b, score, conf, f"legal name matches alias '{a}'")

            if best[0]:
                return best[0], best[2], best[3]
            return "", "LOW", "no strong token/alias match"

        # Learned mapping file
        learned_rows = []
        for did, dist in learned_text_counts.items():
            total = int(sum(dist.values()))
            if total < CFG.learned_min_tx:
                continue
            top_brand, top_cnt = sorted(dist.items(), key=lambda kv: kv[1], reverse=True)[0]
            share = float(top_cnt / max(total, 1))
            if share < CFG.learned_min_share:
                continue
            conf = "HIGH" if share >= 0.95 else "MED"
            if top_brand not in approved:
                continue
            learned_rows.append(
                {
                    "developer_id": did,
                    "suggested_brand": top_brand,
                    "tx_rows_text_resolved": total,
                    "top_brand_share": round(share, 4),
                    "suggestion_confidence": conf,
                    "suggestion_reason": f"mode brand from TEXT evidence (min_tx={CFG.learned_min_tx}, min_share={CFG.learned_min_share})",
                }
            )
        learned_df = pd.DataFrame(learned_rows).sort_values(["tx_rows_text_resolved", "top_brand_share"], ascending=[False, False]) if learned_rows else pd.DataFrame(
            columns=["developer_id", "suggested_brand", "tx_rows_text_resolved", "top_brand_share", "suggestion_confidence", "suggestion_reason"]
        )
        learned_path = OUTPUT_DIR / "learned_public_brand_map_developer_id.csv"
        learned_df.to_csv(learned_path, index=False)

        learned_lookup = {str(r["developer_id"]): str(r["suggested_brand"]) for r in learned_rows}

        # Suggestions file for overrides
        sugg_rows = []
        for did, cnt in top_sorted:
            legal = unresolved_dev_names.get(did, "")
            b, conf, reason = suggest_from_legal_name(did, legal)
            if not b and did in learned_lookup:
                b = learned_lookup[did]
                conf = "MED"
                reason = "learned map from TEXT evidence"
            sugg_rows.append(
                {
                    "developer_id": did,
                    "developer_legal_name_en": legal,
                    "tx_rows_unresolved": int(cnt),
                    "suggested_brand": b,
                    "suggestion_confidence": conf,
                    "suggestion_reason": reason,
                }
            )
        sugg_df = pd.DataFrame(sugg_rows) if sugg_rows else pd.DataFrame(
            columns=["developer_id", "developer_legal_name_en", "tx_rows_unresolved", "suggested_brand", "suggestion_confidence", "suggestion_reason"]
        )
        sugg_path = OUTPUT_DIR / "suggested_public_brand_overrides_developer_id.csv"
        sugg_df.to_csv(sugg_path, index=False)

        if "suggested_brand" in sugg_df.columns and "suggestion_confidence" in sugg_df.columns and "tx_rows_unresolved" in sugg_df.columns:
            high_resolvable = int(
                sugg_df[(sugg_df["suggested_brand"].astype(str).str.strip() != "") & (sugg_df["suggestion_confidence"] == "HIGH")]["tx_rows_unresolved"].sum()
            )
        else:
            high_resolvable = 0

        # Write an actionable audit file and fail hard.
        audit_path = OUTPUT_DIR / "brand_resolution_audit_v2.json"
        top_unmapped = sorted(unresolved_dev_counts.items(), key=lambda kv: kv[1], reverse=True)[:500]
        top_unmapped = [
            {
                "developer_id": did,
                "count_rows": cnt,
                "developer_name_en": unresolved_dev_names.get(did, ""),
            }
            for did, cnt in top_unmapped
        ]
        audit_payload = {
            "unresolved_label_rows": int(unresolved_label_rows),
            "coverage_by_source": brand_source_counts,
            "brand_required_reg_types": ["ALL"],
            "secondary_market_brand": None,
            "examples": unresolved_examples[:500],
            "unmapped_developer_ids_top": top_unmapped,
            "unresolved_coverage_curve": curve,
            "closure_artifacts": {
                "suggested_overrides_developer_id_csv": str(sugg_path),
                "learned_brand_map_developer_id_csv": str(learned_path),
                "high_conf_rows_resolvable_if_applied": high_resolvable,
                "total_unresolved_rows": total_unres,
            },
            "notes": "Label fallback should prevent unresolved. If this triggers, investigate missing developer_id/names. Review suggested_public_brand_overrides_developer_id.csv and learned_public_brand_map_developer_id.csv.",
        }
        audit_path.write_text(json.dumps(audit_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        raise RuntimeError(
            "Brand label generation failed (should be near-impossible): "
            f"{unresolved_label_rows} rows have empty developer_brand_label. See {audit_path}"
        )

    # Expose dev_monthly metrics + learned text brand distributions for baseline/delta/suggestions in main
    audits["_dev_monthly"] = dev_monthly
    audits["_max_year_month"] = max_year_month
    audits["_learned_text_counts"] = learned_text_counts
    # compact: keep only top 5 examples per developer_id
    compact_examples = {}
    for did, d in dev_text_examples.items():
        top = sorted(d.items(), key=lambda kv: kv[1], reverse=True)[:5]
        compact_examples[did] = [{"text": t, "count": int(c)} for t, c in top]
    audits["_dev_text_examples_top5"] = compact_examples
    audits["_project_text_counts"] = project_text_counts
    audits["_dev_master_counts"] = dev_master_counts
    audits["_dev_area_counts"] = dev_area_counts
    audits["_dev_text_project_numbers_count"] = {k: int(len(v)) for k, v in dev_text_project_numbers.items()}
    audits["_label_dev_counts"] = label_dev_counts
    audits["_dev_label_counts"] = dev_label_counts
    audits["_label_area_counts"] = label_area_counts
    audits["_label_master_samples"] = label_master_samples
    audits["_label_building_samples"] = label_building_samples
    audits["_label_source_counts"] = label_source_counts
    audits["_dev_legal_en_counts"] = dev_legal_en_counts
    audits["_dev_legal_ar_counts"] = dev_legal_ar_counts
    audits["_dev_umbrella_counts"] = dev_umbrella_counts
    return agg, audits


def aggregate_rents_v2(chunk_size: int = 200_000) -> pd.DataFrame:
    path = CLEANED_DIR / "Rent_Contracts_Cleaned.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing cleaned rents: {path}")

    results = []
    usecols = [
        "contract_start_date_parsed",
        "annual_amount",
        "area_name_en",
        "bedrooms",
        "actual_area",
        "property_usage_en",
    ]
    for chunk in pd.read_csv(path, chunksize=chunk_size, usecols=usecols, low_memory=False):
        chunk = chunk[chunk["property_usage_en"] == "Residential"]
        chunk["date"] = pd.to_datetime(chunk["contract_start_date_parsed"], errors="coerce")
        chunk = chunk.dropna(subset=["date"])
        chunk["year_month"] = chunk["date"].dt.strftime("%Y-%m")
        # Map rent areas (strings) -> area_id using a mapping built from Transactions.
        # This avoids relying on brittle joins to Lkp_Areas name_en.
        if "_area_name_to_id" not in aggregate_rents_v2.__dict__:
            raise RuntimeError("aggregate_rents_v2 requires aggregate_rents_v2._area_name_to_id to be set")
        area_map = aggregate_rents_v2.__dict__["_area_name_to_id"]
        chunk["area_id"] = chunk["area_name_en"].astype(str).str.strip().map(area_map)
        chunk["area_id"] = chunk["area_id"].apply(_safe_int_str)
        chunk = chunk.dropna(subset=["area_id"])

        chunk["annual_amount"] = pd.to_numeric(chunk["annual_amount"], errors="coerce").astype(float)
        chunk = chunk[chunk["annual_amount"].notna()]
        chunk = chunk[(chunk["annual_amount"] > 0) & (chunk["annual_amount"] < 100_000_000)]
        # rent sqft
        chunk["actual_area"] = pd.to_numeric(chunk["actual_area"], errors="coerce")
        chunk["rent_sqft"] = np.where(chunk["actual_area"] > 0, chunk["annual_amount"] / chunk["actual_area"], np.nan)

        results.append(chunk)

    df = pd.concat(results, ignore_index=True)
    agg = (
        df.groupby(["year_month", "area_id", "bedrooms"])
        .agg(median_rent=("annual_amount", "median"), rent_count=("annual_amount", "count"), median_rent_sqft=("rent_sqft", "median"))
        .reset_index()
        .rename(columns={"bedrooms": "bedroom"})
    )
    return agg


def build_area_name_to_id_map_from_transactions(chunk_size: int = 500_000) -> Dict[str, str]:
    """
    Build a robust mapping from area_name_en -> dominant area_id using Transactions_Cleaned.csv.
    This is used to attach area_id to datasets that only have area_name_en (e.g., cleaned rents).
    """
    path = CLEANED_DIR / "Transactions_Cleaned.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing cleaned transactions: {path}")
    counts: Dict[Tuple[str, str], int] = {}
    for chunk in pd.read_csv(path, chunksize=chunk_size, usecols=["area_name_en", "area_id"], low_memory=False):
        chunk["area_name_en"] = chunk["area_name_en"].fillna("").astype(str).str.strip()
        chunk["area_id"] = chunk["area_id"].apply(_safe_int_str)
        chunk = chunk[(chunk["area_name_en"] != "") & chunk["area_id"].notna()]
        # count pairs in chunk
        vc = chunk.groupby(["area_name_en", "area_id"]).size()
        for (name, aid), n in vc.items():
            counts[(name, aid)] = counts.get((name, aid), 0) + int(n)
    # pick dominant area_id per name
    best: Dict[str, Tuple[str, int]] = {}
    for (name, aid), n in counts.items():
        cur = best.get(name)
        if cur is None or n > cur[1]:
            best[name] = (aid, n)
    out = {name: aid for name, (aid, _) in best.items()}
    logger.info(f"Built area_name_en->area_id map for {len(out):,} names from Transactions_Cleaned")
    return out


def compute_supply_v2(projects: pd.DataFrame, months: List[str]) -> pd.DataFrame:
    active_statuses = {"ACTIVE", "NOT_STARTED", "PENDING"}
    p = projects.dropna(subset=["area_id"]).copy()
    p["project_status"] = p["project_status"].astype(str).str.strip().str.upper()
    results = []
    for month in months:
        month_date = pd.to_datetime(month + "-01")
        mask = p["project_status"].isin(active_statuses) & (
            p["project_end_date_parsed"].isna() | (p["project_end_date_parsed"] > month_date)
        )
        active = p[mask]
        if active.empty:
            continue
        supply = (
            active.groupby("area_id")
            .agg(
                supply_units=("no_of_units", "sum"),
                supply_buildings=("no_of_buildings", "sum"),
                supply_villas=("no_of_villas", "sum"),
                active_projects=("project_number", "count"),
            )
            .reset_index()
        )
        supply["year_month"] = month
        results.append(supply)
    if not results:
        return pd.DataFrame(columns=["year_month", "area_id", "supply_units", "supply_buildings", "supply_villas", "active_projects"])
    return pd.concat(results, ignore_index=True)


def compute_units_completing_v2(projects: pd.DataFrame, months: List[str]) -> pd.DataFrame:
    """
    Compute an area-level monthly *scheduled completions* series: units completing per (year_month, area_id).

    This is a core investor-facing signal: future supply pressure and handover clustering.

    Definition (V2):
    - Use Projects' canonical handover date: handover_date = completion_date if present else project_end_date.
    - Sum no_of_units for projects whose handover_date falls within the month.
    - Exclude cancelled projects.

    Notes:
    - This is intended to be used as a *known future real* in TFT (for training it is known; for inference
      it must be generated to the same horizon from the Projects schedule).
    """
    p = projects.dropna(subset=["area_id"]).copy()
    # ensure date is parsed (load_projects_v2 sets these)
    if "handover_date" not in p.columns:
        return pd.DataFrame(columns=["year_month", "area_id", "units_completing"])

    # Exclude cancelled projects (cancellation_date_parsed is set in load_projects_v2)
    if "cancellation_date_parsed" in p.columns:
        p = p[p["cancellation_date_parsed"].isna()]
    elif "is_cancelled" in p.columns:
        p = p[~p["is_cancelled"].astype(bool)]

    p = p.dropna(subset=["handover_date"])
    if p.empty:
        return pd.DataFrame(columns=["year_month", "area_id", "units_completing"])

    # Normalize units
    p["no_of_units"] = pd.to_numeric(p.get("no_of_units"), errors="coerce").fillna(0).astype(int)
    p["year_month"] = p["handover_date"].dt.strftime("%Y-%m")

    out = (
        p.groupby(["year_month", "area_id"], as_index=False)
        .agg(units_completing=("no_of_units", "sum"))
    )
    # restrict to observed month range for the training frame
    months_set = set([m for m in (months or []) if m])
    if months_set:
        out = out[out["year_month"].isin(months_set)]
    return out


def attach_area_names(df: pd.DataFrame, lkp_areas: pd.DataFrame) -> pd.DataFrame:
    out = df.merge(lkp_areas[["area_id", "name_en"]].rename(columns={"name_en": "area_name"}), on="area_id", how="left")
    out["area_name"] = out["area_name"].fillna("UNKNOWN_AREA")
    return out


def add_sale_index_features(df: pd.DataFrame, sale_index: pd.DataFrame) -> pd.DataFrame:
    out = df.merge(sale_index, on="year_month", how="left")
    # property-type-specific index
    out["sale_index"] = out["all_monthly_index"]
    is_unit = out["property_type"].astype(str).str.lower().isin(["unit"])
    is_villa = out["property_type"].astype(str).str.lower().isin(["villa"])
    if "flat_monthly_index" in out.columns:
        out.loc[is_unit, "sale_index"] = out.loc[is_unit, "flat_monthly_index"]
    if "villa_monthly_index" in out.columns:
        out.loc[is_villa, "sale_index"] = out.loc[is_villa, "villa_monthly_index"]
    out["sale_index_missing"] = out["sale_index"].isna().astype(int)
    out["sale_index"] = out["sale_index"].ffill().bfill()
    out["sale_index"] = out["sale_index"].fillna(0)
    return out


def create_tft_columns_v2(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # series key
    if CFG.include_developer_in_group_id:
        gid = (
            out["area_id"].astype(str)
            + "_"
            + out["property_type"].astype(str).str.replace(" ", "_")
            + "_"
            + out["bedroom"].astype(str).str.replace(" ", "_")
            + "_"
            + out["reg_type"].astype(str)
            + "_"
            + out["developer_brand"].astype(str).str.replace(r"\s+", "_", regex=True).str.replace(r"[,'\.]", "", regex=True)
        )
    else:
        gid = (
            out["area_id"].astype(str)
            + "_"
            + out["property_type"].astype(str).str.replace(" ", "_")
            + "_"
            + out["bedroom"].astype(str).str.replace(" ", "_")
            + "_"
            + out["reg_type"].astype(str)
        )
    out["group_id"] = gid
    out["date"] = pd.to_datetime(out["year_month"] + "-01")
    out["month"] = out["date"].dt.month
    out["quarter"] = out["date"].dt.quarter
    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12)
    out = out.sort_values(["group_id", "year_month"])
    out["time_idx"] = out.groupby("group_id").cumcount()
    out = out.drop(columns=["date"])
    return out


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    global OUTPUT_DIR
    logger.info("=" * 60)
    logger.info("TFT DATA BUILD V2 - START")
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # Run identity + per-build output folder (single coherent bundle)
    # ------------------------------------------------------------------
    import argparse
    from datetime import datetime, timezone
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa_mode", default="", help="Run QA-only synthetic checks (e.g., dispersion_gate) and exit.")
    args = parser.parse_args()

    build_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    build_timestamp_utc = datetime.now(timezone.utc).isoformat()

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    run_dir = RUNS_DIR / build_id
    run_dir.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR = run_dir

    LATEST_DIR.mkdir(parents=True, exist_ok=True)

    # Run manifest is written early so failures are attributable to a single build_id.
    (run_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "build_id": build_id,
                "build_timestamp_utc": build_timestamp_utc,
                "status": "RUNNING",
                "note": "All artifacts for this run are written under this folder. Promotion to Data/tft/latest occurs only on success.",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    if str(args.qa_mode).strip().lower() == "dispersion_gate":
        # Synthetic proof: create a dispersion table that must trigger the hard gate.
        fake = pd.DataFrame(
            [
                {"label": "SKY COURTS", "tx_count": 1500, "unique_developer_ids": 50, "top_developer_id_share": 0.05,
                 "top_5_developer_ids": "1:75|2:70|3:60|4:55|5:50", "unique_areas": 10, "top_area_share": 0.20,
                 "example_master_project_en_samples": "Dubai Land Residence Complex", "example_building_name_en_samples": "Skycourts Tower A"},
            ]
        )
        fake_path = OUTPUT_DIR / "label_dispersion_report_v2_synthetic.csv"
        fake.to_csv(fake_path, index=False)
        offenders = fake[(fake["tx_count"] >= 1000) & (fake["unique_developer_ids"] >= 10) & (fake["top_developer_id_share"] <= 0.30)]
        off_path = OUTPUT_DIR / "label_dispersion_offenders_v2_synthetic.csv"
        offenders.to_csv(off_path, index=False)
        # Mark run as failed QA
        (OUTPUT_DIR / "run_manifest.json").write_text(
            json.dumps(
                {"build_id": build_id, "build_timestamp_utc": build_timestamp_utc, "status": "FAILED", "failure": "QA_MODE_DISPERSION_GATE"},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        raise RuntimeError(f"QA_MODE dispersion_gate triggered as expected. See {off_path} and {fake_path}")

    # Lookups/dims
    logger.info("[1/8] Loading Projects (new_raw_data) ...")
    projects = load_projects_v2()

    logger.info("[2/8] Loading Developers registry (new_raw_data) ...")
    developers = load_developers_registry()

    # Brand resolver setup
    consolidation_json = LOOKUPS_DIR / "developer_brand_consolidation.json"
    building_devs_json = LOOKUPS_DIR / "building_developers.json"

    # Seed umbrella map deterministically from consolidation mapping (writes umbrella_map.json + audit json)
    # Note: umbrella labels here use consolidation brand keys by default; you can provide aliases later if you prefer
    # top50-style labels (e.g., "Emaar" -> "Emaar Properties").
    umbrella_audit = seed_umbrella_map_from_consolidation(
        developers=developers,
        projects=projects,
        consolidation_json=consolidation_json,
        out_path=Path(CFG.umbrella_map_path),
        umbrella_label_aliases={},  # keep as-is unless you add an explicit mapping layer
    )

    umbrella_payload = json.loads(Path(CFG.umbrella_map_path).read_text(encoding="utf-8"))
    umbrella_by_id = (umbrella_payload.get("map") or {}) if isinstance(umbrella_payload, dict) else {}
    pb_path = Path(CFG.public_brands_path)
    if not pb_path.exists():
        raise RuntimeError(f"Missing curated public brand universe: {pb_path}")
    pb = json.loads(pb_path.read_text(encoding='utf-8'))
    approved = [str(x).strip() for x in (pb.get("brands") or []) if str(x).strip()]
    if not approved:
        raise RuntimeError(f"public_brands.json has empty brands list: {pb_path}")

    logger.info("[3/8] Compiling developer_id -> public brand map ...")
    dev_id_to_brand, brand_unmatched = compile_developer_brand_map_by_id(developers, projects, allowed_brands=approved)

    aliases = BrandResolver.build_brand_aliases(consolidation_json, building_devs_json)
    aliases = {b: aliases.get(b, []) for b in approved}

    # Alias -> canonical brand mapping (separate from output universe)
    pab_path = Path(CFG.public_brand_aliases_path)
    if pab_path.exists():
        pab = json.loads(pab_path.read_text(encoding="utf-8"))
        amap = pab.get("aliases") or {}
        if isinstance(amap, dict):
            for a, canon in amap.items():
                a = str(a).strip()
                canon = str(canon).strip()
                if not a or not canon:
                    continue
                if canon not in approved:
                    raise RuntimeError(f"public_brand_aliases maps to non-canonical brand not in public_brands.json: {canon} (alias={a})")
                aliases.setdefault(canon, []).append(a)
    # Override maps (coverage closure)
    overrides_pn = BrandResolver.load_override_csv(LOOKUPS_DIR / "brand_overrides_project_number.csv", "project_number", "developer_brand")
    overrides_mp = BrandResolver.load_override_csv(LOOKUPS_DIR / "brand_overrides_master_project.csv", "master_project_en", "developer_brand")
    # Validate overrides target brands
    bad_override = [b for b in set(list(overrides_pn.values()) + list(overrides_mp.values())) if b and b not in approved]
    if bad_override:
        raise RuntimeError(f"Overrides contain brands not in approved list: {sorted(set(bad_override))[:20]}")

    resolver = BrandResolver(
        approved_brands=approved,
        brand_aliases=aliases,
        precedence=approved,
        dev_id_to_brand=dev_id_to_brand,
        overrides_project_number=overrides_pn,
        overrides_master_project=overrides_mp,
    )

    logger.info("[4/8] Loading area lookup + building area_geo from KML ...")
    lkp_areas = load_lkp_areas()
    area_geo, kml_stats = build_area_geo_from_kml(lkp_areas)

    logger.info("[5/8] Loading transaction groups + sale index ...")
    sales_group_ids = load_transaction_groups_sales_ids()
    sale_index = load_sale_index()

    # Fact aggregation
    logger.info("[6/8] Aggregating Transactions (area_id + developer_brand) ...")
    # Build suspicious label norms for preventing community/asset strings from becoming developer_brand_label
    lkp_areas_for_labels = load_lkp_areas()

    def _norm_label_any(x: object) -> str:
        s = _norm_ws(x)
        s = re.sub(r"[^\w\s]", " ", s, flags=re.UNICODE)
        s = re.sub(r"\s+", " ", s, flags=re.UNICODE).strip()
        return s

    # Build suspicious area name norms from BOTH EN + AR (Arabic leak prevention)
    suspicious_area_name_norms = set()
    if "name_en" in lkp_areas_for_labels.columns:
        suspicious_area_name_norms.update([_norm_label_any(n) for n in lkp_areas_for_labels["name_en"].astype(str).tolist() if _norm_label_any(n)])
    if "name_ar" in lkp_areas_for_labels.columns:
        suspicious_area_name_norms.update([_norm_label_any(n) for n in lkp_areas_for_labels["name_ar"].astype(str).tolist() if _norm_label_any(n)])

    # Blocked labels list (EN+AR) that must never become developer_brand_label
    blocked_path = Path(CFG.blocked_brand_labels_path)
    blocked_norms = []
    if blocked_path.exists():
        blocked = json.loads(blocked_path.read_text(encoding="utf-8"))
        blocked_norms = [_norm_label_any(x) for x in (blocked.get("labels") or []) if _norm_label_any(x)]
    suspicious_label_tokens = {
        # project/community terms (do not survive as final labels)
        "PROJECT", "PROJECTS",
        "RESIDENTIAL",
        "VILLAGE",
        "SQUARE",
        "PHASE",
        "ESTATE",
        "HILLS",
        "HARBOUR", "HARBOR",
        "CREEK",
        "LAGOON", "LAGOONS",
        "COURT", "COURTS",
        # common place terms (used as supplemental; weak tokens are handled in the detector)
        "BAY",
        "CITYWALK",
        # legacy coverage
        "PALM", "MARINA", "DOWNTOWN", "JUMEIRAH",
        "WAVE", "WAVES",
        "HEIGHTS", "GARDENS",
        "RESIDENCE", "RESIDENCES", "TOWER", "TOWERS",
        "DISTRICT",
        "ISLAND", "ISLANDS",
        "PARK",
        "VIEW", "VIEWS",
    }

    # Brand overrides (legacy/noisy): developer_id -> canonical brand label (SOFT-validated)
    # - Apply only if canonical OR (optionally) rewrite alias->canonical using public_brand_aliases.json.
    # - Never fail build on this file; always emit run-bundle audit artifacts for skipped/rewrite rows.
    ENABLE_ALIAS_REWRITES_FOR_BRAND_OVERRIDES = False

    # Build alias->canonical map (case-insensitive) for suggestions/optional rewrites
    alias_to_canon_ci: Dict[str, str] = {}
    try:
        pab_path = Path(CFG.public_brand_aliases_path)
        if pab_path.exists():
            pab = json.loads(pab_path.read_text(encoding="utf-8"))
            amap = pab.get("aliases") or {}
            if isinstance(amap, dict):
                for a, canon in amap.items():
                    a = _norm_ws(a)
                    canon = _norm_ws(canon)
                    if not a or not canon:
                        continue
                    alias_to_canon_ci[a.upper()] = canon
    except Exception:
        alias_to_canon_ci = {}

    id_brand_overrides: Dict[str, str] = {}
    skipped_rows: List[Dict[str, str]] = []
    rewrite_rows: List[Dict[str, str]] = []
    try:
        ovr_path = LOOKUPS_DIR / "brand_overrides_developer_id.csv"
        if ovr_path.exists():
            ovr = pd.read_csv(ovr_path, low_memory=False)
            if {"developer_id", "developer_brand"} <= set(ovr.columns):
                for _, r in ovr.iterrows():
                    did = _safe_int_str(r.get("developer_id"))
                    b = _norm_ws(r.get("developer_brand"))
                    if not did or not b:
                        continue
                    raw = {k: ("" if v is None else str(v)) for k, v in dict(r).items()}
                    row_raw = json.dumps(raw, ensure_ascii=False)
                    if b in approved:
                        id_brand_overrides[str(did)] = str(b)
                        continue
                    # Non-canonical: attempt alias suggestion/rewrite
                    canon_sugg = alias_to_canon_ci.get(b.upper(), "")
                    if ENABLE_ALIAS_REWRITES_FOR_BRAND_OVERRIDES and canon_sugg and (canon_sugg in approved):
                        id_brand_overrides[str(did)] = str(canon_sugg)
                        rewrite_rows.append(
                            {
                                "developer_id": str(did),
                                "alias_brand": b,
                                "canonical_brand": str(canon_sugg),
                                "row_raw": row_raw,
                            }
                        )
                        continue
                    # Otherwise skip + record
                    skipped_rows.append(
                        {
                            "developer_id": str(did),
                            "override_brand": b,
                            "reason": "NON_CANONICAL_BRAND",
                            "canonical_suggestion": str(canon_sugg) if canon_sugg in approved else "",
                            "row_raw": row_raw,
                        }
                    )
    except Exception as e:
        # Soft-fail: treat as no overrides and still write empty audit artifacts
        logger.warning(f"Could not load/parse brand_overrides_developer_id.csv; continuing without applying it: {e}")

    # Apply any accepted overrides to dev_id_to_brand so BrandResolver LEGAL_MAP uses them too.
    for did, b in id_brand_overrides.items():
        dev_id_to_brand[str(did)] = str(b)

    # Run-bundle artifacts (always written, even if empty)
    skipped_path = OUTPUT_DIR / "noncanonical_brand_overrides_skipped.csv"
    pd.DataFrame(
        skipped_rows,
        columns=["developer_id", "override_brand", "reason", "canonical_suggestion", "row_raw"],
    ).to_csv(skipped_path, index=False)

    rewrites_path = OUTPUT_DIR / "brand_override_alias_rewrites.csv"
    pd.DataFrame(
        rewrite_rows,
        columns=["developer_id", "alias_brand", "canonical_brand", "row_raw"],
    ).to_csv(rewrites_path, index=False)

    # Required summary print (end-of-run visibility)
    if skipped_rows:
        top10 = (
            pd.DataFrame(skipped_rows)["override_brand"]
            .value_counts()
            .head(10)
            .to_dict()
        )
    else:
        top10 = {}
    logger.info(f"Non-canonical brand_overrides_developer_id.csv rows skipped: {len(skipped_rows):,}")
    if top10:
        logger.info(f"Top skipped override_brand values (top10): {top10}")

    entity_owner_overrides: Dict[str, str] = {}
    try:
        own_path = LOOKUPS_DIR / "entity_owner_overrides_developer_id.csv"
        if own_path.exists():
            own = pd.read_csv(own_path, low_memory=False)
            if {"developer_id", "target_brand"} <= set(own.columns):
                for _, r in own.iterrows():
                    did = _safe_int_str(r.get("developer_id"))
                    b = _norm_ws(r.get("target_brand"))
                    if did and b:
                        if b not in approved:
                            raise RuntimeError(f"entity_owner_overrides_developer_id.csv maps to non-canonical brand not in public_brands.json: {b} (developer_id={did})")
                        if _label_is_authority_or_freezone(b):
                            raise RuntimeError(f"entity_owner_overrides_developer_id.csv must not map to authority/freezone: {b} (developer_id={did})")
                        entity_owner_overrides[str(did)] = str(b)
    except Exception as e:
        raise RuntimeError(f"Failed to load entity_owner_overrides_developer_id.csv: {e}")

    transactions, tx_audits = aggregate_transactions_v2(
        projects=projects,
        developers=developers,
        dev_id_to_brand=dev_id_to_brand,
        umbrella_by_id=umbrella_by_id,
        safe_training_brands=set(approved),
        id_brand_overrides=id_brand_overrides,
        entity_owner_overrides=entity_owner_overrides,
        suspicious_label_tokens=suspicious_label_tokens,
        suspicious_area_name_norms=suspicious_area_name_norms,
        blocked_brand_labels_norms=set(blocked_norms),
        sales_group_ids=sales_group_ids,
        brand_resolver=resolver,
    )

    # ------------------------------------------------------------------
    # Owner override self-check (fail-fast on CONFLICT)
    # ------------------------------------------------------------------
    def _top_key(dist: Dict[str, int]) -> str:
        if not dist:
            return ""
        return sorted(dist.items(), key=lambda kv: kv[1], reverse=True)[0][0]

    owner_path = LOOKUPS_DIR / "entity_owner_overrides_developer_id.csv"
    owner_rows = []
    if owner_path.exists():
        try:
            ovr_df = pd.read_csv(owner_path, low_memory=False)
            if {"developer_id", "target_brand"} <= set(ovr_df.columns):
                for _, r in ovr_df.iterrows():
                    did = _safe_int_str(r.get("developer_id"))
                    tb = _norm_ws(r.get("target_brand"))
                    rs = _norm_ws(r.get("reason"))
                    if did and tb:
                        owner_rows.append({"developer_id": str(did), "target_brand": tb, "reason": rs})
        except Exception as e:
            raise RuntimeError(f"Failed to parse {owner_path}: {e}")

    # evidence from tx_audits (raw tx-level)
    dev_en = tx_audits.get("_dev_legal_en_counts") or {}
    dev_ar = tx_audits.get("_dev_legal_ar_counts") or {}
    dev_umb = tx_audits.get("_dev_umbrella_counts") or {}

    project_tokens = {
        "CITYWALK", "CREEK", "HILLS", "LAGOONS", "PALM",
        "BUSINESS", "BAY", "INTERNATIONAL", "CITY", "VILLAGE",
        "PROJECT", "PROJECTS", "RESIDENTIAL", "PHASE", "ESTATE", "HARBOUR", "HARBOR",
    }

    self_rows = []
    conflicts = 0
    for row in owner_rows:
        did = row["developer_id"]
        target = row["target_brand"]
        legal_en_top = _top_key(dev_en.get(did, {}) or {})
        legal_ar_top = _top_key(dev_ar.get(did, {}) or {})
        umb_top = _top_key(dev_umb.get(did, {}) or {})

        notes = []
        status = "QUESTIONABLE"

        # target must be canonical
        if target not in approved:
            status = "CONFLICT"
            notes.append("override_target_not_canonical_public_brand")

        # authority/freezone must not appear here (detect via top legal strings)
        if _label_is_authority_or_freezone(legal_en_top) or _label_is_authority_or_freezone(legal_ar_top):
            status = "CONFLICT"
            notes.append("legal_name_looks_like_authority_freezone")

        # umbrella evidence
        if umb_top and (umb_top in approved) and (umb_top != target):
            status = "CONFLICT"
            notes.append(f"umbrella_top_conflicts (umbrella_top={umb_top})")
        elif umb_top and (umb_top == target):
            status = "OK"
            notes.append("umbrella_top_matches_override")
        elif not umb_top and (target in approved):
            # lexical support from legal name
            ln = _norm_upper_spaces(legal_en_top or legal_ar_top)
            toks = set([t for t in ln.split() if t])
            if toks & project_tokens:
                status = "OK"
                notes.append("no_umbrella_top_but_legal_contains_project_tokens")
            else:
                status = "QUESTIONABLE"
                notes.append("no_umbrella_top_and_legal_tokens_not_strong")

        if status == "CONFLICT":
            conflicts += 1

        self_rows.append(
            {
                "developer_id": did,
                "legal_name_en_top": legal_en_top,
                "legal_name_ar_top": legal_ar_top,
                "umbrella_top": umb_top,
                "override_target_brand": target,
                "self_check_status": status,
                "notes": ";".join([n for n in notes if n]),
            }
        )

    self_path = OUTPUT_DIR / "owner_override_self_check.csv"
    pd.DataFrame(
        self_rows,
        columns=[
            "developer_id",
            "legal_name_en_top",
            "legal_name_ar_top",
            "umbrella_top",
            "override_target_brand",
            "self_check_status",
            "notes",
        ],
    ).to_csv(self_path, index=False)

    if conflicts > 0:
        raise RuntimeError(f"Owner override self-check failed: {conflicts} CONFLICT rows. See {self_path}")

    # ------------------------------------------------------------------
    # Owner hard assertions (fail-fast) + report artifact
    # ------------------------------------------------------------------
    dev_label_counts = tx_audits.get("_dev_label_counts") or {}
    dev_src_counts = tx_audits.get("_dev_label_source_counts") or {}

    def _top_from_map(mp: Dict[str, int]) -> str:
        if not mp:
            return ""
        return sorted(mp.items(), key=lambda kv: kv[1], reverse=True)[0][0]

    assertions = [
        {"developer_id": "26581366", "expected_label": "Meraas", "expected_source": "OWNER_OVERRIDE_DEVELOPER_ID"},
        {"developer_id": "122", "expected_label": "Nakheel", "expected_source": None},
        {"developer_id": "898", "expected_label": "Dubai Properties", "expected_source": None},
        {"developer_id": "153", "expected_label": "DEVELOPER_ID_153", "expected_source": "ID_FALLBACK_AUTHORITY"},
    ]
    rep = {"build_id": build_id, "assertions": []}
    failed = 0
    for a in assertions:
        did = a["developer_id"]
        exp_lbl = a["expected_label"]
        exp_src = a.get("expected_source")
        top_lbl = _top_from_map(dev_label_counts.get(did, {}) or {})
        top_src = _top_from_map(dev_src_counts.get(did, {}) or {})
        ok = (top_lbl == exp_lbl) and ((exp_src is None) or (top_src == exp_src))
        rep["assertions"].append(
            {
                "developer_id": did,
                "expected_label": exp_lbl,
                "observed_top_label": top_lbl,
                "expected_source": exp_src,
                "observed_top_source": top_src,
                "status": "PASS" if ok else "FAIL",
            }
        )
        if not ok:
            failed += 1

    assertions_path = OUTPUT_DIR / "owner_assertions_report.json"
    assertions_path.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    if failed > 0:
        raise RuntimeError(f"Owner assertions failed: {failed} failing assertions. See {assertions_path}")

    logger.info("[7/8] Aggregating Rents (area_id) ...")
    area_name_to_id = build_area_name_to_id_map_from_transactions()
    # attach map to function (keeps signature stable)
    aggregate_rents_v2._area_name_to_id = area_name_to_id  # type: ignore[attr-defined]
    rents = aggregate_rents_v2()

    # Months from transactions
    all_months = sorted(transactions["year_month"].unique())
    logger.info(f"Date range: {all_months[0]} -> {all_months[-1]} ({len(all_months)} months)")

    logger.info("[8/8] Computing supply (projects) ...")
    supply = compute_supply_v2(projects, all_months)
    completions = compute_units_completing_v2(projects, all_months)

    # Merge (v2: tx + rents + supply + geo + index)
    df = transactions.merge(rents, on=["year_month", "area_id", "bedroom"], how="left")
    df = df.merge(supply, on=["year_month", "area_id"], how="left")
    df = df.merge(completions, on=["year_month", "area_id"], how="left")
    df = df.merge(area_geo, on="area_id", how="left")
    df = attach_area_names(df, lkp_areas)
    df = add_sale_index_features(df, sale_index)

    # Fill safe defaults for counts
    for c in ["rent_count", "supply_units", "supply_buildings", "supply_villas", "active_projects", "units_completing"]:
        if c in df.columns:
            df[c] = df[c].fillna(0).astype(int)

    # Create TFT columns (group_id/time_idx) BEFORE rent audit/filter so artifacts can be grouped at series level.
    df = create_tft_columns_v2(df)

    # ------------------------------------------------------------------
    # Rent imputation + audit (for multi-target training: price + rent)
    # ------------------------------------------------------------------
    if {"median_rent", "median_rent_sqft", "rent_count", "group_id", "area_id"} <= set(df.columns):
        try:
            # Sort for proper ffill/bfill
            df = df.sort_values(["group_id", "time_idx"]).reset_index(drop=True)

            # Identify original missing rent (before any imputation)
            rent_count = pd.to_numeric(df["rent_count"], errors="coerce").fillna(0).astype(int)
            orig_miss_rent = df["median_rent"].isna()
            orig_miss_sqft = df["median_rent_sqft"].isna()
            below_min = rent_count < int(CFG.min_rent_count)
            orig_rent_missing = orig_miss_rent | orig_miss_sqft | below_min

            # Initialize imputation tracking columns
            df["rent_imputed"] = 0
            df["rent_imputation_source"] = pd.NA

            # ------------------------------------------------------------------
            # AUDIT: Capture pre-imputation state
            # ------------------------------------------------------------------
            audit = {
                "policy": {
                    "rent_imputation_strategy": str(CFG.rent_imputation_strategy),
                    "min_rent_count": int(CFG.min_rent_count),
                    "definition": "rent_missing := isna(median_rent) OR isna(median_rent_sqft) OR rent_count < min_rent_count",
                },
                "pre_imputation": {
                    "rows_total": int(len(df)),
                    "rows_rent_missing": int(orig_rent_missing.sum()),
                    "rows_rent_missing_rate": float(orig_rent_missing.mean()) if len(df) else 0.0,
                    "rows_median_rent_missing_rate": float(orig_miss_rent.mean()) if len(df) else 0.0,
                    "rows_median_rent_sqft_missing_rate": float(orig_miss_sqft.mean()) if len(df) else 0.0,
                },
                "notes": [
                    "In V2, rent features are joined at (year_month, area_id, bedroom).",
                    "If rent contracts do not exist for a key, rent_count is 0 and median rent fields remain NaN.",
                    "Imputation strategy 'tiered' applies: (1) ffill+bfill within group, (2) area-level median, (3) global median.",
                ],
            }

            # High-signal breakdowns (pre-imputation)
            if "year_month" in df.columns:
                by_month = (
                    df.assign(_rm=orig_rent_missing)
                    .groupby("year_month", as_index=False)["_rm"]
                    .mean()
                    .rename(columns={"_rm": "rent_missing_rate"})
                    .sort_values("rent_missing_rate", ascending=False)
                )
                audit["worst_months_by_rent_missing_rate_top15"] = by_month.head(15).to_dict(orient="records")
                audit["best_months_by_rent_missing_rate_top15"] = by_month.tail(15).sort_values("rent_missing_rate").to_dict(orient="records")

            for col in ["bedroom", "property_type", "reg_type"]:
                if col in df.columns:
                    by = (
                        df.assign(_rm=orig_rent_missing)
                        .groupby(col, as_index=False)
                        .agg(n=("_rm", "size"), rent_missing_rate=("_rm", "mean"))
                        .sort_values(["rent_missing_rate", "n"], ascending=[False, False])
                    )
                    audit[f"rent_missing_rate_by_{col}"] = by.to_dict(orient="records")

            # Group-level top contributors (by row-count contribution)
            grp = (
                df.assign(_rm=orig_rent_missing)
                .groupby("group_id", as_index=False)
                .agg(
                    n=("_rm", "size"),
                    rent_missing_rows=("_rm", "sum"),
                    rent_missing_rate=("_rm", "mean"),
                    rent_count_zero_rows=("rent_count", lambda s: int((pd.to_numeric(s, errors="coerce").fillna(0).astype(int) == 0).sum())),
                )
                .sort_values(["rent_missing_rows", "n"], ascending=[False, False])
            )
            top50 = grp.head(50)
            (OUTPUT_DIR / "top50_group_ids_missing_rent_v2.csv").write_text(top50.to_csv(index=False), encoding="utf-8")
            audit["top50_group_ids_missing_rent_csv"] = str(OUTPUT_DIR / "top50_group_ids_missing_rent_v2.csv")

            # Save unfiltered copy if requested
            if CFG.write_unfiltered_dataset_copy:
                unfiltered_path = OUTPUT_DIR / "tft_training_data_v2_pre_imputation.csv"
                df.to_csv(unfiltered_path, index=False)
                audit["pre_imputation_csv"] = str(unfiltered_path)

            # ------------------------------------------------------------------
            # IMPUTATION STRATEGY
            # ------------------------------------------------------------------
            strategy = str(CFG.rent_imputation_strategy).strip().lower()

            if strategy == "tiered":
                # TIER 1: ffill + bfill within group (handles internal time gaps)
                # This preserves last observed rent for that specific (area, bedroom, property_type, reg_type) segment
                for rent_col in ["median_rent", "median_rent_sqft"]:
                    before_fill = df[rent_col].isna().sum()
                    df[rent_col] = df.groupby("group_id")[rent_col].transform(lambda x: x.ffill().bfill())
                    after_fill = df[rent_col].isna().sum()
                    filled = before_fill - after_fill
                    logger.info(f"Rent imputation tier 1 (group ffill+bfill) on {rent_col}: filled {filled:,} rows")

                # Mark tier 1 imputed rows
                tier1_imputed = orig_rent_missing & df["median_rent"].notna() & df["median_rent_sqft"].notna()
                df.loc[tier1_imputed, "rent_imputed"] = 1
                df.loc[tier1_imputed, "rent_imputation_source"] = "group_ffill_bfill"

                # TIER 2: area-level median (for groups with zero rent observations)
                # This uses rent from other groups in the same area (different bedroom/property_type)
                still_missing = df["median_rent"].isna() | df["median_rent_sqft"].isna()
                if still_missing.any():
                    # Compute area-level medians from observed (non-imputed) rent
                    area_medians = df.loc[~orig_rent_missing].groupby("area_id").agg(
                        area_median_rent=("median_rent", "median"),
                        area_median_rent_sqft=("median_rent_sqft", "median"),
                    )

                    before_tier2 = still_missing.sum()
                    for rent_col, area_col in [("median_rent", "area_median_rent"), ("median_rent_sqft", "area_median_rent_sqft")]:
                        miss = df[rent_col].isna()
                        if miss.any():
                            area_fill = df.loc[miss, "area_id"].map(area_medians[area_col])
                            df.loc[miss, rent_col] = area_fill

                    still_missing_after_t2 = df["median_rent"].isna() | df["median_rent_sqft"].isna()
                    tier2_filled = before_tier2 - still_missing_after_t2.sum()
                    logger.info(f"Rent imputation tier 2 (area median): filled {tier2_filled:,} rows")

                    # Mark tier 2 imputed rows
                    tier2_imputed = orig_rent_missing & ~tier1_imputed & df["median_rent"].notna() & df["median_rent_sqft"].notna()
                    df.loc[tier2_imputed, "rent_imputed"] = 1
                    df.loc[tier2_imputed, "rent_imputation_source"] = "area_median"

                # TIER 3: global median (last resort for areas with no rent data)
                still_missing = df["median_rent"].isna() | df["median_rent_sqft"].isna()
                if still_missing.any():
                    global_median_rent = df.loc[~orig_rent_missing, "median_rent"].median()
                    global_median_sqft = df.loc[~orig_rent_missing, "median_rent_sqft"].median()

                    before_tier3 = still_missing.sum()
                    df.loc[df["median_rent"].isna(), "median_rent"] = global_median_rent
                    df.loc[df["median_rent_sqft"].isna(), "median_rent_sqft"] = global_median_sqft

                    still_missing_after_t3 = df["median_rent"].isna() | df["median_rent_sqft"].isna()
                    tier3_filled = before_tier3 - still_missing_after_t3.sum()
                    logger.info(f"Rent imputation tier 3 (global median): filled {tier3_filled:,} rows")

                    # Mark tier 3 imputed rows
                    tier3_imputed = orig_rent_missing & (df["rent_imputation_source"].isna())
                    df.loc[tier3_imputed & df["median_rent"].notna(), "rent_imputed"] = 1
                    df.loc[tier3_imputed & df["median_rent"].notna(), "rent_imputation_source"] = "global_median"

                # Summary
                imputation_summary = df.groupby("rent_imputation_source", dropna=False).size().to_dict()
                audit["imputation_summary"] = {str(k): int(v) for k, v in imputation_summary.items()}
                audit["post_imputation"] = {
                    "rows_rent_still_missing": int((df["median_rent"].isna() | df["median_rent_sqft"].isna()).sum()),
                    "rows_imputed_total": int(df["rent_imputed"].sum()),
                    "imputation_rate": float(df["rent_imputed"].mean()) if len(df) else 0.0,
                }

                # Hard guarantee: training-ready (no NaNs in rent fields after tiered imputation)
                if df["median_rent"].isna().any() or df["median_rent_sqft"].isna().any():
                    remaining = (df["median_rent"].isna() | df["median_rent_sqft"].isna()).sum()
                    raise RuntimeError(f"Tiered imputation expected to eliminate all NaNs but {remaining} remain.")

                logger.info(
                    f"Rent imputation complete (strategy=tiered): "
                    f"{int(df['rent_imputed'].sum()):,}/{len(df):,} rows imputed ({df['rent_imputed'].mean()*100:.2f}%)"
                )

            elif strategy == "drop":
                # Legacy: drop rows where rent is missing (use only for rent-only models)
                before = len(df)
                dropped = int(orig_rent_missing.sum())
                df = df.loc[~orig_rent_missing].copy()
                df = create_tft_columns_v2(df)
                audit["post_drop"] = {
                    "rows_dropped": dropped,
                    "rows_remaining": len(df),
                    "drop_rate": float(dropped / max(before, 1)),
                }
                logger.info(f"Rent coverage filter applied (strategy=drop): dropped {dropped:,}/{before:,} rows. Remaining: {len(df):,}")

                if df["median_rent"].isna().any() or df["median_rent_sqft"].isna().any():
                    raise RuntimeError("Drop strategy expected to eliminate NaNs but did not.")

            elif strategy == "none":
                # Leave NaNs as-is — training script must handle
                audit["post_imputation"] = {"note": "No imputation applied (strategy=none). Training script must handle NaNs."}
                logger.info("Rent imputation skipped (strategy=none). NaNs remain in median_rent/median_rent_sqft.")

            else:
                raise ValueError(f"Unknown rent_imputation_strategy: {CFG.rent_imputation_strategy}")

            # Write audit
            if CFG.write_rent_missingness_audit:
                (OUTPUT_DIR / "rent_missingness_audit_v2.json").write_text(
                    json.dumps(audit, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )

        except Exception as e:
            raise RuntimeError(f"Rent imputation/audit failed: {e}") from e

    # ------------------------------------------------------------------
    # Top-50 reporting flags (UI only; never used to force-map labels)
    # ------------------------------------------------------------------
    top50_path = Path(CFG.top50_2025_path)
    top50 = json.loads(top50_path.read_text(encoding="utf-8")) if top50_path.exists() else {"brands": [], "canonicalisation_rules": {}}
    top50_list = [str(x) for x in (top50.get("brands") or [])]
    top50_set = set([x.strip() for x in top50_list if x and str(x).strip()])
    rules = top50.get("canonicalisation_rules") or {}
    aliases_map = (rules.get("aliases") or {}) if isinstance(rules, dict) else {}
    # case-insensitive alias map
    aliases_ci = {str(k).strip().upper(): str(v).strip() for k, v in dict(aliases_map).items() if str(k).strip() and str(v).strip()}
    top50_ci = {str(b).strip().upper(): str(b).strip() for b in top50_list if str(b).strip()}

    def _canon_label(x: object) -> str:
        s = _norm_ws(x)
        if not s:
            return ""
        # strict aliases
        if s in aliases_map:
            return str(aliases_map[s]).strip()
        # whitespace collapse for safety
        s2 = re.sub(r"\s+", " ", s).strip()
        if s2 in aliases_map:
            return str(aliases_map[s2]).strip()
        u = s2.upper()
        if u in aliases_ci:
            return str(aliases_ci[u]).strip()
        # if it's a top50 label but different casing, canonicalize casing to top50
        if u in top50_ci:
            return top50_ci[u]
        return s2

    if "developer_brand_label" in df.columns:
        # IMPORTANT: canonicalisation rules are reporting-only. Do NOT rewrite developer_brand_label.
        _top50_match_label = df["developer_brand_label"].map(_canon_label)
        df["is_top50_2025"] = _top50_match_label.isin(top50_set).astype(int)
        df["top50_2025_brand"] = _top50_match_label.where(df["is_top50_2025"] == 1, pd.NA)

    # ------------------------------------------------------------------
    # Suffix leakage gate on developer_brand_label (plus HOLDING policy enforcement)
    # ------------------------------------------------------------------
    # - Corporate suffix regex applies to all labels.
    # - HOLDING/HOLDINGS is allowed ONLY for canonical PUBLIC_BRAND outputs (policy v2).
    suf_pat = re.compile(CFG.corporate_suffix_regex)
    holding_pat = re.compile(r"(?i)\bHOLDINGS?\b")
    if {"developer_brand_label", "developer_brand_label_source", "transaction_count"} <= set(df.columns):
        lbl = df["developer_brand_label"].astype(str)
        src = df["developer_brand_label_source"].astype(str).str.strip()
        txc = pd.to_numeric(df["transaction_count"], errors="coerce").fillna(0).astype(int)

        corporate_invalid = lbl.str.contains(suf_pat, na=False)
        holding_hit = lbl.str.contains(holding_pat, na=False)
        holding_invalid = holding_hit & (src != "PUBLIC_BRAND")

        # Required audit artifact
        holding_audit = {
            "policy": "label_policy_v2",
            "holding_total_tx": int(txc.loc[holding_hit].sum()) if holding_hit.any() else 0,
            "holding_by_source_tx": (
                df.loc[holding_hit].groupby("developer_brand_label_source")["transaction_count"].sum().sort_values(ascending=False).to_dict()
                if holding_hit.any()
                else {}
            ),
            "holding_invalid_total_tx": int(txc.loc[holding_invalid].sum()) if holding_invalid.any() else 0,
            "holding_invalid_by_source_tx": (
                df.loc[holding_invalid].groupby("developer_brand_label_source")["transaction_count"].sum().sort_values(ascending=False).to_dict()
                if holding_invalid.any()
                else {}
            ),
        }
        (OUTPUT_DIR / "holding_policy_audit_v2.json").write_text(
            json.dumps(holding_audit, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        invalid = corporate_invalid | holding_invalid
        if invalid.any():
            bad = (
                df.loc[invalid]
                .groupby("developer_brand_label")["transaction_count"]
                .sum()
                .sort_values(ascending=False)
                .head(500)
                .to_dict()
            )
            report_path = OUTPUT_DIR / "suffix_leakage_report_v2.json"
            report_path.write_text(
                json.dumps(
                    {
                        "suffix_leakage_by_tx_count": bad,
                        "holding_policy_invalid_total_tx": int(txc.loc[holding_invalid].sum()) if holding_invalid.any() else 0,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            raise RuntimeError(f"Suffix/HOLDING leakage detected in developer_brand_label. See {report_path}")

    # ------------------------------------------------------------------
    # Baseline snapshot + delta report + SPV review queue + umbrella suggestions
    # ------------------------------------------------------------------
    # NOTE: run identity and per-build output folder are created at the start of main().
    # Build dev metrics from tx_audits (developer_id -> year_month -> (tx,value))
    dev_monthly = tx_audits.get("_dev_monthly") or {}
    max_ym = tx_audits.get("_max_year_month") or df["year_month"].max()
    max_ym = str(max_ym)
    max_dt = pd.to_datetime(max_ym + "-01")
    cutoff_12m = (max_dt - pd.DateOffset(months=11)).strftime("%Y-%m")

    def sum_dev_metrics(months_map: Dict[str, Tuple[int, float]], *, since_ym: Optional[str] = None) -> Tuple[int, float]:
        tx = 0
        val = 0.0
        for ym, (t, v) in months_map.items():
            if since_ym and str(ym) < since_ym:
                continue
            tx += int(t)
            val += float(v)
        return tx, val

    current_dev_metrics = {}
    for did, mm in dev_monthly.items():
        tx_all, val_all = sum_dev_metrics(mm)
        tx_12, val_12 = sum_dev_metrics(mm, since_ym=cutoff_12m)
        current_dev_metrics[str(did)] = {
            "tx_count_all_time": int(tx_all),
            "sales_value_all_time": float(val_all),
            "tx_count_last_12m": int(tx_12),
            "sales_value_last_12m": float(val_12),
        }

    current_snapshot = {
        "build_id": build_id,
        "build_timestamp_utc": build_timestamp_utc,
        "max_year_month": max_ym,
        "cutoff_last_12m": cutoff_12m,
        "developer_ids_seen": sorted(list(set(tx_audits.get("_run_sets_raw", {}).get("developer_ids_seen", [])))),
        "area_ids_seen": sorted(list(set(tx_audits.get("_run_sets_raw", {}).get("area_ids_seen", [])))),
        "project_numbers_seen": sorted(list(set(tx_audits.get("_run_sets_raw", {}).get("project_numbers_seen", [])))),
        "developer_metrics": current_dev_metrics,
    }

    baseline_path = Path(CFG.baseline_snapshot_path)
    baseline = None
    if baseline_path.exists():
        try:
            baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        except Exception:
            baseline = None

    # Delta report vs last successful baseline
    delta = {
        "note": "Delta vs last successful baseline (if present).",
        "build_id": build_id,
        "build_timestamp_utc": build_timestamp_utc,
        "max_year_month": max_ym,
        "cutoff_last_12m": cutoff_12m,
    }
    if baseline and isinstance(baseline, dict):
        b_dev = set(baseline.get("developer_ids_seen") or [])
        b_area = set(baseline.get("area_ids_seen") or [])
        b_proj = set(baseline.get("project_numbers_seen") or [])
        c_dev = set(current_snapshot["developer_ids_seen"])
        c_area = set(current_snapshot["area_ids_seen"])
        c_proj = set(current_snapshot["project_numbers_seen"])
        new_dev = sorted(list(c_dev - b_dev))
        new_area = sorted(list(c_area - b_area))
        new_proj = sorted(list(c_proj - b_proj))
        delta["new_developer_ids"] = new_dev
        delta["new_area_ids"] = new_area
        delta["new_project_numbers"] = new_proj
    else:
        delta["new_developer_ids"] = []
        delta["new_area_ids"] = []
        delta["new_project_numbers"] = []

    # Rank new developer_ids by last12m metrics
    dev_name_en_map = developers.set_index("developer_id")["developer_name_en"].to_dict()
    ranks = []
    for did in delta.get("new_developer_ids") or []:
        m = current_dev_metrics.get(str(did), {})
        ranks.append(
            {
                "developer_id": str(did),
                "developer_legal_name_en": dev_name_en_map.get(str(did), ""),
                **m,
                "is_umbrella_mapped": int(str(did) in umbrella_by_id),
            }
        )
    ranks = sorted(ranks, key=lambda r: (r.get("sales_value_last_12m", 0.0), r.get("tx_count_last_12m", 0)), reverse=True)
    delta["new_developer_ids_ranked"] = ranks[:500]

    # Ensure delta CSV always has a header even when empty (usability requirement)
    delta_csv_columns = [
        "developer_id",
        "developer_legal_name_en",
        "tx_count_all_time",
        "sales_value_all_time",
        "tx_count_last_12m",
        "sales_value_last_12m",
        "is_umbrella_mapped",
    ]

    # Tripwire: new dev_id in top25 by last12m value/count AND not umbrella-mapped
    all_rank = []
    for did, m in current_dev_metrics.items():
        all_rank.append({"developer_id": did, **m, "developer_legal_name_en": dev_name_en_map.get(did, ""), "is_umbrella_mapped": int(did in umbrella_by_id)})
    top25_value = sorted(all_rank, key=lambda r: r.get("sales_value_last_12m", 0.0), reverse=True)[:25]
    top25_tx = sorted(all_rank, key=lambda r: r.get("tx_count_last_12m", 0), reverse=True)[:25]
    alerts = []
    for lst, kind in [(top25_value, "sales_value_last_12m"), (top25_tx, "tx_count_last_12m")]:
        for r in lst:
            if (r["developer_id"] in (delta.get("new_developer_ids") or [])) and (r["is_umbrella_mapped"] == 0):
                alerts.append({"rank_kind": kind, **r})
    delta["tripwire_alerts"] = alerts

    delta_json_path = OUTPUT_DIR / "delta_report_v2.json"
    delta_csv_path = OUTPUT_DIR / "delta_report_v2_new_developers.csv"

    # SPV candidates ranked (conservative token heuristic)
    # SPV/community/asset tokens (boundary-safe via tokenization)
    spv_tokens = {
        "CITY", "BAY", "PALM", "MARINA", "DOWNTOWN", "JUMEIRAH",
        "COURT", "COURTS",
        "ESTATE", "HARBOUR", "HARBOR", "CREEK", "HEIGHTS", "GARDENS", "HILLS",
        "LAGOON", "LAGOONS", "PHASE",
        "RESIDENCE", "RESIDENCES", "TOWER", "TOWERS",
        "DISTRICT",
        "ISLAND", "ISLANDS",
        "PARK",
        "VIEW", "VIEWS",
    }
    spv_rows = []
    for did, m in current_dev_metrics.items():
        name = dev_name_en_map.get(did, "")
        toks = [t.upper() for t in _tokenize_preserve(name)]
        hits = sorted(list(set([t for t in toks if t in spv_tokens])))
        if not hits:
            continue
        spv_rows.append(
            {
                "developer_id": did,
                "developer_legal_name_en": name,
                "spv_token_hits": "|".join(hits),
                "is_umbrella_mapped": int(did in umbrella_by_id),
                **m,
            }
        )
    spv_df = pd.DataFrame(spv_rows).sort_values(["sales_value_last_12m", "tx_count_last_12m"], ascending=[False, False]) if spv_rows else pd.DataFrame(
        columns=["developer_id", "developer_legal_name_en", "spv_token_hits", "is_umbrella_mapped", "tx_count_all_time", "sales_value_all_time", "tx_count_last_12m", "sales_value_last_12m"]
    )
    spv_path = OUTPUT_DIR / "spv_candidates_ranked.csv"
    spv_df.to_csv(spv_path, index=False)

    # ------------------------------------------------------------------
    # Label dispersion report + hard gate (fake developer labels)
    # ------------------------------------------------------------------
    label_dev_counts = tx_audits.get("_label_dev_counts") or {}
    label_area_counts = tx_audits.get("_label_area_counts") or {}
    label_master_samples = tx_audits.get("_label_master_samples") or {}
    label_building_samples = tx_audits.get("_label_building_samples") or {}

    disp_rows = []
    for label, dev_map in label_dev_counts.items():
        tx_count = int(sum(dev_map.values()))
        unique_devs = int(len(dev_map))
        top_devs = sorted(dev_map.items(), key=lambda kv: kv[1], reverse=True)[:5]
        top_share = float(top_devs[0][1] / max(tx_count, 1)) if top_devs else 0.0
        area_map = label_area_counts.get(label, {})
        unique_areas = int(len(area_map))
        top_area = sorted(area_map.items(), key=lambda kv: kv[1], reverse=True)[:1]
        top_area_share = float(top_area[0][1] / max(sum(area_map.values()), 1)) if top_area else 0.0
        mp = label_master_samples.get(label, {})
        bn = label_building_samples.get(label, {})
        mp_s = [k for k, _ in sorted(mp.items(), key=lambda kv: kv[1], reverse=True)[:5]]
        bn_s = [k for k, _ in sorted(bn.items(), key=lambda kv: kv[1], reverse=True)[:5]]
        disp_rows.append(
            {
                "label": label,
                "tx_count": tx_count,
                "unique_developer_ids": unique_devs,
                "top_developer_id_share": round(top_share, 4),
                "top_5_developer_ids": "|".join([f"{d}:{c}" for d, c in top_devs]),
                "unique_areas": unique_areas,
                "top_area_share": round(top_area_share, 4),
                "example_master_project_en_samples": " | ".join(mp_s),
                "example_building_name_en_samples": " | ".join(bn_s),
            }
        )

    disp_df = pd.DataFrame(disp_rows).sort_values(["tx_count", "unique_developer_ids"], ascending=[False, False]) if disp_rows else pd.DataFrame(
        columns=[
            "label","tx_count","unique_developer_ids","top_developer_id_share","top_5_developer_ids","unique_areas","top_area_share",
            "example_master_project_en_samples","example_building_name_en_samples"
        ]
    )
    disp_path = OUTPUT_DIR / "label_dispersion_report_v2.csv"
    disp_df.to_csv(disp_path, index=False)

    # Presence check for specific historically-problematic fake labels (grep-like proof)
    presence_terms = ["SKY COURTS", "BUSINESS BAY", "THE PALM JUMEIRAH", "INTERNATIONAL CITY"]
    presence_rows = []
    labels_upper = set([str(x).upper() for x in disp_df["label"].astype(str).tolist()]) if not disp_df.empty else set()
    for t in presence_terms:
        presence_rows.append({"term": t, "present": int(t.upper() in labels_upper)})
    (OUTPUT_DIR / "developer_brand_label_presence_check_v2.csv").write_text(
        pd.DataFrame(presence_rows).to_csv(index=False),
        encoding="utf-8",
    )

    # ------------------------------------------------------------------
    # Project/area label survival hard gate (post-resolution)
    # Fail if ANY project/area-like label survives as a FINAL developer_brand_label with tx_count>=1000.
    # This gate is intentionally computed from final labels (dispersion table), not from raw legal strings.
    # ------------------------------------------------------------------
    label_dev_counts = tx_audits.get("_label_dev_counts") or {}
    label_source_counts = tx_audits.get("_label_source_counts") or {}

    blocked_norms_set = set([_norm_label_any(x) for x in (blocked_norms or []) if _norm_label_any(x)])
    suspicious_area_norms_set = set([_norm_label_any(x) for x in (suspicious_area_name_norms or set()) if _norm_label_any(x)])
    suspicious_tokens_set = set([str(x).upper() for x in (suspicious_label_tokens or set()) if str(x).strip()])

    project_area_off = []
    for _, r in disp_df.iterrows():
        lbl = str(r.get("label", "")).strip()
        txc = int(r.get("tx_count", 0))
        if txc < 1000:
            continue
        if not _label_is_project_or_area(
            lbl,
            suspicious_label_tokens=suspicious_tokens_set,
            suspicious_area_name_norms=suspicious_area_norms_set,
            blocked_norms=blocked_norms_set,
        ):
            continue
        # pick top developer_id if available
        dev_map = label_dev_counts.get(lbl, {}) or {}
        dev_id = ""
        if dev_map:
            dev_id = sorted(dev_map.items(), key=lambda kv: kv[1], reverse=True)[0][0]
        src_map = label_source_counts.get(lbl, {}) or {}
        src = ""
        if src_map:
            src = sorted(src_map.items(), key=lambda kv: kv[1], reverse=True)[0][0]
        project_area_off.append(
            {"label": lbl, "tx_count": txc, "developer_id": dev_id, "label_source": src}
        )

    # New required artifact (primary)
    proj_path = OUTPUT_DIR / "project_area_labels_surviving_gate.csv"
    proj_df = pd.DataFrame(
        project_area_off,
        columns=["label", "tx_count", "developer_id", "label_source"],
    )
    if not proj_df.empty:
        proj_df = proj_df.sort_values(["tx_count"], ascending=[False])
    proj_df.to_csv(proj_path, index=False)

    # Back-compat artifact name (kept for checklists; content matches the new gate)
    off_path2 = OUTPUT_DIR / "suspicious_label_gate_offenders.csv"
    proj_df.to_csv(off_path2, index=False)

    if len(proj_df) > 0:
        raise RuntimeError(f"Project/area labels survived final label resolution. See {proj_path}")

    # ------------------------------------------------------------------
    # High-impact suspicious labels (non-top50) ranked
    # ------------------------------------------------------------------
    # Build normalized area names for matching
    try:
        lkp_areas2 = load_lkp_areas()
        area_name_set = set([_norm_upper_spaces(n) for n in lkp_areas2["name_en"].astype(str).tolist() if _norm_upper_spaces(n)])
    except Exception:
        area_name_set = set()

    susp_rows = []
    for did, m in current_dev_metrics.items():
        did = str(did)
        legal = dev_name_en_map.get(did, "")
        label = normalize_legal_name_to_label(legal, lang="en") or f"DEVELOPER_ID_{did}"
        canon_label = _canon_label(label)
        is_top50 = int(canon_label in top50_set)
        if is_top50 == 1:
            continue
        toks = [t.upper() for t in _tokenize_preserve(label)]
        token_hits = sorted(list(set([t for t in toks if t in spv_tokens])))
        matches_area = int(_norm_upper_spaces(label) in area_name_set) if area_name_set else 0
        # label appears as a top master project for this dev
        mc = (tx_audits.get("_dev_master_counts") or {}).get(did, {})
        top_master = ""
        top_master_share = 0.0
        if mc:
            tot = sum(mc.values())
            k, v = sorted(mc.items(), key=lambda kv: kv[1], reverse=True)[0]
            top_master = str(k)
            top_master_share = float(v / max(tot, 1))
        label_in_master = int(_norm_upper_spaces(label) == _norm_upper_spaces(top_master)) if top_master else 0

        if not (token_hits or matches_area or label_in_master):
            continue

        susp_rows.append(
            {
                "developer_id": did,
                "label": label,
                "token_hits": "|".join(token_hits),
                "matches_area_name_en": matches_area,
                "label_equals_top_master_project_en": label_in_master,
                "top_master_project_en": top_master,
                "top_master_project_share": round(top_master_share, 4),
                "sales_value_last_12m": m["sales_value_last_12m"],
                "tx_count_last_12m": m["tx_count_last_12m"],
                "tx_count_all_time": m["tx_count_all_time"],
            }
        )

    susp_df = pd.DataFrame(susp_rows).sort_values(["sales_value_last_12m", "tx_count_last_12m", "tx_count_all_time"], ascending=[False, False, False]) if susp_rows else pd.DataFrame(
        columns=[
            "developer_id","label","token_hits","matches_area_name_en","label_equals_top_master_project_en","top_master_project_en","top_master_project_share",
            "sales_value_last_12m","tx_count_last_12m","tx_count_all_time"
        ]
    )
    susp_path = OUTPUT_DIR / "high_impact_non_top50_suspicious_labels.csv"
    susp_df.to_csv(susp_path, index=False)

    # Hard-fail gate: catches SKY COURTS / BUSINESS BAY / INTERNATIONAL CITY type fake labels.
    # Do NOT blanket-exempt allowlisted labels. Only allow exemption for canonical training brands,
    # and NEVER exempt labels that look like project/building/community/authority strings.
    safe_training_brands = set(approved) if isinstance(approved, list) else set()

    def _gate_suspicious(label: str) -> bool:
        if not label:
            return False
        toks = [t.upper() for t in _tokenize_preserve(label)]
        # non-overridable denylist tokens (includes WAVES/COURTS/BAY/CITY per spec)
        deny = {
            "WAVE", "WAVES",
            "COURT", "COURTS",
            "BAY", "CITY",
            "PALM", "MARINA", "DOWNTOWN", "JUMEIRAH",
            "TOWER", "TOWERS",
            "RESIDENCE", "RESIDENCES",
            "PHASE", "ESTATE", "HARBOUR", "HARBOR", "CREEK", "LAGOON", "LAGOONS", "ISLAND", "ISLANDS",
            "DMCC",
        }
        if any(t in deny for t in toks):
            return True
        return False

    is_exempt = disp_df["label"].astype(str).isin(safe_training_brands)
    is_suspicious = disp_df["label"].astype(str).map(_gate_suspicious)
    offenders = disp_df[
        (disp_df["tx_count"].astype(int) >= 1000)
        & (disp_df["unique_developer_ids"].astype(int) >= 10)
        & (disp_df["top_developer_id_share"].astype(float) <= 0.30)
        & ((~is_exempt) | (is_suspicious))
    ]
    if len(offenders) > 0:
        off_path = OUTPUT_DIR / "label_dispersion_offenders_v2.csv"
        offenders.head(200).to_csv(off_path, index=False)
        raise RuntimeError(
            f"Fake developer label dispersion detected. See {off_path} and {disp_path}. "
            f"Also review {susp_path}."
        )

    # ------------------------------------------------------------------
    # High-impact SPV outliers (non-top50) queue
    # ------------------------------------------------------------------
    dev_master_counts = tx_audits.get("_dev_master_counts") or {}
    dev_area_counts = tx_audits.get("_dev_area_counts") or {}
    dev_examples = tx_audits.get("_dev_text_examples_top5") or {}
    # Projects units sum by developer_id
    p_units = projects[["developer_id", "no_of_units"]].copy()
    p_units["developer_id"] = p_units["developer_id"].astype(str).str.strip()
    p_units["no_of_units"] = pd.to_numeric(p_units["no_of_units"], errors="coerce").fillna(0).astype(int)
    units_by_dev = p_units.groupby("developer_id")["no_of_units"].sum().to_dict()

    # Build top50 set for tagging
    top50_path = Path(CFG.top50_2025_path)
    top50 = json.loads(top50_path.read_text(encoding="utf-8")) if top50_path.exists() else {"brands": [], "canonicalisation_rules": {}}
    top50_list = [str(x) for x in (top50.get("brands") or [])]
    top50_set = set([x.strip() for x in top50_list if x and str(x).strip()])
    # use same reporting-only canonicaliser
    rules = top50.get("canonicalisation_rules") or {}
    aliases_map = (rules.get("aliases") or {}) if isinstance(rules, dict) else {}
    aliases_ci = {str(k).strip().upper(): str(v).strip() for k, v in dict(aliases_map).items() if str(k).strip() and str(v).strip()}
    top50_ci = {str(b).strip().upper(): str(b).strip() for b in top50_list if str(b).strip()}

    def canon_top50(x: object) -> str:
        s = _norm_ws(x)
        if not s:
            return ""
        if s in aliases_map:
            return str(aliases_map[s]).strip()
        s2 = re.sub(r"\s+", " ", s).strip()
        if s2 in aliases_map:
            return str(aliases_map[s2]).strip()
        u = s2.upper()
        if u in aliases_ci:
            return str(aliases_ci[u]).strip()
        if u in top50_ci:
            return top50_ci[u]
        return s2

    # Build dev-level outlier rows using current_dev_metrics
    outlier_rows = []
    for did, m in current_dev_metrics.items():
        did = str(did)
        legal = dev_name_en_map.get(did, "")
        umb = umbrella_by_id.get(did, "")
        umb_label = umb if umb else ""
        is_umb = int(bool(umb))
        # brand label is normalized legal identity for this view (no collapsing)
        brand_label = normalize_legal_name_to_label(legal, lang="en") or f"DEVELOPER_ID_{did}"
        is_top50 = int(canon_top50(brand_label) in top50_set)

        # spv hits from legal name token heuristic (reuse logic)
        toks = [t.upper() for t in _tokenize_preserve(legal)]
        hits = sorted(list(set([t for t in toks if t in spv_tokens])))

        # concentration shares
        mc = dev_master_counts.get(did, {})
        ac = dev_area_counts.get(did, {})
        top_master, top_master_share = ("", None)
        if mc:
            total = sum(mc.values())
            k, v = sorted(mc.items(), key=lambda kv: kv[1], reverse=True)[0]
            top_master = str(k)
            top_master_share = float(v / max(total, 1))
        top_area, top_area_share = ("", None)
        if ac:
            total = sum(ac.values())
            k, v = sorted(ac.items(), key=lambda kv: kv[1], reverse=True)[0]
            top_area = str(k)
            top_area_share = float(v / max(total, 1))

        units_sum = int(units_by_dev.get(did, 0))
        impact = (m["sales_value_last_12m"] >= CFG.spv_outliers_value_aed) or (m["tx_count_last_12m"] >= CFG.spv_outliers_tx_12m) or (units_sum >= CFG.spv_outliers_units)
        flag = (is_top50 == 0) and (
            (len(hits) > 0)
            or (top_master_share is not None and top_master_share >= CFG.spv_outliers_concentration_share)
            or (top_area_share is not None and top_area_share >= CFG.spv_outliers_concentration_share)
        ) and impact
        if not flag:
            continue

        ex = dev_examples.get(did, [])
        ex_str = " | ".join([e.get("text","") for e in ex if e.get("text")])[:500]

        outlier_rows.append(
            {
                "developer_id": did,
                "legal_name_en": legal,
                "is_top50_2025": is_top50,
                "is_umbrella_mapped": is_umb,
                "umbrella_label": umb_label if umb_label else None,
                "sales_value_last_12m": m["sales_value_last_12m"],
                "tx_count_last_12m": m["tx_count_last_12m"],
                "sales_value_all_time": m["sales_value_all_time"],
                "tx_count_all_time": m["tx_count_all_time"],
                "projects_units_sum": units_sum,
                "spv_token_hits": "|".join(hits),
                "top_master_project_en": top_master if top_master else None,
                "top_master_project_share": round(top_master_share, 4) if top_master_share is not None else None,
                "top_area_id": top_area if top_area else None,
                "top_area_share": round(top_area_share, 4) if top_area_share is not None else None,
                "evidence_examples": ex_str,
            }
        )

    outlier_df = pd.DataFrame(outlier_rows).sort_values(["sales_value_last_12m", "tx_count_last_12m"], ascending=[False, False]) if outlier_rows else pd.DataFrame(
        columns=[
            "developer_id","legal_name_en","is_top50_2025","is_umbrella_mapped","umbrella_label",
            "sales_value_last_12m","tx_count_last_12m","sales_value_all_time","tx_count_all_time",
            "projects_units_sum","spv_token_hits","top_master_project_en","top_master_project_share",
            "top_area_id","top_area_share","evidence_examples"
        ]
    )
    outlier_path = OUTPUT_DIR / "high_impact_spv_outliers_non_top50.csv"
    outlier_df.to_csv(outlier_path, index=False)

    # Outlier tripwire: alert/fail if any row exceeds stricter threshold and is not umbrella-mapped
    outlier_fail = outlier_df[(outlier_df["is_umbrella_mapped"].astype(int) == 0) & (outlier_df["sales_value_last_12m"].astype(float) >= CFG.spv_outliers_fail_value_aed)] if not outlier_df.empty else outlier_df
    delta["high_impact_spv_outliers"] = {
        "file": str(outlier_path),
        "count": int(len(outlier_df)),
        "fail_threshold_value_aed": float(CFG.spv_outliers_fail_value_aed),
        "fail_unmapped_count": int(len(outlier_fail)) if not outlier_fail.empty else 0,
        "fail_unmapped_top": outlier_fail.head(25).to_dict(orient="records") if not outlier_fail.empty else [],
    }

    # SPV tripwire: always alert for unmapped SPVs in top N.
    spv_top = spv_df.sort_values("sales_value_last_12m", ascending=False).head(CFG.spv_tripwire_top_n) if not spv_df.empty else spv_df
    spv_unmapped = spv_top[spv_top["is_umbrella_mapped"].astype(int) == 0] if not spv_top.empty else spv_top

    delta["spv_tripwire"] = {
        "top_n": int(CFG.spv_tripwire_top_n),
        "unmapped_count": int(len(spv_unmapped)),
        "unmapped_top": spv_unmapped.head(50).to_dict(orient="records") if not spv_unmapped.empty else [],
        "hard_fail_criteria": {
            "value_aed_ge": float(CFG.spv_tripwire_fail_value_aed),
            "or_in_top_n_by_value": int(CFG.spv_tripwire_fail_top_n_by_value),
            "or_high_conf_suggestion_unapplied": bool(CFG.spv_tripwire_fail_if_high_conf_unapplied),
        },
        "action": "Map these developer_ids in Data/lookups/umbrella_map.json and record decisions; avoid manual discovery.",
    }

    # ------------------------------------------------------------------
    # Umbrella override decisions (review log) + delta-only surfacing
    # ------------------------------------------------------------------
    decisions_path = OUTPUT_DIR / "umbrella_override_decisions.csv"
    if not decisions_path.exists():
        pd.DataFrame(columns=["developer_id", "decision", "reason", "suggested_umbrella", "decided_at"]).to_csv(decisions_path, index=False)
    try:
        decisions = pd.read_csv(decisions_path, low_memory=False)
    except Exception:
        decisions = pd.DataFrame(columns=["developer_id", "decision", "reason", "suggested_umbrella", "decided_at"])
    decisions["developer_id"] = decisions.get("developer_id", pd.Series([], dtype="object")).astype(str).str.strip()
    decisions["decision"] = decisions.get("decision", pd.Series([], dtype="object")).astype(str).str.strip().str.lower()
    decided_ids = set(decisions.loc[decisions["developer_id"] != "", "developer_id"].tolist())
    approved_ids = set(decisions.loc[(decisions["developer_id"] != "") & (decisions["decision"] == "approved"), "developer_id"].tolist())
    rejected_ids = set(decisions.loc[(decisions["developer_id"] != "") & (decisions["decision"] == "rejected"), "developer_id"].tolist())

    # ------------------------------------------------------------------
    # Community guard false-negative check (artifact)
    # ------------------------------------------------------------------
    # We validate that well-known legitimate developer names that may look area-like
    # (e.g., "City Developments", "Meydan", "Dubai South") are not being forced into DEVELOPER_ID
    # in actual labeling.
    try:
        dev_label_counts = tx_audits.get("_label_dev_counts") or {}
        # invert (dev_id -> {label: count}) for target devs only
        target_terms = [
            "City Developments",
            "Meydan",
            "Dubai South",
            "Dubai South Properties",
            "Dubai Holding",
            "Majid Al Futtaim",
            "DMCC",
            "TECOM",
        ]
        devs_en = developers[["developer_id", "developer_name_en"]].copy()
        devs_en["developer_id"] = devs_en["developer_id"].astype(str).str.strip()
        devs_en["developer_name_en"] = devs_en["developer_name_en"].fillna("").astype(str)
        pat = re.compile("|".join([re.escape(x) for x in target_terms]), re.IGNORECASE)
        target = devs_en[devs_en["developer_name_en"].str.contains(pat, na=False)].copy()
        target_ids = set(target["developer_id"].tolist())

        inv = {did: {} for did in target_ids}
        for lbl, mp in dev_label_counts.items():
            for did, cnt in mp.items():
                if did in inv:
                    inv[did][lbl] = inv[did].get(lbl, 0) + int(cnt)

        # Guard determination is based on legal-normalized label (fallback path)
        area_name_set = set([_norm_upper_spaces(n) for n in load_lkp_areas()["name_en"].astype(str).tolist() if _norm_upper_spaces(n)])
        guard_tokens = set([t.upper() for t in suspicious_label_tokens])  # uses same tokens as guard

        rows = []
        for _, r in target.iterrows():
            did = str(r["developer_id"])
            legal = str(r["developer_name_en"])
            legal_norm = normalize_legal_name_to_label(legal, lang="en")
            toks = [t.upper() for t in _tokenize_preserve(legal_norm)]
            token_hits = sorted(list(set([t for t in toks if t in guard_tokens])))
            area_match = int(_norm_upper_spaces(legal_norm) in area_name_set) if legal_norm else 0
            guard = int(bool(token_hits) or bool(area_match))
            reason = ("token_hit" if token_hits else "") + ("|area_match" if area_match else "")
            reason = reason.strip("|")

            # actual resulting label observed (top label for that developer_id across all tx)
            dist = inv.get(did, {})
            top_label = ""
            if dist:
                top_label = sorted(dist.items(), key=lambda kv: kv[1], reverse=True)[0][0]

            m = current_dev_metrics.get(did, {})
            rows.append(
                {
                    "developer_id": did,
                    "legal_name_en": legal,
                    "legal_normalized_label": legal_norm,
                    "guard_triggered": guard,
                    "reason": reason,
                    "resulting_label_top_observed": top_label,
                    "tx_count_all_time": int(m.get("tx_count_all_time", 0)),
                    "last12m_tx": int(m.get("tx_count_last_12m", 0)),
                }
            )
        out = pd.DataFrame(rows).sort_values(["last12m_tx", "tx_count_all_time"], ascending=[False, False])
        out.to_csv(OUTPUT_DIR / "community_guard_false_negative_check.csv", index=False)
    except Exception:
        # Don't fail build on this diagnostic; it is for review.
        pass

    # Apply-ready umbrella overrides (deterministic columns) + delta-only view
    suggested_umbrella_path = OUTPUT_DIR / "suggested_umbrella_overrides_developer_id.csv"
    # (generated below as before) — but we'll also write a delta-only file excluding decided and already umbrella-mapped.
    suggested_umbrella_delta_path = OUTPUT_DIR / "suggested_umbrella_overrides_developer_id_delta.csv"

    # NOTE: delta outputs are written later, after we compute hard-gate failures (so the JSON is complete).

    # ------------------------------------------------------------------
    # Public brand override suggestions + learned mapping (reviewable; not auto-applied)
    # ------------------------------------------------------------------
    learned_text_counts = tx_audits.get("_learned_text_counts") or {}
    dev_text_project_numbers_count = tx_audits.get("_dev_text_project_numbers_count") or {}
    learned_rows = []
    for did, dist in learned_text_counts.items():
        total = int(sum(dist.values()))
        if total < CFG.learned_min_tx:
            continue
        top_brand, top_cnt = sorted(dist.items(), key=lambda kv: kv[1], reverse=True)[0]
        share = float(top_cnt / max(total, 1))
        if share < CFG.learned_min_share:
            continue
        n_projects = int(dev_text_project_numbers_count.get(str(did), 0))
        # HIGH must be rare: require volume + consistency + multi-project support
        if (total >= 30) and (share >= 0.95) and (n_projects >= 5):
            conf = "HIGH"
        elif share >= 0.85:
            conf = "MED"
        else:
            conf = "LOW"
        learned_rows.append(
            {
                "developer_id": str(did),
                "suggested_brand": str(top_brand),
                "tx_rows_text_resolved": int(total),
                "top_brand_share": round(share, 4),
                "distinct_project_numbers": n_projects,
                "suggestion_confidence": conf,
                "suggestion_reason": f"mode brand from marketing text evidence (min_tx={CFG.learned_min_tx}, min_share={CFG.learned_min_share})",
            }
        )
    learned_public_path = OUTPUT_DIR / "learned_public_brand_map_developer_id.csv"
    pd.DataFrame(learned_rows).sort_values(["tx_rows_text_resolved", "top_brand_share"], ascending=[False, False]).to_csv(learned_public_path, index=False)

    # Suggest overrides prioritized by impact
    sugg_public_rows = []
    for did, m in sorted(current_dev_metrics.items(), key=lambda kv: kv[1]["sales_value_last_12m"], reverse=True)[:2000]:
        legal = dev_name_en_map.get(str(did), "")
        # learned suggestion if available
        lmatch = next((r for r in learned_rows if r["developer_id"] == str(did)), None)
        if lmatch:
            sugg_public_rows.append(
                {
                    "developer_id": str(did),
                    "developer_legal_name_en": legal,
                    "tx_count_last_12m": m["tx_count_last_12m"],
                    "sales_value_last_12m": m["sales_value_last_12m"],
                    "suggested_brand": lmatch["suggested_brand"],
                    "suggestion_confidence": lmatch["suggestion_confidence"],
                    "suggestion_reason": "learned from marketing text evidence",
                }
            )
    sugg_public_path = OUTPUT_DIR / "suggested_public_brand_overrides_developer_id.csv"
    pd.DataFrame(sugg_public_rows).to_csv(sugg_public_path, index=False)

    # ------------------------------------------------------------------
    # Umbrella mapping suggestions (review queue; not auto-applied)
    # ------------------------------------------------------------------
    dev_text_examples_top5 = tx_audits.get("_dev_text_examples_top5") or {}
    project_text_counts = tx_audits.get("_project_text_counts") or {}

    # project linkage modes: project_number -> (top_brand, share, total, top2)
    proj_mode = {}
    for pn, dist in project_text_counts.items():
        total = int(sum(dist.values()))
        if total <= 0:
            continue
        items = sorted(dist.items(), key=lambda kv: kv[1], reverse=True)
        top_b, top_c = items[0]
        share = float(top_c / total)
        top2 = items[:2]
        proj_mode[str(pn)] = {"top_brand": str(top_b), "share": round(share, 4), "total": total, "top2": [(str(b), int(c)) for b, c in top2]}

    umb_sugg_rows = []
    # Seed coverage (already in umbrella_map) is HIGH; list high-impact ones for audit
    for did, m in sorted(current_dev_metrics.items(), key=lambda kv: kv[1]["sales_value_last_12m"], reverse=True)[:2000]:
        if str(did) in umbrella_by_id:
            continue
        # Only consider SPV-like legal names (review queue)
        name = dev_name_en_map.get(str(did), "")
        toks = [t.upper() for t in _tokenize_preserve(name)]
        hits = sorted(list(set([t for t in toks if t in spv_tokens])))
        if not hits:
            continue
        # Suggest umbrella if marketing text is extremely consistent to one public brand
        dist = learned_text_counts.get(str(did), {})
        if dist:
            total = int(sum(dist.values()))
            top_brand, top_cnt = sorted(dist.items(), key=lambda kv: kv[1], reverse=True)[0]
            share = float(top_cnt / max(total, 1))
            n_projects = int(dev_text_project_numbers_count.get(str(did), 0))
            if (total >= 30) and (share >= 0.95) and (n_projects >= 5):
                conf = "HIGH"
            elif share >= 0.85 and total >= 10:
                conf = "MED"
            else:
                conf = "LOW"
            if conf in {"HIGH", "MED"}:
                umb_sugg_rows.append(
                    {
                        "developer_id": str(did),
                        "developer_legal_name_en": name,
                        "spv_token_hits": "|".join(hits),
                        "suggested_umbrella_brand": str(top_brand),
                        "confidence": conf,
                        "rule_triggered": "marketing_text_consistency",
                        "n_rows": total,
                        "share": round(share, 4),
                        "distinct_project_numbers": n_projects,
                    }
                )
    umb_path = OUTPUT_DIR / "umbrella_mapping_suggestions.csv"
    pd.DataFrame(umb_sugg_rows).sort_values(["confidence", "share", "n_rows"], ascending=[True, False, False]).to_csv(umb_path, index=False)

    # Apply-ready umbrella overrides file (deterministic columns)
    apply_rows = []
    # Build developer_id -> its projects for linkage consistency
    proj_by_dev = projects[["developer_id", "project_number"]].dropna().copy()
    proj_by_dev["developer_id"] = proj_by_dev["developer_id"].astype(str).str.strip()
    proj_by_dev["project_number"] = proj_by_dev["project_number"].astype(str).str.strip()
    dev_to_projects = proj_by_dev.groupby("developer_id")["project_number"].apply(list).to_dict()

    for r in umb_sugg_rows:
        did = str(r["developer_id"])
        suggested = str(r["suggested_umbrella_brand"])
        # project linkage consistency: among this dev's projects that have a mode, what share match suggested?
        pns = dev_to_projects.get(did, [])
        modes = []
        for pn in pns:
            m = proj_mode.get(str(pn))
            if m:
                modes.append((pn, m["top_brand"], m["share"], m["total"]))
        total_projects = len(modes)
        match_projects = [x for x in modes if x[1] == suggested]
        share_projects = round((len(match_projects) / total_projects), 4) if total_projects else None
        conflicts = [x for x in modes if x[1] != suggested]
        conflicts = sorted(conflicts, key=lambda x: x[3], reverse=True)[:10]
        conflict_str = "; ".join([f"pn={pn}: {b} ({tot})" for pn, b, sh, tot in conflicts])

        ex = dev_text_examples_top5.get(did, [])
        ex_str = " | ".join([e.get("text","") for e in ex if e.get("text")])[:500]

        m = current_dev_metrics.get(did, {})
        apply_rows.append(
            {
                "developer_id": did,
                "legal_name_en": dev_name_en_map.get(did, ""),
                "last12m_value": m.get("sales_value_last_12m", 0.0),
                "last12m_tx": m.get("tx_count_last_12m", 0),
                "suggested_umbrella": suggested,
                "confidence": r.get("confidence", ""),
                "rule_triggered": r.get("rule_triggered", ""),
                "evidence_examples": ex_str,
                "project_linkage_share": share_projects,
                "project_linkage_total_projects": total_projects,
                "project_linkage_conflicts_top": conflict_str,
            }
        )

    suggested_umbrella_path = OUTPUT_DIR / "suggested_umbrella_overrides_developer_id.csv"
    pd.DataFrame(apply_rows).sort_values(["last12m_value", "last12m_tx"], ascending=[False, False]).to_csv(suggested_umbrella_path, index=False)

    # Delta-only: exclude already mapped and already decided
    if apply_rows:
        df_app = pd.DataFrame(apply_rows)
        df_app["developer_id"] = df_app["developer_id"].astype(str).str.strip()
        df_app = df_app[~df_app["developer_id"].isin(decided_ids)]
        df_app = df_app[df_app["developer_id"].map(lambda d: d not in umbrella_by_id)]
        df_app.to_csv(suggested_umbrella_delta_path, index=False)

    # ------------------------------------------------------------------
    # Hard-gate SPV tripwire (two-tier)
    # Fail only when:
    #  (a) last12m_value>=1B OR in top10 by value, OR
    #  (b) there is a HIGH-confidence suggested umbrella mapping that remains unapplied.
    # Unapplied means: developer_id not umbrella-mapped.
    # Decision handling:
    # - rejected => do not fail rule (b)
    # - approved but still unmapped => fail rule (b)
    # - no decision => fail rule (b) if HIGH-confidence suggestion exists
    # ------------------------------------------------------------------
    failures = []
    if not spv_top.empty:
        # top10 by value set
        top10_ids = set(spv_top.sort_values("sales_value_last_12m", ascending=False).head(CFG.spv_tripwire_fail_top_n_by_value)["developer_id"].astype(str).tolist())
        # Build high-confidence suggestion lookup (from apply_rows generated above)
        hi_sugg = {}
        for r in apply_rows:
            if str(r.get("confidence","")).upper() == "HIGH":
                hi_sugg[str(r["developer_id"])] = str(r.get("suggested_umbrella",""))

        for _, r in spv_unmapped.iterrows():
            did = str(r.get("developer_id"))
            val = float(r.get("sales_value_last_12m", 0.0))
            in_top10 = did in top10_ids
            cond_a = (val >= CFG.spv_tripwire_fail_value_aed) or in_top10
            # cond_b: high-confidence suggestion exists and remains unapplied, and isn't explicitly rejected.
            if CFG.spv_tripwire_fail_if_high_conf_unapplied and (did in hi_sugg) and (did not in rejected_ids):
                cond_b = (did not in umbrella_by_id)
            else:
                cond_b = False
            if cond_a or cond_b:
                failures.append(
                    {
                        "developer_id": did,
                        "developer_legal_name_en": r.get("developer_legal_name_en",""),
                        "sales_value_last_12m": val,
                        "tx_count_last_12m": int(r.get("tx_count_last_12m", 0)),
                        "spv_token_hits": r.get("spv_token_hits",""),
                        "rule_a_triggered": bool(cond_a),
                        "rule_b_triggered": bool(cond_b),
                        "suggested_umbrella_high_conf": hi_sugg.get(did, ""),
                    }
                )
    delta["spv_tripwire_failures"] = failures

    # Write an approval-queue CSV for tripwire failures (reviewable + auditable)
    try:
        top50 = json.loads(Path(CFG.top50_2025_path).read_text(encoding="utf-8"))
        top50_list = [str(x) for x in (top50.get("brands") or [])]
        top50_set = set([x.strip() for x in top50_list if x and str(x).strip()])
        rules = top50.get("canonicalisation_rules") or {}
        aliases_map = (rules.get("aliases") or {}) if isinstance(rules, dict) else {}
        aliases_ci = {str(k).strip().upper(): str(v).strip() for k, v in dict(aliases_map).items() if str(k).strip() and str(v).strip()}
        top50_ci = {str(b).strip().upper(): str(b).strip() for b in top50_list if str(b).strip()}

        def canon_top50(x: object) -> str:
            s = _norm_ws(x)
            if not s:
                return ""
            if s in aliases_map:
                return str(aliases_map[s]).strip()
            s2 = re.sub(r"\s+", " ", s).strip()
            if s2 in aliases_map:
                return str(aliases_map[s2]).strip()
            u = s2.upper()
            if u in aliases_ci:
                return str(aliases_ci[u]).strip()
            if u in top50_ci:
                return top50_ci[u]
            return s2

        # suggestions join map
        sugg_by_id = {str(r["developer_id"]): r for r in apply_rows}

        qrows = []
        for f in failures:
            did = str(f.get("developer_id", ""))
            legal = dev_name_en_map.get(did, str(f.get("developer_legal_name_en", "")))
            # compute label used for top50 tagging (same fallback chain, without using marketing text)
            label = dev_id_to_brand.get(did, "")  # best-effort explicit brand map
            if not label:
                label = normalize_legal_name_to_label(legal, lang="en") or f"DEVELOPER_ID_{did}" if did else "DEVELOPER_ID_MISSING"
            is_top50 = int(canon_top50(label) in top50_set)
            sug = sugg_by_id.get(did, {})
            why = ("value>=1B_or_top10" if bool(f.get("rule_a_triggered")) else "")
            why += ("|HIGH_unapplied" if bool(f.get("rule_b_triggered")) else "")
            why = why.strip("|")
            qrows.append(
                {
                    "developer_id": did,
                    "legal_name_en": legal,
                    "last12m_value": float(f.get("sales_value_last_12m", 0.0)),
                    "last12m_tx": int(f.get("tx_count_last_12m", 0)),
                    "is_top50_2025": is_top50,
                    "is_umbrella_mapped": 0,
                    "suggested_umbrella": str(sug.get("suggested_umbrella", "")),
                    "suggestion_confidence": str(sug.get("confidence", "")),
                    "evidence_examples": str(sug.get("evidence_examples", "")),
                    "why_flagged": why,
                }
            )
        pd.DataFrame(qrows).sort_values(["last12m_value", "last12m_tx"], ascending=[False, False]).to_csv(
            OUTPUT_DIR / "spv_tripwire_failures.csv", index=False
        )
    except Exception:
        pass

    # Write delta outputs now (after computing failures and other tripwire payloads)
    delta_json_path.write_text(json.dumps(delta, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(delta.get("new_developer_ids_ranked") or [], columns=delta_csv_columns).to_csv(delta_csv_path, index=False)

    # Hard gate: fail before writing training outputs / baseline snapshot
    if len(failures) > 0:
        # Mark run as failed (do not promote anything to latest)
        (OUTPUT_DIR / "run_manifest.json").write_text(
            json.dumps(
                {
                    "build_id": build_id,
                    "build_timestamp_utc": build_timestamp_utc,
                    "status": "FAILED",
                    "failure": "SPV_TRIPWIRE_HARD_GATE",
                    "spv_tripwire_failures_count": len(failures),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        raise RuntimeError(
            f"SPV tripwire hard gate triggered: {len(failures)} SPV candidates require umbrella mapping. "
            f"See {delta_json_path} and {spv_path}"
        )

    # High-impact review list: top 200 dev_ids where label came from normalized legal (not PUBLIC_BRAND)
    if {"developer_id", "developer_brand_label_source", "transaction_count"}.issubset(df.columns):
        review = (
            df.groupby(["developer_id", "developer_brand_label_source"])
            .agg(tx=("transaction_count", "sum"))
            .reset_index()
        )
        review = review[review["developer_brand_label_source"].isin(["LEGAL_NORMALIZED_EN", "LEGAL_NORMALIZED_AR", "ID_FALLBACK", "MISSING_ID_FALLBACK"])]
        review = review.sort_values("tx", ascending=False).head(200)
        review["developer_legal_name_en"] = review["developer_id"].map(dev_name_en_map)
        review_path = OUTPUT_DIR / "high_impact_legal_normalized_developers_top200.csv"
        review.to_csv(review_path, index=False)

    # Baseline snapshot is written at the very end after successful outputs.

    # Note: we intentionally do NOT gate on resolver-only `developer_brand`.
    # The leakage gate applies to `developer_brand_label` (stable label).

    # Output
    out_csv = OUTPUT_DIR / "tft_training_data_v2.csv"
    df.to_csv(out_csv, index=False)

    # ------------------------------------------------------------
    # Reporting (brand vs legal; recent window; active projects)
    # ------------------------------------------------------------
    # derive last 5y cutoff from dataset max month
    max_month = pd.to_datetime(df["year_month"].max() + "-01")
    cutoff = (max_month - pd.DateOffset(years=5)).strftime("%Y-%m")
    df_recent = df[df["year_month"] >= cutoff]

    # Active projects brands (Projects is timeline authority)
    active_statuses = {"ACTIVE", "NOT_STARTED", "PENDING"}
    proj_active = projects.copy()
    proj_active["project_status"] = proj_active["project_status"].astype(str).str.strip().str.upper()
    proj_active = proj_active[proj_active["project_status"].isin(active_statuses)]
    proj_active["developer_id"] = proj_active["developer_id"].astype(str).str.strip()
    # Active projects brand labels:
    # - prefer explicit/public brand map
    # - else normalize legal EN name
    dev_name_en_map = developers.set_index("developer_id")["developer_name_en"].to_dict()
    proj_active["developer_brand_label"] = proj_active["developer_id"].map(dev_id_to_brand).fillna("")
    miss = proj_active["developer_brand_label"].astype(str).str.strip() == ""
    if miss.any():
        legal = proj_active.loc[miss, "developer_id"].map(dev_name_en_map)
        proj_active.loc[miss, "developer_brand_label"] = legal.map(lambda x: normalize_legal_name_to_label(x, lang="en"))
    proj_active["developer_brand_label"] = proj_active["developer_brand_label"].fillna("").astype(str).str.strip()
    active_brands = sorted(set([b for b in proj_active["developer_brand_label"].astype(str).tolist() if b]))

    top_brands = (
        df.groupby("developer_brand_label")["transaction_count"]
        .sum()
        .sort_values(ascending=False)
        .head(50)
        .to_dict()
    )

    stats = {
        "version": "v2",
        "build_id": build_id,
        "build_timestamp_utc": build_timestamp_utc,
        "rows": int(len(df)),
        "unique_groups": int(df["group_id"].nunique()),
        "unique_area_ids": int(df["area_id"].nunique()),
        "date_range": {"start": str(df["year_month"].min()), "end": str(df["year_month"].max()), "months": int(df["year_month"].nunique())},
        "config": {
            "filter_sales_only": CFG.filter_sales_only,
            "include_developer_in_group_id": CFG.include_developer_in_group_id,
            "warn_kml_match_rate": CFG.warn_kml_match_rate,
            "hard_fail_kml_match_rate": CFG.hard_fail_kml_match_rate,
            "require_full_brand_coverage": CFG.require_full_brand_coverage,
        },
        "audits": {
            "transactions_offplan_mapping": tx_audits,
            "kml": kml_stats,
            "brand_unmatched_entities_sample": {k: v[:25] for k, v in brand_unmatched.items()},
            "brand_resolver": {
                "approved_brands_count": len(approved),
                "approved_brands_sample": approved[:50],
                "overrides_project_number_count": len(overrides_pn),
                "overrides_master_project_count": len(overrides_mp),
            },
            "umbrella_seed_audit": umbrella_audit,
        },
        "brand_reporting": {
            "cutoff_recent_5y": cutoff,
            "unique_developer_id": int(df["developer_id"].nunique()) if "developer_id" in df.columns else None,
            "unique_developer_legal_name_en": int(df["developer_legal_name_en"].nunique()) if "developer_legal_name_en" in df.columns else None,
            "unique_developer_brand_label_total": int(df["developer_brand_label"].nunique()) if "developer_brand_label" in df.columns else None,
            "unique_developer_brand_label_recent_5y": int(df_recent["developer_brand_label"].nunique()) if "developer_brand_label" in df_recent.columns else None,
            "unique_developer_brand_active_projects": int(len(active_brands)),
            "top_50_brands_by_tx_count": top_brands,
        },
        "modelling_grain": {
            "include_developer_in_group_id": CFG.include_developer_in_group_id,
            "unique_area_ids": int(df["area_id"].nunique()) if "area_id" in df.columns else None,
            "unique_bedroom_categories": int(df["bedroom"].nunique()) if "bedroom" in df.columns else None,
            "unique_property_types": int(df["property_type"].nunique()) if "property_type" in df.columns else None,
            "unique_reg_types": int(df["reg_type"].nunique()) if "reg_type" in df.columns else None,
            "expected_upper_bound_area_bed_property_reg": (
                int(df["area_id"].nunique())
                * int(df["bedroom"].nunique())
                * int(df["property_type"].nunique())
                * int(df["reg_type"].nunique())
            ) if all(c in df.columns for c in ["area_id", "bedroom", "property_type", "reg_type"]) else None,
            "unique_groups": int(df["group_id"].nunique()) if "group_id" in df.columns else None,
        },
        "null_counts_top20": df.isna().sum().sort_values(ascending=False).head(20).to_dict(),
        "columns": list(df.columns),
    }

    out_stats = OUTPUT_DIR / "build_stats_v2.json"
    out_stats.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")

    # Baseline snapshot: only write after a successful run and after outputs are written.
    # Write to run folder first, then promote to latest.
    (OUTPUT_DIR / "baseline_snapshot_v2.json").write_text(json.dumps(current_snapshot, ensure_ascii=False, indent=2), encoding="utf-8")

    # Promote to latest (atomic-ish: write files into latest folder only after success)
    latest_baseline = Path(CFG.baseline_snapshot_path)
    latest_baseline.parent.mkdir(parents=True, exist_ok=True)
    latest_baseline.write_text(json.dumps(current_snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    (LATEST_DIR / "latest_build_id.txt").write_text(build_id + "\n", encoding="utf-8")
    (LATEST_DIR / "latest_build_timestamp_utc.txt").write_text(build_timestamp_utc + "\n", encoding="utf-8")

    # Promote key artifacts for downstream consumers
    # (keep legacy filenames under Data/tft/latest to avoid mixed-output state under Data/tft root)
    for src_name in [
        "tft_training_data_v2.csv",
        "build_stats_v2.json",
        "brand_resolution_audit_v2.json",
        "suffix_leakage_report_v2.json",
        "holding_policy_audit_v2.json",
        "owner_override_self_check.csv",
        "owner_assertions_report.json",
        "noncanonical_brand_overrides_skipped.csv",
        "brand_override_alias_rewrites.csv",
        "umbrella_seed_audit_v2.json",
        "delta_report_v2.json",
        "delta_report_v2_new_developers.csv",
        "spv_candidates_ranked.csv",
        "spv_tripwire_failures.csv",
        "label_dispersion_report_v2.csv",
        "project_area_labels_surviving_gate.csv",
        "suspicious_label_gate_offenders.csv",
        "high_impact_non_top50_suspicious_labels.csv",
        "community_guard_false_negative_check.csv",
        "developer_brand_label_presence_check_v2.csv",
    ]:
        src = OUTPUT_DIR / src_name
        if src.exists():
            (LATEST_DIR / src_name).write_bytes(src.read_bytes())

    # Mark run as success
    (OUTPUT_DIR / "run_manifest.json").write_text(
        json.dumps(
            {
                "build_id": build_id,
                "build_timestamp_utc": build_timestamp_utc,
                "status": "SUCCESS",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    logger.info("=" * 60)
    logger.info("TFT DATA BUILD V2 - COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Output: {out_csv}")
    logger.info(f"Stats: {out_stats}")
    logger.info(f"Rows: {len(df):,} | Groups: {df['group_id'].nunique():,}")


if __name__ == "__main__":
    main()


