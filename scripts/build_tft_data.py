#!/usr/bin/env python3
"""
Build TFT-compatible training data from ALL cleaned sources.

Sources included:
- Transactions (prices, all types, residential only)
- Rent Contracts (residential rents)
- Projects (developer + supply data)
- Units (unit registrations)
- Buildings (building data)
- Valuation (govt valuations)
- EIBOR (raw rates only - no computed features)
- Tourism (visitors + inventory)

Output: Data/tft/tft_training_data.csv

Usage:
    python scripts/build_tft_data.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import List, Dict, Optional
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# === PATHS ===
CLEANED_DIR = Path('Data/cleaned')
OUTPUT_DIR = Path('Data/tft')

# === CONSTANTS ===

# Bedroom standardization (Transactions → Rent format)
BEDROOM_MAP = {
    '1 B/R': '1BR',
    '2 B/R': '2BR',
    '3 B/R': '3BR',
    '4 B/R': '4BR',
    '5 B/R': '5BR',
    '6 B/R': '6BR+',
    '7 B/R': '6BR+',
    '8 B/R': '6BR+',
    '9 B/R': '6BR+',
    '10 B/R': '6BR+',
    'Studio': 'Studio',
    'PENTHOUSE': 'Penthouse',
    'Single Room': 'Room',
    # Exclude non-residential
    'Office': None,
    'Shop': None,
    'Store': None,
    'GYM': None
}

# Valid property types (residential focus)
VALID_PROPERTY_TYPES = ['Unit', 'Villa']

# Project statuses that count as active supply
ACTIVE_PROJECT_STATUSES = ['ACTIVE', 'NOT_STARTED', 'PENDING']

# Quarter to months mapping for tourism data
QUARTER_TO_MONTHS = {
    'Q1': ['01', '02', '03'],
    'Q2': ['04', '05', '06'],
    'Q3': ['07', '08', '09'],
    'Q4': ['10', '11', '12']
}

# Reg type standardization
REG_TYPE_MAP = {
    'Off-Plan Properties': 'OffPlan',
    'Existing Properties': 'Ready'
}

# --- Developer brand alias mapping (e.g., Binghatti, Danube registered under master developers) ---
BUILDING_DEVS_PATH = Path('Data/lookups/building_developers.json')
DEVELOPER_BRAND_CONSOLIDATION_PATH = Path('Data/lookups/developer_brand_consolidation.json')


def _norm_str(x: object) -> str:
    """Lowercased, whitespace-collapsed string for robust matching."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ''
    s = str(x).strip().lower()
    return ' '.join(s.split())


def _mode_or_nan(s: pd.Series) -> object:
    """Stable mode aggregator for groupby; returns NaN if empty/all-missing."""
    try:
        s2 = s.dropna()
        if len(s2) == 0:
            return np.nan
        m = s2.mode()
        return m.iat[0] if len(m) else s2.iat[0]
    except Exception:
        return np.nan


def _load_building_developer_aliases() -> Dict[str, Dict]:
    """
    Load building-developer (brand) alias rules.

    Returns:
      brand_rules: dict keyed by canonical brand name, containing:
        - aliases_norm: list[str]
        - registered_under: Optional[str] (Arabic registered dev)
        - registered_under_by_area: Optional[dict[str,str]] (area_name_en -> Arabic registered dev)
    """
    if not BUILDING_DEVS_PATH.exists():
        return {}
    try:
        with open(BUILDING_DEVS_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
        raw = data.get('building_developers_without_own_data', {}) or {}
        brand_rules: Dict[str, Dict] = {}
        for brand, info in raw.items():
            aliases = info.get('aliases') or []
            brand_rules[brand] = {
                'aliases_norm': [_norm_str(a) for a in aliases if _norm_str(a)],
                'registered_under': info.get('registered_under'),
                'registered_under_by_area': info.get('registered_under_by_area') or {}
            }
        return brand_rules
    except Exception as e:
        logger.warning(f"Could not load building_developers.json: {e}")
        return {}


def _load_developer_brand_consolidation_map() -> Dict[str, str]:
    """
    Load explicit registered-entity -> brand consolidations (e.g., DAMAC entities -> 'Damac').
    Returns: dict mapping registered developer_name (Arabic entity) to brand string.
    """
    if not DEVELOPER_BRAND_CONSOLIDATION_PATH.exists():
        return {}
    try:
        with open(DEVELOPER_BRAND_CONSOLIDATION_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
        brands = data.get('brands') or {}
        out: Dict[str, str] = {}
        for brand, info in brands.items():
            ents = info.get('registered_entities') or []
            for e in ents:
                key = str(e).strip()
                if key:
                    out[key] = brand
        return out
    except Exception as e:
        logger.warning(f"Could not load developer_brand_consolidation.json: {e}")
        return {}


def _infer_developer_brand(
    project_name_en: object,
    building_name_en: object,
    area_name_en: object,
    developer_registered: str,
    brand_rules: Dict[str, Dict]
) -> Optional[str]:
    """
    Deterministic brand inference using raw name fields.

    We only assign a brand if:
      - Any alias is present in project_name_en or building_name_en, AND
      - The registered developer is consistent with the expected registered-under rules
        (or registered is ALL_DEVELOPERS fallback).
    """
    if not brand_rules:
        return None
    reg = (developer_registered or '').strip()
    reg_ok_fallback = (reg == 'ALL_DEVELOPERS')
    txt = f"{_norm_str(project_name_en)} {_norm_str(building_name_en)}"
    if not txt.strip():
        return None
    area = (area_name_en or '').strip()

    for brand, rule in brand_rules.items():
        aliases = rule.get('aliases_norm') or []
        if not aliases:
            continue
        if not any(a in txt for a in aliases):
            continue

        by_area = rule.get('registered_under_by_area') or {}
        expected = by_area.get(area) if area else None
        if expected:
            if reg_ok_fallback or reg == expected:
                return brand
            continue

        expected_global = rule.get('registered_under')
        if expected_global:
            if reg_ok_fallback or reg == expected_global:
                return brand
            continue

        # If no registered-under constraints provided, accept alias hit.
        return brand

    return None


# === AGGREGATION FUNCTIONS ===

def _infer_developer_brand_series(
    df: pd.DataFrame,
    brand_rules: Dict[str, Dict]
) -> pd.Series:
    """
    Vectorized brand inference for a transaction chunk.

    Requires columns:
      - project_name_en (optional)
      - building_name_en (optional)
      - area_name_en
      - developer_registered_name
    """
    if not brand_rules or df.empty:
        return pd.Series(np.nan, index=df.index)

    proj = df['project_name_en'] if 'project_name_en' in df.columns else ''
    bld = df['building_name_en'] if 'building_name_en' in df.columns else ''
    txt = (proj.fillna('').astype(str) + ' ' + bld.fillna('').astype(str)).str.lower()
    # collapse whitespace for robust substring checks
    txt = txt.str.replace(r'\s+', ' ', regex=True).str.strip()

    area = df['area_name_en'].fillna('').astype(str).str.strip()
    reg = df['developer_registered_name'].fillna('').astype(str).str.strip()
    reg_ok_fallback = (reg == 'ALL_DEVELOPERS')

    out = pd.Series(np.nan, index=df.index, dtype=object)

    for brand, rule in brand_rules.items():
        remaining = out.isna()
        if not remaining.any():
            break

        aliases = rule.get('aliases_norm') or []
        aliases = [a for a in aliases if a]
        if not aliases:
            continue

        # alias hit (non-regex contains)
        alias_hit = pd.Series(False, index=df.index)
        for a in aliases:
            alias_hit = alias_hit | txt.str.contains(a, regex=False)

        alias_hit = alias_hit & remaining
        if not alias_hit.any():
            continue

        by_area = rule.get('registered_under_by_area') or {}
        expected = None
        if by_area:
            expected = area.map(by_area)
            # Support either a single expected registered entity (string) or a list of allowed entities.
            is_str = expected.apply(lambda v: isinstance(v, str))
            is_list = expected.apply(lambda v: isinstance(v, list))

            exp_str = expected.where(is_str, other=pd.NA)
            exp_list = expected.where(is_list, other=pd.NA)

            ok_str = exp_str.notna() & (reg_ok_fallback | (reg == exp_str))

            ok_list = pd.Series(False, index=df.index)
            idx = exp_list.notna()
            if idx.any():
                ok_list.loc[idx] = reg_ok_fallback.loc[idx] | pd.Series(
                    [r in lst for r, lst in zip(reg.loc[idx].tolist(), exp_list.loc[idx].tolist())],
                    index=reg.loc[idx].index
                )

            ok_by_area = ok_str | ok_list

        expected_global = rule.get('registered_under')
        if expected_global:
            ok_global = reg_ok_fallback | (reg == expected_global)
        else:
            # STRICT: if a rule is area-scoped (registered_under_by_area) and no global expected is defined,
            # we do NOT allow assigning the brand outside known areas (forces gaps to surface in audits).
            ok_global = pd.Series(False, index=df.index) if by_area else pd.Series(True, index=df.index)

        # If by-area expectation exists for a row, enforce it; else enforce global (or none).
        if expected is not None:
            allowed = (expected.notna() & ok_by_area) | (expected.isna() & ok_global)
        else:
            allowed = ok_global

        assign = alias_hit & allowed
        if assign.any():
            out.loc[assign] = brand

    return out


def load_projects() -> pd.DataFrame:
    """Load projects data for developer linking and supply computation."""
    logger.info("Loading Projects...")
    
    projects = pd.read_csv(
        CLEANED_DIR / 'Projects_Cleaned.csv',
        usecols=[
            'project_number', 'developer_name', 'master_developer_name',
            'project_start_date_parsed', 'project_end_date_parsed',
            'completion_date_parsed',
            'project_status', 'percent_completed',
            'area_name_en', 'master_project_en',
            'no_of_units', 'no_of_buildings', 'no_of_villas'
        ]
    )
    
    # Parse dates
    projects['project_end_date'] = pd.to_datetime(
        projects['project_end_date_parsed'], errors='coerce'
    )
    projects['project_start_date'] = pd.to_datetime(
        projects['project_start_date_parsed'], errors='coerce'
    )
    projects['completion_date'] = pd.to_datetime(
        projects['completion_date_parsed'], errors='coerce'
    )

    # Canonical handover date:
    # - If completion_date is present, treat it as actual handover/completion.
    # - Else fall back to project_end_date (planned/expected end).
    projects['handover_date'] = projects['completion_date'].fillna(projects['project_end_date'])
    
    logger.info(f"  Loaded {len(projects):,} projects, {projects['developer_name'].nunique()} unique developers")
    return projects


def aggregate_transactions(projects: pd.DataFrame, chunk_size: int = 200_000) -> pd.DataFrame:
    """
    Aggregate transactions to monthly by area/property_type/bedroom/reg_type/developer.
    Links to developers via project_number and includes project phase data.
    
    Filters:
    - property_usage_en == 'Residential'
    - property_type_en in ['Unit', 'Villa']
    - All procedure_name_en types included
    
    Output columns:
    - year_month, area_name, property_type, bedroom, reg_type, developer_name
    - median_price, transaction_count
    - Project phase columns: months_since_launch, months_to_handover, 
      project_percent_complete, project_duration_months, phase_ratio
    """
    logger.info("Aggregating Transactions...")
    
    # --- Developer resolution strategy (no "Unknown") ---
    # 1) Prefer exact project_number -> developer_name from Projects (strongest link)
    # 2) If project_number is missing (common in Ready transactions), use master_project_en -> master_developer_name mapping
    # 3) If still missing, fall back to an explicit market bucket "ALL_DEVELOPERS" (never "Unknown")

    # Create project_number -> developer + timeline mapping
    # Note: do NOT include master_project_en in this merge payload to avoid suffixing the transaction column.
    proj_info = projects[[
        'project_number',
        'developer_name',
        'master_developer_name',
        'project_start_date',
        'handover_date',
        'percent_completed'
    ]].drop_duplicates()
    proj_info = proj_info.dropna(subset=['project_number'])
    proj_info['project_number'] = proj_info['project_number'].astype(int).astype(str)
    
    # Pre-compute master_project_en -> master_developer_name (fallback for Ready transactions missing project_number)
    master_map = projects[['master_project_en', 'master_developer_name']].dropna().copy()
    master_map['master_project_en'] = master_map['master_project_en'].astype(str).str.strip()
    master_map['master_developer_name'] = master_map['master_developer_name'].astype(str).str.strip()
    # Choose the most frequent master_developer_name per master_project_en
    master_map = (
        master_map.groupby(['master_project_en', 'master_developer_name'])
        .size()
        .reset_index(name='n')
        .sort_values(['master_project_en', 'n'], ascending=[True, False])
        .drop_duplicates('master_project_en')
        .set_index('master_project_en')['master_developer_name']
        .to_dict()
    )

    # Pre-compute master_project_en -> representative timeline (start/handover/percent/duration)
    # This fills project-phase features for rows missing project_number but having master_project_en.
    _timeline = projects[['master_project_en', 'project_start_date', 'handover_date', 'percent_completed']].dropna(subset=['master_project_en']).copy()
    _timeline['master_project_en'] = _timeline['master_project_en'].astype(str).str.strip()
    master_timeline = (
        _timeline.groupby('master_project_en')
        .agg(
            # Use MIN start (earliest launch) and MAX handover (latest completion) to avoid
            # incorrectly assigning an early handover to later-phase transactions.
            project_start_date=('project_start_date', 'min'),
            handover_date=('handover_date', 'max'),
            percent_completed=('percent_completed', 'median')
        )
        .reset_index()
    )
    master_timeline['project_duration_months'] = (
        (master_timeline['handover_date'] - master_timeline['project_start_date']).dt.days / 30.44
    ).clip(lower=0)
    master_start_map = master_timeline.set_index('master_project_en')['project_start_date'].to_dict()
    master_handover_map = master_timeline.set_index('master_project_en')['handover_date'].to_dict()
    master_percent_map = master_timeline.set_index('master_project_en')['percent_completed'].to_dict()
    master_duration_map = master_timeline.set_index('master_project_en')['project_duration_months'].to_dict()

    # Canonicalize developer_name strings using developer_reference.csv (dedupe variants)
    dev_ref_path = Path('Data/lookups/developer_reference.csv')
    dev_norm_map = {}
    if dev_ref_path.exists():
        try:
            dev_ref = pd.read_csv(dev_ref_path, usecols=['developer_name'])
            # normalize by stripping and removing whitespace for robust matching
            for dn in dev_ref['developer_name'].dropna().astype(str):
                canon = dn.strip()
                key = ''.join(canon.split())
                dev_norm_map[key] = canon
        except Exception as e:
            logger.warning(f"Could not load developer_reference.csv for canonicalization: {e}")

    # Brand alias rules (Binghatti / Danube / etc) – derived deterministically from raw names
    brand_rules = _load_building_developer_aliases()
    # Explicit consolidations (e.g., multiple DAMAC entities -> 'Damac')
    consolidation_map = _load_developer_brand_consolidation_map()

    # Pre-compute project duration in months
    proj_info['project_duration_months'] = (
        (proj_info['handover_date'] - proj_info['project_start_date']).dt.days / 30.44
    ).clip(lower=0)
    
    results = []
    total_rows = 0
    filtered_rows = 0
    
    usecols = [
        'transaction_id', 'procedure_name_en', 'instance_date_parsed',
        'property_type_en', 'property_usage_en', 'reg_type_en',
        'area_name_en', 'rooms_en', 'procedure_area', 'actual_worth',
        'project_number', 'project_name_en', 'master_project_en',
        'building_name_en',
        'has_parking', 'transaction_year', 'transaction_month'
    ]
    
    for chunk in pd.read_csv(
        CLEANED_DIR / 'Transactions_Cleaned.csv',
        chunksize=chunk_size,
        usecols=usecols
    ):
        total_rows += len(chunk)
        
        # Filter to Residential only
        chunk = chunk[chunk['property_usage_en'] == 'Residential']
        
        # Filter to Unit/Villa only
        chunk = chunk[chunk['property_type_en'].isin(VALID_PROPERTY_TYPES)]
        
        # Standardize bedrooms
        chunk['bedroom'] = chunk['rooms_en'].map(BEDROOM_MAP)
        chunk = chunk.dropna(subset=['bedroom'])
        
        # Standardize reg_type (keep both DLD label and lifecycle stage label)
        chunk['reg_type_dld'] = chunk['reg_type_en'].map(REG_TYPE_MAP)
        chunk = chunk.dropna(subset=['reg_type_dld'])
        # Default stage is the DLD label; we may flip to Ready if the transaction occurs after handover
        chunk['reg_type'] = chunk['reg_type_dld']
        
        # Create year_month
        chunk['year_month'] = (
            chunk['transaction_year'].astype(int).astype(str) + '-' +
            chunk['transaction_month'].astype(int).astype(str).str.zfill(2)
        )
        
        # Parse transaction date for phase calculations
        chunk['transaction_date'] = pd.to_datetime(chunk['instance_date_parsed'], errors='coerce')
        
        # Compute price per sqft
        chunk['price_sqft'] = chunk['actual_worth'] / chunk['procedure_area']
        # Investor-facing sanity bounds (AED/sqft):
        # - below 100 is almost always a data issue
        # - above 50k is extremely likely a parsing/area issue (keep conservative cap)
        chunk = chunk[(chunk['price_sqft'] >= 100) & (chunk['price_sqft'] <= 50_000)]
        
        # Link to project info (developer + timeline)
        chunk['project_number'] = pd.to_numeric(chunk['project_number'], errors='coerce')
        chunk['project_number'] = chunk['project_number'].fillna(-1).astype(int).astype(str)
        chunk['project_number'] = chunk['project_number'].replace('-1', None)
        chunk = chunk.merge(proj_info, on='project_number', how='left')

        # Fill missing timeline fields via master_project_en where possible
        chunk['master_project_en'] = chunk['master_project_en'].astype(str).str.strip()
        chunk.loc[chunk['master_project_en'].str.lower().isin(['nan', 'none', '']), 'master_project_en'] = np.nan
        has_mp = chunk['master_project_en'].notna()
        if has_mp.any():
            missing_start = chunk['project_start_date'].isna() & has_mp
            if missing_start.any():
                chunk.loc[missing_start, 'project_start_date'] = chunk.loc[missing_start, 'master_project_en'].map(master_start_map)
            missing_handover = chunk['handover_date'].isna() & has_mp
            if missing_handover.any():
                chunk.loc[missing_handover, 'handover_date'] = chunk.loc[missing_handover, 'master_project_en'].map(master_handover_map)
            missing_pc = chunk['percent_completed'].isna() & has_mp
            if missing_pc.any():
                chunk.loc[missing_pc, 'percent_completed'] = chunk.loc[missing_pc, 'master_project_en'].map(master_percent_map)
            missing_dur = chunk['project_duration_months'].isna() & has_mp
            if missing_dur.any():
                chunk.loc[missing_dur, 'project_duration_months'] = chunk.loc[missing_dur, 'master_project_en'].map(master_duration_map)
        
        # Compute project phase columns (supports both OffPlan and Ready when timeline is known)
        # Months since project launch
        chunk['months_since_launch'] = np.where(
            chunk['project_start_date'].notna() & chunk['transaction_date'].notna(),
            ((chunk['transaction_date'] - chunk['project_start_date']).dt.days / 30.44).clip(lower=0),
            np.nan
        )
        
        # Months to handover (signed): positive pre-handover, negative post-handover
        chunk['months_to_handover_signed'] = np.where(
            chunk['handover_date'].notna() & chunk['transaction_date'].notna(),
            ((chunk['handover_date'] - chunk['transaction_date']).dt.days / 30.44),
            np.nan
        )

        # Data quality guard:
        # Lifecycle stage reclassification:
        # If transaction happens AFTER handover_date, treat the market regime as Ready (post-handover),
        # even if DLD reg_type says OffPlan. This aligns trends to investor reality.
        chunk['dld_offplan_after_handover'] = 0
        has_handover = chunk['handover_date'].notna() & chunk['transaction_date'].notna()
        after_handover = has_handover & (chunk['transaction_date'] >= chunk['handover_date'])
        dld_offplan = chunk['reg_type_dld'].astype(str) == 'OffPlan'
        flip = dld_offplan & after_handover
        if flip.any():
            chunk.loc[flip, 'reg_type'] = 'Ready'
            chunk.loc[flip, 'dld_offplan_after_handover'] = 1
        # Keep the original non-negative months_to_handover for compatibility (pre-handover countdown)
        chunk['months_to_handover'] = np.where(
            np.isfinite(chunk['months_to_handover_signed']),
            np.maximum(chunk['months_to_handover_signed'], 0),
            np.nan
        )
        # Months since handover (post-handover age)
        chunk['months_since_handover'] = np.where(
            np.isfinite(chunk['months_to_handover_signed']),
            np.maximum(-chunk['months_to_handover_signed'], 0),
            np.nan
        )
        # Handover window indicator: within +/- 6 months of handover (captures “handover bump” zone)
        chunk['handover_window_6m'] = np.where(
            np.isfinite(chunk['months_to_handover_signed']),
            (np.abs(chunk['months_to_handover_signed']) <= 6).astype(int),
            0
        )
        
        # Project percent complete (from projects table)
        chunk['project_percent_complete'] = chunk['percent_completed']
        
        # Phase ratio (0.0 = just launched, 1.0 = at handover)
        chunk['phase_ratio'] = np.where(
            chunk['project_duration_months'].notna() & (chunk['project_duration_months'] > 0),
            (chunk['months_since_launch'] / chunk['project_duration_months']).clip(0, 1),
            np.nan
        )
        
        # Resolve developer_name with a strict "no Unknown" policy:
        # - If merge found developer_name, keep it
        # - Else fallback to master_project_en -> master_developer_name
        # - Else fallback to explicit market bucket "ALL_DEVELOPERS"
        chunk['master_project_en'] = chunk['master_project_en'].astype(str).str.strip()
        chunk['developer_name'] = chunk['developer_name'].astype(str).str.strip()
        # If merge produced "nan" string, treat as missing
        chunk.loc[chunk['developer_name'].str.lower().isin(['nan', 'none', '']), 'developer_name'] = np.nan
        chunk.loc[chunk['master_project_en'].str.lower().isin(['nan', 'none', '']), 'master_project_en'] = np.nan

        # Fallback using master_project mapping (covers most Ready transactions missing project_number)
        missing_dev = chunk['developer_name'].isna()
        if missing_dev.any():
            chunk.loc[missing_dev, 'developer_name'] = chunk.loc[missing_dev, 'master_project_en'].map(master_map)

        # Final fallback: explicit market bucket (never Unknown)
        chunk['developer_name'] = chunk['developer_name'].fillna('ALL_DEVELOPERS')

        # Canonicalize developer names (dedupe variants) where possible
        if dev_norm_map:
            keys = chunk['developer_name'].astype(str).str.strip().apply(lambda x: ''.join(str(x).split()))
            chunk['developer_name'] = keys.map(dev_norm_map).fillna(chunk['developer_name'])

        # Preserve registered developer before brand override (audit + downstream explainability)
        chunk['developer_registered_name'] = chunk['developer_name']

        # 0) Explicit brand consolidations for registered entities (e.g., DAMAC)
        # Use object dtype to avoid pandas dtype warnings when assigning strings.
        chunk['developer_brand'] = pd.Series([pd.NA] * len(chunk), index=chunk.index, dtype="object")
        if consolidation_map:
            cons = chunk['developer_registered_name'].map(consolidation_map)
            has_cons = cons.notna()
            if has_cons.any():
                chunk.loc[has_cons, 'developer_brand'] = cons.loc[has_cons]
                chunk.loc[has_cons, 'developer_name'] = cons.loc[has_cons]

        # Infer brand developer from raw name fields (project/building names), gated by registered-under rules.
        # If inferred, use brand as the modeling/trend developer key (so Binghatti/Danube become first-class series).
        if brand_rules:
            chunk['developer_brand'] = _infer_developer_brand_series(chunk, brand_rules)
            has_brand = chunk['developer_brand'].notna()
            if has_brand.any():
                chunk.loc[has_brand, 'developer_name'] = chunk.loc[has_brand, 'developer_brand']
        # else: keep existing developer_brand (from consolidation) if any
        
        filtered_rows += len(chunk)
        results.append(chunk)
    
    logger.info(f"  Processed {total_rows:,} rows, {filtered_rows:,} after filters")
    
    # Combine all chunks
    df = pd.concat(results, ignore_index=True)
    
    # Aggregate by group INCLUDING developer_name
    # Include median of phase columns for the group
    agg = df.groupby(['year_month', 'area_name_en', 'property_type_en', 'bedroom', 'reg_type', 'developer_name']).agg(
        median_price=('price_sqft', 'median'),
        transaction_count=('transaction_id', 'count'),
        # Project phase columns (median for the group)
        months_since_launch=('months_since_launch', 'median'),
        months_to_handover=('months_to_handover', 'median'),
        months_to_handover_signed=('months_to_handover_signed', 'median'),
        months_since_handover=('months_since_handover', 'median'),
        handover_window_6m=('handover_window_6m', 'median'),
        dld_offplan_after_handover=('dld_offplan_after_handover', 'max'),
        project_percent_complete=('project_percent_complete', 'median'),
        project_duration_months=('project_duration_months', 'median'),
        phase_ratio=('phase_ratio', 'median'),
        # Audit columns (stable within group in almost all cases)
        developer_registered_name=('developer_registered_name', _mode_or_nan),
        developer_brand=('developer_brand', _mode_or_nan)
    ).reset_index()
    
    # Rename columns
    agg = agg.rename(columns={
        'area_name_en': 'area_name',
        'property_type_en': 'property_type'
    })

    # Carry DLD reg_type label (audit only) by re-joining a stable mode per group/month.
    # This preserves what the source labeled the transaction as, while `reg_type` becomes lifecycle stage.
    try:
        dld_mode = df.groupby(['year_month', 'area_name_en', 'property_type_en', 'bedroom', 'reg_type', 'developer_name']).agg(
            reg_type_dld=('reg_type_dld', _mode_or_nan)
        ).reset_index()
        agg = agg.merge(
            dld_mode.rename(columns={'area_name_en': 'area_name', 'property_type_en': 'property_type'}),
            on=['year_month', 'area_name', 'property_type', 'bedroom', 'reg_type', 'developer_name'],
            how='left'
        )
    except Exception:
        agg['reg_type_dld'] = np.nan
    
    logger.info(f"  Aggregated to {len(agg):,} rows, {agg['area_name'].nunique()} areas, {agg['developer_name'].nunique()} developers")
    return agg


def compute_developer_overall_context(transactions: pd.DataFrame) -> pd.DataFrame:
    """
    Compute developer-overall statistics per month.
    
    This provides hierarchical context: how is this developer performing
    across ALL their projects (not just this specific area/bedroom)?
    
    Output columns:
    - year_month, developer_name
    - dev_overall_median_price, dev_overall_transactions
    """
    logger.info("Computing Developer-Overall Context...")
    
    # Aggregate across all areas/bedrooms for each developer
    dev_overall = transactions.groupby(['year_month', 'developer_name']).agg(
        dev_overall_median_price=('median_price', 'median'),
        dev_overall_transactions=('transaction_count', 'sum')
    ).reset_index()
    
    logger.info(f"  Computed context for {dev_overall['developer_name'].nunique()} developers")
    return dev_overall


def compute_market_overall_context(transactions: pd.DataFrame) -> pd.DataFrame:
    """
    Compute market-overall statistics per month.
    
    This provides hierarchical context: what is the entire market doing?
    
    Output columns:
    - year_month
    - market_median_price, market_transactions
    """
    logger.info("Computing Market-Overall Context...")
    
    # Aggregate across ALL groups for each month
    market_overall = transactions.groupby(['year_month']).agg(
        market_median_price=('median_price', 'median'),
        market_transactions=('transaction_count', 'sum')
    ).reset_index()
    
    logger.info(f"  Computed context for {len(market_overall)} months")
    return market_overall


def aggregate_rents(chunk_size: int = 200_000) -> pd.DataFrame:
    """
    Aggregate rent contracts to monthly by area/bedroom.
    
    Output columns:
    - year_month, area_name, bedroom
    - median_rent, rent_count, median_rent_sqft
    """
    logger.info("Aggregating Rent Contracts...")
    
    results = []
    total_rows = 0
    
    usecols = [
        'contract_start_date_parsed', 'annual_amount',
        'area_name_en', 'bedrooms', 'actual_area', 'property_usage_en'
    ]
    
    for chunk in pd.read_csv(
        CLEANED_DIR / 'Rent_Contracts_Cleaned.csv',
        chunksize=chunk_size,
        usecols=usecols
    ):
        total_rows += len(chunk)
        
        # Filter to Residential
        chunk = chunk[chunk['property_usage_en'] == 'Residential']
        
        # Parse date and create year_month
        chunk['date'] = pd.to_datetime(chunk['contract_start_date_parsed'], errors='coerce')
        chunk = chunk.dropna(subset=['date'])
        chunk['year_month'] = chunk['date'].dt.strftime('%Y-%m')
        
        # Filter valid rents
        # Some months/areas appear to have annual_amount in a 100x scale (likely fils vs AED).
        # Correct using rent_per_sqft sanity: if rent_sqft is implausibly high, divide annual_amount by 100.
        chunk['annual_amount'] = pd.to_numeric(chunk['annual_amount'], errors='coerce').astype(float)
        chunk = chunk[chunk['annual_amount'].notna()]
        chunk = chunk[(chunk['annual_amount'] > 0) & (chunk['annual_amount'] < 100_000_000)]

        # Compute provisional rent_sqft for unit correction
        area_ok = chunk['actual_area'].notna() & (chunk['actual_area'] > 0)
        provisional_rent_sqft = pd.Series(np.nan, index=chunk.index)
        provisional_rent_sqft.loc[area_ok] = chunk.loc[area_ok, 'annual_amount'] / chunk.loc[area_ok, 'actual_area']

        # If rent_sqft is wildly high, treat as 100x and rescale
        scale_fix = provisional_rent_sqft.notna() & (provisional_rent_sqft > 2_000)
        if scale_fix.any():
            chunk.loc[scale_fix, 'annual_amount'] = chunk.loc[scale_fix, 'annual_amount'] / 100.0
        
        # Compute rent per sqft where area available
        chunk['rent_sqft'] = np.where(
            chunk['actual_area'] > 0,
            chunk['annual_amount'] / chunk['actual_area'],
            np.nan
        )
        
        results.append(chunk)
    
    logger.info(f"  Processed {total_rows:,} rows")
    
    df = pd.concat(results, ignore_index=True)
    
    # Aggregate
    agg = df.groupby(['year_month', 'area_name_en', 'bedrooms']).agg(
        median_rent=('annual_amount', 'median'),
        rent_count=('annual_amount', 'count'),
        median_rent_sqft=('rent_sqft', 'median')
    ).reset_index()
    
    agg = agg.rename(columns={
        'area_name_en': 'area_name',
        'bedrooms': 'bedroom'
    })
    
    logger.info(f"  Aggregated to {len(agg):,} rows")
    return agg


def compute_supply(projects: pd.DataFrame, months: List[str]) -> pd.DataFrame:
    """
    Compute supply pipeline by area/month.
    
    For each month, count units/buildings/villas in active projects.
    
    Output columns:
    - year_month, area_name
    - supply_units, supply_buildings, supply_villas, active_projects
    """
    logger.info("Computing Supply Pipeline...")
    
    # Filter to projects with area
    active_proj = projects.dropna(subset=['area_name_en'])
    
    results = []
    
    for month in months:
        month_date = pd.to_datetime(month + '-01')
        
        # Projects active in this month:
        # - Status in ACTIVE_PROJECT_STATUSES
        # - AND (end_date > month OR end_date is null)
        mask = (
            active_proj['project_status'].isin(ACTIVE_PROJECT_STATUSES) &
            (
                (active_proj['project_end_date'].isna()) |
                (active_proj['project_end_date'] > month_date)
            )
        )
        
        active = active_proj[mask]
        
        # Aggregate by area
        if len(active) > 0:
            supply = active.groupby('area_name_en').agg(
                supply_units=('no_of_units', 'sum'),
                supply_buildings=('no_of_buildings', 'sum'),
                supply_villas=('no_of_villas', 'sum'),
                active_projects=('project_number', 'count')
            ).reset_index()
            supply['year_month'] = month
            supply = supply.rename(columns={'area_name_en': 'area_name'})
            results.append(supply)
    
    if results:
        df = pd.concat(results, ignore_index=True)
        logger.info(f"  Computed supply for {len(df):,} area-months")
        return df
    else:
        return pd.DataFrame(columns=['year_month', 'area_name', 'supply_units', 'supply_buildings', 'supply_villas', 'active_projects'])


def compute_developer_stats(projects: pd.DataFrame, months: List[str]) -> pd.DataFrame:
    """
    Compute time-varying developer statistics.
    
    For each developer/month, compute cumulative stats up to that month.
    
    Output columns:
    - year_month, developer_name
    - dev_total_projects, dev_completed_projects, dev_total_units, dev_avg_completion
    """
    logger.info("Computing Developer Statistics...")

    # Consolidate registered entities into brand where configured (e.g., DAMAC, EMAAR)
    consolidation_map = _load_developer_brand_consolidation_map()
    if consolidation_map:
        projects = projects.copy()
        projects['developer_name'] = (
            projects['developer_name']
            .astype(str).str.strip()
            .map(consolidation_map)
            .fillna(projects['developer_name'])
        )
    
    # Get unique developers
    developers = projects['developer_name'].dropna().unique()
    
    results = []
    
    for month in months:
        month_date = pd.to_datetime(month + '-01')
        
        for dev in developers:
            dev_projects = projects[projects['developer_name'] == dev]
            
            # Projects started by this month
            started = dev_projects[
                (dev_projects['project_start_date'].notna()) &
                (dev_projects['project_start_date'] <= month_date)
            ]
            
            if len(started) == 0:
                continue
            
            # Completed projects (FINISHED status and completion before month)
            completed = started[started['project_status'] == 'FINISHED']
            
            results.append({
                'year_month': month,
                'developer_name': dev,
                'dev_total_projects': len(started),
                'dev_completed_projects': len(completed),
                'dev_total_units': started['no_of_units'].sum(),
                'dev_avg_completion': started['percent_completed'].mean()
            })
    
    if results:
        df = pd.DataFrame(results)
        logger.info(f"  Computed stats for {df['developer_name'].nunique()} developers across {len(months)} months")
        return df
    else:
        return pd.DataFrame(columns=['year_month', 'developer_name', 'dev_total_projects', 'dev_completed_projects', 'dev_total_units', 'dev_avg_completion'])


def compute_supply_schedule(projects: pd.DataFrame, chunk_size: int = 200_000) -> pd.DataFrame:
    """
    Compute bedroom-level supply schedule from off-plan transactions.
    
    For each area × bedroom × completion_month, count how many off-plan units
    are scheduled to complete (hand over).
    
    This is a TIME-VARYING KNOWN feature because we know the handover schedule
    from off-plan contracts.
    
    Output columns:
    - year_month (= completion month), area_name, bedroom
    - units_completing: count of off-plan units completing this month
    """
    logger.info("Computing Bedroom-Level Supply Schedule...")
    
    # Create project to handover date mapping
    proj_handover = projects[['project_number', 'handover_date', 'area_name_en']].drop_duplicates()
    proj_handover = proj_handover.dropna(subset=['project_number', 'handover_date'])
    proj_handover['project_number'] = proj_handover['project_number'].astype(int).astype(str)
    proj_handover['handover_month'] = proj_handover['handover_date'].dt.strftime('%Y-%m')
    
    results = []
    total_offplan = 0
    
    usecols = [
        'transaction_id', 'reg_type_en', 'area_name_en', 'rooms_en', 
        'project_number', 'property_usage_en', 'property_type_en', 'instance_date_parsed'
    ]
    
    for chunk in pd.read_csv(
        CLEANED_DIR / 'Transactions_Cleaned.csv',
        chunksize=chunk_size,
        usecols=usecols
    ):
        # Filter to off-plan residential only
        chunk = chunk[chunk['reg_type_en'] == 'Off-Plan Properties']
        chunk = chunk[chunk['property_usage_en'] == 'Residential']
        chunk = chunk[chunk['property_type_en'].isin(VALID_PROPERTY_TYPES)]
        
        # Standardize bedrooms
        chunk['bedroom'] = chunk['rooms_en'].map(BEDROOM_MAP)
        chunk = chunk.dropna(subset=['bedroom'])
        
        # Link to project handover date
        chunk['project_number'] = pd.to_numeric(chunk['project_number'], errors='coerce')
        chunk['project_number'] = chunk['project_number'].fillna(-1).astype(int).astype(str)
        chunk['project_number'] = chunk['project_number'].replace('-1', None)
        
        chunk = chunk.merge(
            proj_handover[['project_number', 'handover_month', 'handover_date']], 
            on='project_number', 
            how='left'
        )
        
        # Keep only transactions with known handover dates
        chunk = chunk.dropna(subset=['handover_month', 'handover_date'])

        # Filter out DLD OffPlan transactions that occur after handover (post-handover transfers/late registrations),
        # as they should not contribute to future completion pipeline counts.
        chunk['transaction_date'] = pd.to_datetime(chunk['instance_date_parsed'], errors='coerce')
        chunk = chunk[chunk['transaction_date'].notna()]
        chunk = chunk[chunk['transaction_date'] < chunk['handover_date']]
        
        total_offplan += len(chunk)
        results.append(chunk[['handover_month', 'area_name_en', 'bedroom', 'transaction_id']])
    
    logger.info(f"  Found {total_offplan:,} off-plan transactions with handover dates")
    
    if not results:
        return pd.DataFrame(columns=['year_month', 'area_name', 'bedroom', 'units_completing'])
    
    # Combine and aggregate
    df = pd.concat(results, ignore_index=True)
    
    # Count units completing by handover_month × area × bedroom
    schedule = df.groupby(['handover_month', 'area_name_en', 'bedroom']).agg(
        units_completing=('transaction_id', 'count')
    ).reset_index()
    
    schedule = schedule.rename(columns={
        'handover_month': 'year_month',
        'area_name_en': 'area_name'
    })
    
    logger.info(f"  Created supply schedule: {len(schedule):,} area-bedroom-months")
    logger.info(f"  Date range: {schedule['year_month'].min()} to {schedule['year_month'].max()}")
    
    return schedule


def aggregate_valuations(chunk_size: int = 50_000) -> pd.DataFrame:
    """
    Aggregate government valuations to monthly by area.
    
    Output columns:
    - year_month, area_name
    - govt_valuation_median, valuation_count
    """
    logger.info("Aggregating Valuations...")
    
    results = []
    
    usecols = [
        'procedure_id', 'actual_worth', 'procedure_area',
        'property_type_en', 'area_name_en', 'instance_date_parsed'
    ]
    
    for chunk in pd.read_csv(
        CLEANED_DIR / 'Valuation_Cleaned.csv',
        chunksize=chunk_size,
        usecols=usecols
    ):
        # Parse date
        chunk['date'] = pd.to_datetime(chunk['instance_date_parsed'], errors='coerce')
        chunk = chunk.dropna(subset=['date'])
        chunk['year_month'] = chunk['date'].dt.strftime('%Y-%m')
        
        # Compute value per sqft
        chunk['value_sqft'] = chunk['actual_worth'] / chunk['procedure_area']
        chunk = chunk[(chunk['value_sqft'] > 0) & (chunk['value_sqft'] < 100000)]
        
        results.append(chunk)
    
    df = pd.concat(results, ignore_index=True)
    
    agg = df.groupby(['year_month', 'area_name_en']).agg(
        govt_valuation_median=('value_sqft', 'median'),
        valuation_count=('procedure_id', 'count')
    ).reset_index()
    
    agg = agg.rename(columns={'area_name_en': 'area_name'})
    
    logger.info(f"  Aggregated to {len(agg):,} rows")
    return agg


def aggregate_unit_registrations(chunk_size: int = 200_000) -> pd.DataFrame:
    """
    Count unit registrations by area/month.
    
    Output columns:
    - year_month, area_name
    - units_registered
    """
    logger.info("Aggregating Unit Registrations...")
    
    results = []
    
    usecols = ['property_id', 'area_name_en', 'creation_date_parsed']
    
    for chunk in pd.read_csv(
        CLEANED_DIR / 'Units_Cleaned.csv',
        chunksize=chunk_size,
        usecols=usecols
    ):
        chunk['date'] = pd.to_datetime(chunk['creation_date_parsed'], errors='coerce')
        chunk = chunk.dropna(subset=['date'])
        chunk['year_month'] = chunk['date'].dt.strftime('%Y-%m')
        
        results.append(chunk)
    
    df = pd.concat(results, ignore_index=True)
    
    agg = df.groupby(['year_month', 'area_name_en']).agg(
        units_registered=('property_id', 'count')
    ).reset_index()
    
    agg = agg.rename(columns={'area_name_en': 'area_name'})
    
    logger.info(f"  Aggregated to {len(agg):,} rows")
    return agg


def aggregate_building_registrations(chunk_size: int = 50_000) -> pd.DataFrame:
    """
    Aggregate building data by area/month.
    
    Output columns:
    - year_month, area_name
    - buildings_registered, avg_building_floors, avg_building_flats
    """
    logger.info("Aggregating Building Registrations...")
    
    results = []
    
    usecols = ['property_id', 'area_name_en', 'floors', 'flats', 'creation_date_parsed']
    
    for chunk in pd.read_csv(
        CLEANED_DIR / 'Buildings_Cleaned.csv',
        chunksize=chunk_size,
        usecols=usecols
    ):
        chunk['date'] = pd.to_datetime(chunk['creation_date_parsed'], errors='coerce')
        chunk = chunk.dropna(subset=['date'])
        chunk['year_month'] = chunk['date'].dt.strftime('%Y-%m')
        
        # Convert numeric columns
        chunk['floors'] = pd.to_numeric(chunk['floors'], errors='coerce')
        chunk['flats'] = pd.to_numeric(chunk['flats'], errors='coerce')
        
        results.append(chunk)
    
    df = pd.concat(results, ignore_index=True)
    
    agg = df.groupby(['year_month', 'area_name_en']).agg(
        buildings_registered=('property_id', 'count'),
        avg_building_floors=('floors', 'mean'),
        avg_building_flats=('flats', 'mean')
    ).reset_index()
    
    agg = agg.rename(columns={'area_name_en': 'area_name'})
    
    logger.info(f"  Aggregated to {len(agg):,} rows")
    return agg


def load_eibor() -> pd.DataFrame:
    """
    Load EIBOR rates - RAW VALUES ONLY, no computed features.
    
    Output columns:
    - year_month
    - eibor_overnight, eibor_1w, eibor_1m, eibor_3m, eibor_6m, eibor_12m
    """
    logger.info("Loading EIBOR (raw rates only)...")
    
    df = pd.read_csv(CLEANED_DIR / 'eibor_monthly.csv')
    
    # Convert year_month to YYYY-MM format
    df['year_month'] = pd.to_datetime(df['year_month']).dt.strftime('%Y-%m')
    
    # Keep ONLY raw rates - EXCLUDE all computed columns
    raw_cols = ['year_month', 'overnight', '1_week', '1_month', '3_month', '6_month', '12_month']
    df = df[raw_cols]
    
    # Rename to standard format
    df = df.rename(columns={
        'overnight': 'eibor_overnight',
        '1_week': 'eibor_1w',
        '1_month': 'eibor_1m',
        '3_month': 'eibor_3m',
        '6_month': 'eibor_6m',
        '12_month': 'eibor_12m'
    })
    
    logger.info(f"  Loaded {len(df)} months of EIBOR data")
    return df


def load_tourism() -> pd.DataFrame:
    """
    Load tourism data, expand quarterly to monthly.
    
    Output columns:
    - year_month
    - visitors_total, hotel_rooms, hotel_apartments
    """
    logger.info("Loading Tourism Data...")
    
    # Load visitors
    visitors = pd.read_csv(CLEANED_DIR / 'tourism_visitors.csv')
    # Filter to Total only
    visitors = visitors[visitors['region'] == 'Total']
    visitors = visitors[visitors['quarter'].notna()]  # Exclude yearly totals
    
    # Load inventory
    inventory = pd.read_csv(CLEANED_DIR / 'tourism_inventory.csv')
    inventory = inventory[inventory['quarter'].notna()]
    
    # Expand quarterly to monthly
    results = []
    
    for _, row in visitors.iterrows():
        year = int(row['year'])
        quarter = row['quarter']
        
        if quarter in QUARTER_TO_MONTHS:
            for month in QUARTER_TO_MONTHS[quarter]:
                results.append({
                    'year_month': f"{year}-{month}",
                    'visitors_total': row['visitors_thousands']
                })
    
    visitors_monthly = pd.DataFrame(results)
    
    # Process inventory similarly
    inv_results = []
    for _, row in inventory.iterrows():
        year = int(row['year'])
        quarter = row['quarter']
        
        if quarter in QUARTER_TO_MONTHS:
            for month in QUARTER_TO_MONTHS[quarter]:
                inv_results.append({
                    'year_month': f"{year}-{month}",
                    'hotel_rooms': row['num_hotel_rooms'],
                    'hotel_apartments': row['num_hotel_apartments']
                })
    
    inventory_monthly = pd.DataFrame(inv_results)
    
    # Merge visitors and inventory
    if len(visitors_monthly) > 0 and len(inventory_monthly) > 0:
        tourism = visitors_monthly.merge(inventory_monthly, on='year_month', how='outer')
    elif len(visitors_monthly) > 0:
        tourism = visitors_monthly
    elif len(inventory_monthly) > 0:
        tourism = inventory_monthly
    else:
        tourism = pd.DataFrame(columns=['year_month', 'visitors_total', 'hotel_rooms', 'hotel_apartments'])
    
    logger.info(f"  Loaded tourism data for {len(tourism)} months")
    return tourism


def merge_all_sources(
    transactions: pd.DataFrame,
    rents: pd.DataFrame,
    supply: pd.DataFrame,
    supply_schedule: pd.DataFrame,
    developer_stats: pd.DataFrame,
    valuations: pd.DataFrame,
    unit_regs: pd.DataFrame,
    building_regs: pd.DataFrame,
    eibor: pd.DataFrame,
    tourism: pd.DataFrame,
    dev_overall_context: pd.DataFrame,
    market_overall_context: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge all sources into single TFT dataset.
    
    Base: transactions (has area/property_type/bedroom/reg_type/developer granularity)
    Includes hierarchical context: developer-overall and market-overall.
    Includes bedroom-level supply schedule.
    """
    logger.info("Merging all sources...")
    
    df = transactions.copy()
    initial_rows = len(df)
    
    # Join rents (on area + bedroom only - rents don't have reg_type)
    df = df.merge(
        rents,
        on=['year_month', 'area_name', 'bedroom'],
        how='left'
    )
    logger.info(f"  After rents merge: {len(df):,} rows")
    
    # Join supply (on area only)
    df = df.merge(
        supply,
        on=['year_month', 'area_name'],
        how='left'
    )
    logger.info(f"  After supply merge: {len(df):,} rows")
    
    # Join supply schedule (bedroom-level, on area + bedroom)
    # This is TIME-VARYING KNOWN - we know when units will complete
    df = df.merge(
        supply_schedule,
        on=['year_month', 'area_name', 'bedroom'],
        how='left'
    )
    # Fill 0 for months with no completions
    df['units_completing'] = df['units_completing'].fillna(0).astype(int)
    logger.info(f"  After supply schedule merge: {len(df):,} rows")
    
    # Join developer stats (on developer_name + year_month)
    df = df.merge(
        developer_stats,
        on=['year_month', 'developer_name'],
        how='left'
    )
    logger.info(f"  After developer stats merge: {len(df):,} rows")
    
    # Join valuations (on area)
    df = df.merge(
        valuations,
        on=['year_month', 'area_name'],
        how='left'
    )
    logger.info(f"  After valuations merge: {len(df):,} rows")
    
    # Join unit registrations (on area)
    df = df.merge(
        unit_regs,
        on=['year_month', 'area_name'],
        how='left'
    )
    logger.info(f"  After unit regs merge: {len(df):,} rows")
    
    # Join building registrations (on area)
    df = df.merge(
        building_regs,
        on=['year_month', 'area_name'],
        how='left'
    )
    logger.info(f"  After building regs merge: {len(df):,} rows")
    
    # Join EIBOR (global, on year_month only)
    df = df.merge(
        eibor,
        on='year_month',
        how='left'
    )
    logger.info(f"  After EIBOR merge: {len(df):,} rows")
    
    # Join tourism (global, on year_month only)
    df = df.merge(
        tourism,
        on='year_month',
        how='left'
    )
    logger.info(f"  After tourism merge: {len(df):,} rows")
    
    # Join developer-overall context (hierarchical - how is this developer doing overall?)
    df = df.merge(
        dev_overall_context,
        on=['year_month', 'developer_name'],
        how='left'
    )
    logger.info(f"  After developer-overall context merge: {len(df):,} rows")
    
    # Join market-overall context (hierarchical - how is the entire market doing?)
    df = df.merge(
        market_overall_context,
        on='year_month',
        how='left'
    )
    logger.info(f"  After market-overall context merge: {len(df):,} rows")
    
    # Fill NaN for count columns with 0
    count_cols = [
        'supply_units', 'supply_buildings', 'supply_villas', 'active_projects',
        'valuation_count', 'units_registered', 'buildings_registered', 'rent_count'
    ]
    for col in count_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0).astype(int)

    # ============================================================
    # Missing value strategy (investor-grade):
    # - Avoid treating "missing" as 0 for macro/tourism/valuation/building features.
    # - Add explicit missingness flags and impute safely.
    # ============================================================

    # --- 1) Valuation + building structure: median-impute + missingness flag ---
    for col in ['govt_valuation_median', 'avg_building_floors', 'avg_building_flats']:
        if col not in df.columns:
            continue
        miss_col = f"{col}_missing"
        df[miss_col] = df[col].isna().astype(int)
        # Prefer area-level median (investor: local level matters), fall back to global median, then 0.
        area_median = df.groupby('area_name')[col].transform('median')
        global_median = df[col].median(skipna=True)
        df[col] = df[col].fillna(area_median)
        if pd.notna(global_median):
            df[col] = df[col].fillna(global_median)
        df[col] = df[col].fillna(0)

    # --- 2) Macro time series (EIBOR) and tourism: fill internal gaps, do NOT extrapolate beyond observed range ---
    macro_eibor = [c for c in ['eibor_overnight', 'eibor_1w', 'eibor_1m', 'eibor_3m', 'eibor_6m', 'eibor_12m'] if c in df.columns]
    macro_tourism = [c for c in ['visitors_total', 'hotel_rooms', 'hotel_apartments'] if c in df.columns]

    def _fill_month_macro_with_flags(cols: list[str], flag_name: str) -> None:
        if not cols or 'year_month' not in df.columns:
            return
        # Month-level series (one value per month)
        month_idx = pd.to_datetime(pd.Series(df['year_month'].unique()).sort_values(), errors='coerce')
        month_idx = month_idx.dropna().dt.strftime('%Y-%m').tolist()
        month_frame = df.groupby('year_month')[cols].first().reindex(month_idx)

        # Track observed range per column and fill internal gaps only
        for c in cols:
            s = month_frame[c]
            first = s.first_valid_index()
            last = s.last_valid_index()
            filled = s.ffill().bfill()
            if first is not None and last is not None:
                # wipe extrapolated edges
                mask_outside = (filled.index < first) | (filled.index > last)
                filled = filled.mask(mask_outside)
            month_frame[c] = filled

        # Flag missing months (any missing across cols)
        df[flag_name] = df['year_month'].map(month_frame[cols].isna().any(axis=1)).fillna(True).astype(int)

        # Final fill: keep values where known, set remaining missing to 0 (but missingness flag preserves semantics)
        month_frame = month_frame.fillna(0)
        for c in cols:
            df[c] = df['year_month'].map(month_frame[c]).fillna(0)

    _fill_month_macro_with_flags(macro_eibor, 'eibor_missing')
    _fill_month_macro_with_flags(macro_tourism, 'tourism_missing')
    
    logger.info(f"  Final merged dataset: {len(df):,} rows")
    return df


def create_tft_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add TFT-required columns:
    - time_idx: sequential integer per group
    - group_id: {area}_{property_type}_{bedroom}_{reg_type}_{developer}
    - month, quarter: calendar features
    - month_sin, month_cos: cyclical encoding
    """
    logger.info("Creating TFT columns...")
    
    # Create group_id (now includes developer for granular trends)
    df['group_id'] = (
        df['area_name'].str.replace(' ', '_').str.replace("'", "") + '_' +
        df['property_type'].str.replace(' ', '_') + '_' +
        df['bedroom'].str.replace(' ', '_') + '_' +
        df['reg_type'] + '_' +
        df['developer_name'].str.replace(' ', '_').str.replace("'", "").str.replace(",", "").str.replace(".", "")
    )
    
    # Extract calendar features
    df['date'] = pd.to_datetime(df['year_month'] + '-01')
    df['month'] = df['date'].dt.month
    df['quarter'] = df['date'].dt.quarter
    
    # Cyclical encoding
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # Create time_idx per group (sequential 0,1,2,...)
    df = df.sort_values(['group_id', 'year_month'])
    df['time_idx'] = df.groupby('group_id').cumcount()
    
    # Drop temp column
    df = df.drop(columns=['date'])
    
    logger.info(f"  Created {df['group_id'].nunique():,} unique groups")
    return df


def main():
    """Execute full TFT data build pipeline."""
    logger.info("=" * 60)
    logger.info("TFT DATA BUILD - START")
    logger.info("=" * 60)
    
    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Step 1: Load projects (needed for developer linking and supply)
    logger.info("\n[1/10] Loading Projects...")
    projects = load_projects()
    
    # Step 2: Aggregate transactions
    logger.info("\n[2/10] Aggregating Transactions...")
    transactions = aggregate_transactions(projects)
    
    # Step 3: Get all unique months from transactions
    all_months = sorted(transactions['year_month'].unique())
    logger.info(f"  Date range: {all_months[0]} to {all_months[-1]} ({len(all_months)} months)")
    
    # Step 4: Aggregate rents
    logger.info("\n[3/10] Aggregating Rent Contracts...")
    rents = aggregate_rents()
    
    # Step 5: Compute supply
    logger.info("\n[4/10] Computing Supply Pipeline...")
    supply = compute_supply(projects, all_months)
    
    # Step 6: Compute developer stats
    logger.info("\n[5/13] Computing Developer Statistics...")
    developer_stats = compute_developer_stats(projects, all_months)
    
    # Step 6b: Compute bedroom-level supply schedule
    logger.info("\n[6/13] Computing Bedroom-Level Supply Schedule...")
    supply_schedule = compute_supply_schedule(projects)
    
    # Step 7: Aggregate valuations
    logger.info("\n[7/13] Aggregating Valuations...")
    valuations = aggregate_valuations()
    
    # Step 8: Aggregate unit registrations
    logger.info("\n[7/10] Aggregating Unit Registrations...")
    unit_regs = aggregate_unit_registrations()
    
    # Step 9: Aggregate building registrations
    logger.info("\n[8/10] Aggregating Building Registrations...")
    building_regs = aggregate_building_registrations()
    
    # Step 10: Load EIBOR
    logger.info("\n[9/10] Loading EIBOR...")
    eibor = load_eibor()
    
    # Step 11: Load tourism
    logger.info("\n[10/12] Loading Tourism...")
    tourism = load_tourism()
    
    # Step 12: Compute hierarchical context columns
    logger.info("\n[11/12] Computing Developer-Overall Context...")
    dev_overall_context = compute_developer_overall_context(transactions)
    
    logger.info("\n[12/12] Computing Market-Overall Context...")
    market_overall_context = compute_market_overall_context(transactions)
    
    # Step 13: Merge all sources
    logger.info("\n[MERGE] Merging all sources...")
    df = merge_all_sources(
        transactions, rents, supply, supply_schedule, developer_stats,
        valuations, unit_regs, building_regs, eibor, tourism,
        dev_overall_context, market_overall_context
    )
    
    # Step 13: Create TFT columns
    logger.info("\n[TFT] Creating TFT columns...")
    df = create_tft_columns(df)
    
    # Step 14: Final column order
    columns = [
        # TFT required
        'time_idx', 'year_month', 'group_id',
        # Static categoricals
        'area_name', 'property_type', 'bedroom', 'reg_type', 'developer_name',
        # DLD reg type label (audit only; reg_type above is lifecycle stage)
        'reg_type_dld',
        # Developer audit (registered vs brand, when brand alias matched)
        'developer_registered_name', 'developer_brand',
        # Time-varying known (calendar)
        'month', 'quarter', 'month_sin', 'month_cos',
        # Time-varying known - supply schedule (we know when units will complete)
        'units_completing',
        # Time-varying unknown - prices (TARGET)
        'median_price', 'transaction_count',
        # Time-varying unknown - project phase (for off-plan appreciation tracking)
        'months_since_launch', 'months_to_handover',
        'months_to_handover_signed', 'months_since_handover', 'handover_window_6m',
        'dld_offplan_after_handover',
        'project_percent_complete', 'project_duration_months', 'phase_ratio',
        # Time-varying unknown - rents
        'median_rent', 'rent_count', 'median_rent_sqft',
        # Time-varying unknown - valuations
        'govt_valuation_median', 'valuation_count',
        'govt_valuation_median_missing',
        # Time-varying unknown - supply
        'supply_units', 'supply_buildings', 'supply_villas', 'active_projects',
        # Time-varying unknown - registrations
        'units_registered', 'buildings_registered',
        'avg_building_floors', 'avg_building_flats',
        'avg_building_floors_missing', 'avg_building_flats_missing',
        # Time-varying unknown - developer stats (from projects)
        'dev_total_projects', 'dev_completed_projects', 'dev_total_units', 'dev_avg_completion',
        # HIERARCHICAL CONTEXT - Developer Overall (how is this developer doing across ALL their projects?)
        'dev_overall_median_price', 'dev_overall_transactions',
        # HIERARCHICAL CONTEXT - Market Overall (how is the entire market doing?)
        'market_median_price', 'market_transactions',
        # Time-varying unknown - EIBOR (raw rates only)
        'eibor_overnight', 'eibor_1w', 'eibor_1m', 'eibor_3m', 'eibor_6m', 'eibor_12m',
        'eibor_missing',
        # Time-varying unknown - tourism
        'visitors_total', 'hotel_rooms', 'hotel_apartments',
        'tourism_missing'
    ]
    
    # Only include columns that exist
    final_columns = [c for c in columns if c in df.columns]
    df = df[final_columns]
    
    # Step 15: Save
    output_path = OUTPUT_DIR / 'tft_training_data.csv'
    df.to_csv(output_path, index=False)
    
    # Step 16: Save stats
    stats = {
        'total_rows': len(df),
        'unique_groups': df['group_id'].nunique(),
        'unique_areas': df['area_name'].nunique(),
        'date_range': {
            'start': df['year_month'].min(),
            'end': df['year_month'].max(),
            'months': df['year_month'].nunique()
        },
        'reg_type_distribution': df.groupby('reg_type')['group_id'].nunique().to_dict(),
        'columns': list(df.columns),
        'null_counts': df.isnull().sum().to_dict()
    }
    
    with open(OUTPUT_DIR / 'build_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)
    
    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("TFT DATA BUILD - COMPLETE")
    logger.info("=" * 60)
    logger.info(f"\nOutput: {output_path}")
    logger.info(f"Size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")
    logger.info(f"\nStats:")
    logger.info(f"  Total rows: {len(df):,}")
    logger.info(f"  Unique groups: {df['group_id'].nunique():,}")
    logger.info(f"  Unique areas: {df['area_name'].nunique()}")
    logger.info(f"  Date range: {df['year_month'].min()} to {df['year_month'].max()}")
    logger.info(f"  Columns: {len(df.columns)}")
    logger.info(f"\nReg type distribution:")
    for rt, count in df.groupby('reg_type')['group_id'].nunique().items():
        logger.info(f"  {rt}: {count:,} groups")
    logger.info(f"\nNull counts (top 10):")
    null_counts = df.isnull().sum().sort_values(ascending=False)
    for col, count in null_counts.head(10).items():
        if count > 0:
            logger.info(f"  {col}: {count:,} ({count/len(df)*100:.1f}%)")


if __name__ == '__main__':
    main()

