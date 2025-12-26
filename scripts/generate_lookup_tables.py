"""
Generate Lookup Tables for LLM Integration

This script generates lookup tables from cleaned data for:
- Developer statistics (projects completed, units delivered, avg delay)
- Area statistics (12m/36m price changes, current median)
- Rent benchmarks (median rent by area/bedroom)
- Area mapping (all unique areas + abbreviations)
- Developer mapping (Arabic developers extracted for manual English mapping)

Output directory: Data/lookups/
"""

import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_CLEANED = PROJECT_ROOT / "Data" / "cleaned"
DATA_TFT = PROJECT_ROOT / "Data" / "tft"
OUTPUT_DIR = PROJECT_ROOT / "Data" / "lookups"


def generate_developer_stats():
    """
    Generate developer statistics from Projects_Cleaned.csv
    
    Columns:
    - developer_name: Arabic name
    - projects_total: Total projects
    - projects_completed: Completed projects (status = Finished)
    - projects_active: Active projects
    - total_units: Sum of all units
    - avg_completion_percent: Average completion %
    - avg_delay_months: Average delay (estimated from start/end dates)
    """
    logger.info("Generating developer_stats.csv...")
    
    projects_path = DATA_CLEANED / "Projects_Cleaned.csv"
    if not projects_path.exists():
        logger.error(f"Projects file not found: {projects_path}")
        return None
    
    df = pd.read_csv(projects_path, low_memory=False)
    logger.info(f"  Loaded {len(df):,} projects")
    
    # Parse dates
    for col in ['project_start_date_parsed', 'project_end_date_parsed']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    
    # Aggregate by developer
    stats = df.groupby('developer_name').agg(
        projects_total=('project_number', 'nunique'),
        projects_completed=('project_status', lambda x: (x == 'FINISHED').sum()),
        projects_active=('project_status', lambda x: (x != 'FINISHED').sum()),
        total_units=('no_of_units', 'sum'),
        avg_completion_percent=('percent_completed', 'mean'),
    ).reset_index()
    
    # Calculate completion rate
    stats['completion_rate'] = (stats['projects_completed'] / stats['projects_total'] * 100).round(1)
    
    # Calculate average project duration (for delay estimation)
    # Using end_date - start_date where both exist
    df['project_duration_months'] = (
        (df['project_end_date_parsed'] - df['project_start_date_parsed']).dt.days / 30
    ).fillna(0)
    
    duration_by_dev = df.groupby('developer_name')['project_duration_months'].mean().round(1)
    stats = stats.merge(
        duration_by_dev.rename('avg_duration_months').reset_index(),
        on='developer_name',
        how='left'
    )
    
    # Fill NA
    stats['avg_completion_percent'] = stats['avg_completion_percent'].fillna(0).round(1)
    stats['avg_delay_months'] = 0  # Would need actual handover dates vs projected
    
    # Sort by total units
    stats = stats.sort_values('total_units', ascending=False)
    
    output_path = OUTPUT_DIR / "developer_stats.csv"
    stats.to_csv(output_path, index=False)
    logger.info(f"  Saved {len(stats):,} developers to {output_path}")
    
    return stats


def generate_area_stats():
    """
    Generate area statistics from Transactions_Cleaned.csv
    
    Columns:
    - area_name: Area name
    - current_median_sqft: Current median price per sqft
    - price_change_12m: 12-month price change %
    - price_change_36m: 36-month price change %
    - transaction_count_12m: Transactions in last 12 months
    - supply_pipeline: Units from active projects (from Projects)
    """
    logger.info("Generating area_stats.csv...")
    
    tx_path = DATA_CLEANED / "Transactions_Cleaned.csv"
    if not tx_path.exists():
        logger.error(f"Transactions file not found: {tx_path}")
        return None
    
    df = pd.read_csv(tx_path, low_memory=False)
    logger.info(f"  Loaded {len(df):,} transactions")
    
    # Filter to residential
    df = df[df['property_usage_en'] == 'Residential'].copy()
    
    # Parse date
    df['date'] = pd.to_datetime(df['instance_date_parsed'], errors='coerce')
    df['year_month'] = df['date'].dt.to_period('M')
    
    # Calculate price per sqft
    df['price_sqft'] = df['actual_worth'] / df['procedure_area'].replace(0, np.nan)
    df = df[df['price_sqft'].notna() & (df['price_sqft'] > 0) & (df['price_sqft'] < 50000)]
    
    # Get date boundaries
    latest_date = df['date'].max()
    date_12m_ago = latest_date - pd.DateOffset(months=12)
    date_36m_ago = latest_date - pd.DateOffset(months=36)
    
    # Current period (last 3 months) for "current" median
    current_start = latest_date - pd.DateOffset(months=3)
    
    results = []
    
    for area in df['area_name_en'].unique():
        area_df = df[df['area_name_en'] == area]
        
        # Current median (last 3 months)
        current_df = area_df[area_df['date'] >= current_start]
        current_median = current_df['price_sqft'].median() if len(current_df) > 0 else np.nan
        
        # 12 months ago median
        df_12m_ago = area_df[(area_df['date'] >= date_12m_ago - pd.DateOffset(months=3)) & 
                            (area_df['date'] < date_12m_ago)]
        median_12m_ago = df_12m_ago['price_sqft'].median() if len(df_12m_ago) > 0 else np.nan
        
        # 36 months ago median
        df_36m_ago = area_df[(area_df['date'] >= date_36m_ago - pd.DateOffset(months=3)) & 
                            (area_df['date'] < date_36m_ago)]
        median_36m_ago = df_36m_ago['price_sqft'].median() if len(df_36m_ago) > 0 else np.nan
        
        # Calculate changes
        change_12m = ((current_median - median_12m_ago) / median_12m_ago * 100) if pd.notna(median_12m_ago) and median_12m_ago > 0 else np.nan
        change_36m = ((current_median - median_36m_ago) / median_36m_ago * 100) if pd.notna(median_36m_ago) and median_36m_ago > 0 else np.nan
        
        # Transaction count last 12m
        tx_count_12m = len(area_df[area_df['date'] >= date_12m_ago])
        
        results.append({
            'area_name': area,
            'current_median_sqft': round(current_median, 2) if pd.notna(current_median) else None,
            'price_change_12m': round(change_12m, 1) if pd.notna(change_12m) else None,
            'price_change_36m': round(change_36m, 1) if pd.notna(change_36m) else None,
            'transaction_count_12m': tx_count_12m,
        })
    
    stats = pd.DataFrame(results)
    
    # Load supply pipeline from Projects
    projects_path = DATA_CLEANED / "Projects_Cleaned.csv"
    if projects_path.exists():
        projects = pd.read_csv(projects_path, low_memory=False)
        # Active projects (not Finished)
        active = projects[projects['project_status'] != 'FINISHED']
        supply = active.groupby('area_name_en')['no_of_units'].sum().reset_index()
        supply.columns = ['area_name', 'supply_pipeline']
        stats = stats.merge(supply, on='area_name', how='left')
        stats['supply_pipeline'] = stats['supply_pipeline'].fillna(0).astype(int)
    
    # Sort by transaction count
    stats = stats.sort_values('transaction_count_12m', ascending=False)
    
    output_path = OUTPUT_DIR / "area_stats.csv"
    stats.to_csv(output_path, index=False)
    logger.info(f"  Saved {len(stats):,} areas to {output_path}")
    
    return stats


def generate_rent_benchmarks():
    """
    Generate rent benchmarks from Rent_Contracts_Cleaned.csv
    
    Columns:
    - area_name: Area name
    - bedrooms: Bedroom count
    - median_annual_rent: Median annual rent
    - rent_count: Number of contracts
    - median_rent_sqft: Median rent per sqft
    """
    logger.info("Generating rent_benchmarks.csv...")
    
    rent_path = DATA_CLEANED / "Rent_Contracts_Cleaned.csv"
    if not rent_path.exists():
        logger.error(f"Rent contracts file not found: {rent_path}")
        return None
    
    df = pd.read_csv(rent_path, low_memory=False)
    logger.info(f"  Loaded {len(df):,} rent contracts")
    
    # Filter residential
    if 'property_usage_en' in df.columns:
        df = df[df['property_usage_en'] == 'Residential'].copy()
    
    # Parse date and filter to recent 12 months
    df['date'] = pd.to_datetime(df['contract_start_date_parsed'], errors='coerce')
    latest_date = df['date'].max()
    date_12m_ago = latest_date - pd.DateOffset(months=12)
    df = df[df['date'] >= date_12m_ago]
    
    # Calculate rent per sqft
    df['rent_sqft'] = df['annual_amount'] / df['actual_area'].replace(0, np.nan)
    
    # Aggregate by area and bedroom
    stats = df.groupby(['area_name_en', 'bedrooms']).agg(
        median_annual_rent=('annual_amount', 'median'),
        rent_count=('annual_amount', 'count'),
        median_rent_sqft=('rent_sqft', 'median')
    ).reset_index()
    
    stats.columns = ['area_name', 'bedrooms', 'median_annual_rent', 'rent_count', 'median_rent_sqft']
    
    # Round values
    stats['median_annual_rent'] = stats['median_annual_rent'].round(0)
    stats['median_rent_sqft'] = stats['median_rent_sqft'].round(2)
    
    # Sort
    stats = stats.sort_values(['area_name', 'bedrooms'])
    
    output_path = OUTPUT_DIR / "rent_benchmarks.csv"
    stats.to_csv(output_path, index=False)
    logger.info(f"  Saved {len(stats):,} rent benchmarks to {output_path}")
    
    return stats


def generate_area_mapping():
    """
    Generate area mapping with all unique areas and common abbreviations.
    
    Output: JSON with:
    - abbreviations: Common abbreviations to full names
    - all_areas: List of all unique area names
    """
    logger.info("Generating area_mapping.json...")
    
    # Load areas from TFT data + lookups for broader coverage
    all_area_set = set()
    tft_path = DATA_TFT / "tft_training_data.csv"
    if tft_path.exists():
        df = pd.read_csv(tft_path, low_memory=False, usecols=['area_name'])
        all_area_set.update(df['area_name'].dropna().unique().tolist())
    else:
        logger.warning(f"TFT data file not found: {tft_path} (will still build mapping from lookups)")

    area_stats_path = OUTPUT_DIR / "area_stats.csv"
    if area_stats_path.exists():
        df_area = pd.read_csv(area_stats_path, low_memory=False, usecols=['area_name'])
        all_area_set.update(df_area['area_name'].dropna().unique().tolist())

    all_areas = sorted(all_area_set)
    logger.info(f"  Found {len(all_areas)} unique areas (union of TFT + area_stats)")

    # Build mapping from reference file if available (source of truth)
    ref_path = OUTPUT_DIR / "area_reference.csv"
    abbreviations: dict = {}
    dld_to_common: dict = {}
    ambiguous_aliases: dict = {}

    def _clean_alias(x) -> str:
        if x is None:
            return ""
        if isinstance(x, float) and pd.isna(x):
            return ""
        s = str(x).strip()
        if not s or s.lower() == "nan":
            return ""
        return s

    def _add_alias(alias: str, dld_area: str):
        if not alias:
            return
        key = alias.strip()
        if not key:
            return
        abbreviations[key] = dld_area

    if ref_path.exists():
        ref = pd.read_csv(ref_path, low_memory=False)
        # Ensure required columns exist
        needed = {"master_project_en", "area_name_en", "total_units", "project_count", "common_aliases"}
        missing = needed - set(ref.columns)
        if missing:
            logger.warning(f"  area_reference.csv missing columns: {missing}. Falling back to hardcoded abbreviations.")
        else:
            # Ensure all DLD areas from reference are included in all_areas for resolution
            all_area_set.update([a for a in ref["area_name_en"].dropna().unique().tolist()])
            all_areas = sorted(all_area_set)

            # Resolve ambiguous marketing names / aliases:
            # If a name maps to multiple DLD areas, do NOT treat it as a 100% abbreviation match.
            # Store it in ambiguous_aliases with a dominant target (highest total_units).
            ref2 = ref.dropna(subset=["master_project_en", "area_name_en"]).copy()
            ref2["total_units"] = pd.to_numeric(ref2.get("total_units"), errors="coerce").fillna(0)

            def dominant_target(dfsub: pd.DataFrame) -> str:
                dfsub = dfsub.sort_values("total_units", ascending=False)
                return str(dfsub.iloc[0]["area_name_en"]).strip()

            # 1) Marketing name ambiguity
            for mp, grp in ref2.groupby("master_project_en"):
                dld_set = set(grp["area_name_en"].tolist())
                mp_clean = _clean_alias(mp)
                if not mp_clean:
                    continue
                if len(dld_set) == 1:
                    _add_alias(mp_clean, list(dld_set)[0])
                else:
                    ambiguous_aliases[mp_clean] = {
                        "dominant": dominant_target(grp),
                        "candidates": sorted(list(dld_set)),
                    }

            # 2) Common aliases ambiguity
            # Explode comma-separated aliases and group by alias text
            rows = []
            for _, row in ref2.iterrows():
                dld = _clean_alias(row.get("area_name_en"))
                aliases = _clean_alias(row.get("common_aliases"))
                if not dld or not aliases:
                    continue
                for a in [x.strip() for x in aliases.split(",") if x.strip()]:
                    a_clean = _clean_alias(a)
                    if a_clean:
                        rows.append((a_clean, dld, float(row.get("total_units") or 0)))
            if rows:
                alias_df = pd.DataFrame(rows, columns=["alias", "area_name_en", "total_units"])
                for alias, grp in alias_df.groupby("alias"):
                    dld_set = set(grp["area_name_en"].tolist())
                    if len(dld_set) == 1:
                        _add_alias(alias, list(dld_set)[0])
                    else:
                        grp2 = grp.sort_values("total_units", ascending=False)
                        ambiguous_aliases[alias] = {
                            "dominant": str(grp2.iloc[0]["area_name_en"]).strip(),
                            "candidates": sorted(list(dld_set)),
                        }

            # Build dld_to_common display name: pick the master_project_en with highest total_units per DLD area
            tmp = ref.dropna(subset=["master_project_en", "area_name_en"]).copy()
            tmp["total_units"] = pd.to_numeric(tmp.get("total_units"), errors="coerce").fillna(0)
            # idxmax by total_units per area_name_en
            idx = tmp.groupby("area_name_en")["total_units"].idxmax()
            best = tmp.loc[idx, ["area_name_en", "master_project_en"]]
            for _, row in best.iterrows():
                dld = str(row["area_name_en"]).strip()
                disp = str(row["master_project_en"]).strip()
                if dld and disp:
                    dld_to_common[dld] = disp

            # Validate that mapping targets exist in all_areas (DLD names)
            invalid_targets = sorted({v for v in abbreviations.values() if v not in all_areas})
            if invalid_targets:
                logger.warning(f"  Found {len(invalid_targets)} abbreviation targets not present in all_areas. Example: {invalid_targets[:5]}")

            if ambiguous_aliases:
                logger.warning(f"  Found {len(ambiguous_aliases)} ambiguous aliases (one name maps to multiple DLD areas). These are NOT treated as 100% abbreviation matches.")
    else:
        logger.warning("  area_reference.csv not found; falling back to minimal abbreviations.")

    # Minimal fallback abbreviations (only if missing)
    minimal = {
        "JVC": "Al Barsha South Fourth",
        "JVT": "Al Barsha South Fifth",
        "JLT": "Al Thanyah Fifth",
        "BB": "Business Bay",
        "DM": "Marsa Dubai",
    }
    for k, v in minimal.items():
        abbreviations.setdefault(k, v)

    mapping = {
        "abbreviations": abbreviations,
        "dld_to_common": dld_to_common,
        "ambiguous_aliases": ambiguous_aliases,
        "all_areas": all_areas
    }
    
    output_path = OUTPUT_DIR / "area_mapping.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)
    
    logger.info(f"  Saved area mapping to {output_path}")
    
    return mapping


def generate_developer_mapping_template():
    """
    Extract all unique Arabic developer names from TFT data
    and create a template for manual English mapping.
    
    Output: JSON template that needs MANUAL editing to add English names.
    """
    logger.info("Generating developer_mapping.json template...")
    
    tft_path = DATA_TFT / "tft_training_data.csv"
    if not tft_path.exists():
        logger.error(f"TFT data file not found: {tft_path}")
        return None
    
    df = pd.read_csv(tft_path, low_memory=False, usecols=['developer_name'])
    developers = sorted(df['developer_name'].unique().tolist())
    
    # Remove "Unknown" from list
    developers = [d for d in developers if d != "Unknown"]
    
    logger.info(f"  Found {len(developers)} unique developers (excluding 'Unknown')")
    
    # Create template with known mappings
    known_mappings = {
        "اعمار العقارية ش. م. ع": {"english": "Emaar Properties", "aliases": ["Emaar", "EMAAR"]},
        "بن غاتي للتطوير العقاري": {"english": "Binghatti Developers", "aliases": ["Binghatti"]},
        "داماك العقارية": {"english": "Damac Properties", "aliases": ["Damac", "DAMAC"]},
        "شركة نخيل (شمخ)": {"english": "Nakheel", "aliases": ["Nakheel PJSC"]},
        "سوبها": {"english": "Sobha Realty", "aliases": ["Sobha"]},
        "مراس القابضة": {"english": "Meraas", "aliases": ["Meraas Holding"]},
        "دبي للعقارات": {"english": "Dubai Properties", "aliases": ["DP"]},
        "عزيزي للتطوير العقاري": {"english": "Azizi Developments", "aliases": ["Azizi"]},
        "دانوب العقارية": {"english": "Danube Properties", "aliases": ["Danube"]},
        "ماج للتطوير العقاري": {"english": "MAG Property Development", "aliases": ["MAG"]},
        "سيليكت جروب": {"english": "Select Group", "aliases": ["Select"]},
        "إلينغتون العقارية": {"english": "Ellington Properties", "aliases": ["Ellington"]},
        "قرية جميرا (شذمم)": {"english": "Jumeirah Golf Estates", "aliases": ["JGE"]},
        "الفرجان ( شذمم )": {"english": "Al Furjan", "aliases": []},
        "أورا للتطوير العقاري": {"english": "Ora Developers", "aliases": ["Ora"]},
        "صمعان العقارية": {"english": "Samana Developers", "aliases": ["Samana"]},
    }
    
    mappings = []
    for dev in developers:
        if dev in known_mappings:
            mappings.append({
                "arabic": dev,
                "english": known_mappings[dev]["english"],
                "aliases": known_mappings[dev]["aliases"]
            })
        else:
            # Template entry - needs manual filling
            mappings.append({
                "arabic": dev,
                "english": "",  # NEEDS MANUAL ENTRY
                "aliases": []
            })
    
    # Sort by whether english is filled (known ones first)
    mappings = sorted(mappings, key=lambda x: (x["english"] == "", x["arabic"]))
    
    output = {
        "_instructions": "Fill in the 'english' field for each Arabic developer name. Add common aliases.",
        "_total_developers": len(mappings),
        "_mapped": len([m for m in mappings if m["english"]]),
        "_unmapped": len([m for m in mappings if not m["english"]]),
        "mappings": mappings
    }
    
    output_path = OUTPUT_DIR / "developer_mapping.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    logger.info(f"  Saved developer mapping template to {output_path}")
    logger.info(f"  NOTE: {output['_unmapped']} developers need MANUAL English name mapping!")
    
    return output


def main():
    """Generate all lookup tables."""
    logger.info("=" * 60)
    logger.info("GENERATING LOOKUP TABLES FOR LLM INTEGRATION")
    logger.info("=" * 60)
    
    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Generate all tables
    generate_developer_stats()
    print()
    
    generate_area_stats()
    print()
    
    generate_rent_benchmarks()
    print()
    
    generate_area_mapping()
    print()
    
    generate_developer_mapping_template()
    print()
    
    logger.info("=" * 60)
    logger.info("LOOKUP TABLE GENERATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"\nOutput files in: {OUTPUT_DIR}")
    logger.info("\n⚠️  IMPORTANT: developer_mapping.json needs MANUAL editing!")
    logger.info("   - Open the file and fill in English names for Arabic developers")
    logger.info("   - Add common aliases (e.g., 'Emaar', 'EMAAR' for Emaar Properties)")


if __name__ == "__main__":
    main()
