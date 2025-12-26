"""
Master Data Cleaning Script

Cleans all datasets for model training:
1. Rent_Contracts.csv - Bedroom parsing, date validation, residential filtering
2. Transactions.csv - Entity resolution, date validation
3. Projects.csv - Date parsing, status standardization
4. Apply entity resolution mappings to all datasets

Run this script before model training to ensure data quality.
"""

import pandas as pd
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import json
import sys

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent))

from bedroom_mapping import parse_bedrooms, get_bedroom_numeric

# Minimal residential/property type constants used for filtering
PROPERTY_TYPES = ["Unit", "Villa"]
RESIDENTIAL_USAGE = "Residential"

def load_entity_mappings(profiles_dir: Path) -> dict:
    """Load entity resolution mappings."""
    mappings = {}
    
    entity_dir = profiles_dir / "entity_resolution"
    if not entity_dir.exists():
        print("WARNING: Entity resolution directory not found. Skipping entity resolution.")
        return mappings
    
    # Load area mappings (case standardization)
    canonical_areas_path = entity_dir / "canonical_areas.json"
    if canonical_areas_path.exists():
        with open(canonical_areas_path) as f:
            canonical_areas = json.load(f)
        # Create case-insensitive mapping
        mappings['areas'] = {area.upper(): area for area in canonical_areas}
        print(f"Loaded {len(mappings['areas'])} canonical area names")
    
    # Load project fuzzy mappings
    fuzzy_path = entity_dir / "project_fuzzy_mapping.json"
    if fuzzy_path.exists():
        with open(fuzzy_path) as f:
            mappings['projects'] = json.load(f)
        print(f"Loaded {len(mappings['projects'])} project fuzzy mappings")
    
    return mappings


def standardize_area_name(area: str, mapping: dict) -> str:
    """Standardize area name using canonical mapping."""
    if pd.isna(area):
        return area
    
    area_upper = str(area).strip().upper()
    return mapping.get(area_upper, area)


def apply_project_mapping(project: str, mapping: dict) -> str:
    """Apply project fuzzy mapping."""
    if pd.isna(project):
        return project
    
    project_str = str(project).strip()
    return mapping.get(project_str, project_str)


def clean_transactions(input_path: Path, output_path: Path, mappings: dict, chunk_size: int = 100_000):
    """Clean Transactions.csv."""
    print("\n" + "=" * 60)
    print("CLEANING TRANSACTIONS")
    print("=" * 60)
    
    stats = {
        "total_rows": 0,
        "valid_rows": 0,
        "invalid_date_rows": 0,
        "area_standardized": 0,
        "duplicates_removed": 0,
        "residential_filtered": 0,
        "missing_price_or_area_removed": 0,
    }
    
    cleaned_chunks = []
    
    chunk_iter = pd.read_csv(
        input_path,
        chunksize=chunk_size,
        low_memory=False,
        on_bad_lines='warn'
    )
    
    for chunk in tqdm(chunk_iter, desc="Processing chunks"):
        stats["total_rows"] += len(chunk)
        
        # Parse dates (flexible): try day-first parse, then fall back to year-first parsing
        chunk['instance_date_parsed'] = pd.to_datetime(
            chunk['instance_date'],
            dayfirst=True,
            errors='coerce',
            infer_datetime_format=True
        )

        # Second pass for values that still failed: try year-first / alternative parsing
        mask_unparsed = chunk['instance_date_parsed'].isna() & chunk['instance_date'].notna()
        if mask_unparsed.any():
            chunk.loc[mask_unparsed, 'instance_date_parsed'] = pd.to_datetime(
                chunk.loc[mask_unparsed, 'instance_date'],
                dayfirst=False,
                errors='coerce',
                infer_datetime_format=True
            )

        # Populate year/month where available (keep rows even if date parsing failed)
        chunk['transaction_year'] = chunk['instance_date_parsed'].dt.year
        chunk['transaction_month'] = chunk['instance_date_parsed'].dt.month

        # Count rows where a non-empty instance_date could not be parsed (but DO NOT drop them)
        stats["invalid_date_rows"] += int(((chunk['instance_date_parsed'].isna()) & (chunk['instance_date'].notna())).sum())

        # Standardize whitespace in column names (do not rename semantics)
        chunk.columns = [c.strip() if isinstance(c, str) else c for c in chunk.columns]

        # Remove obvious duplicate rows (prefer using transaction_id when available)
        if 'transaction_id' in chunk.columns:
            before = len(chunk)
            chunk = chunk.drop_duplicates(subset=['transaction_id'])
            stats['duplicates_removed'] += before - len(chunk)
        else:
            before = len(chunk)
            chunk = chunk.drop_duplicates()
            stats['duplicates_removed'] += before - len(chunk)

        # Filter to residential/property types only (exclude commercial/land)
        before = len(chunk)
        if 'property_usage_en' in chunk.columns:
            chunk = chunk[chunk['property_usage_en'] == RESIDENTIAL_USAGE]
        elif 'property_type_en' in chunk.columns:
            chunk = chunk[chunk['property_type_en'].isin(PROPERTY_TYPES)]
        stats['residential_filtered'] += before - len(chunk)

        # Handle missing critical values: actual_worth and procedure_area required for price calculations
        if 'actual_worth' in chunk.columns and 'procedure_area' in chunk.columns:
            before = len(chunk)
            chunk = chunk[chunk['actual_worth'].notna() & chunk['procedure_area'].notna() & (chunk['procedure_area'] > 0)]
            stats['missing_price_or_area_removed'] += before - len(chunk)
        
        # Apply entity resolution for areas
        if 'areas' in mappings:
            original_areas = chunk['area_name_en'].copy()
            chunk['area_name_en'] = chunk['area_name_en'].apply(
                lambda x: standardize_area_name(x, mappings['areas'])
            )
            stats["area_standardized"] += (original_areas != chunk['area_name_en']).sum()
        
        # Apply project mapping
        if 'projects' in mappings and 'master_project_en' in chunk.columns:
            chunk['master_project_en'] = chunk['master_project_en'].apply(
                lambda x: apply_project_mapping(x, mappings['projects'])
            )
        
        stats["valid_rows"] += len(chunk)
        cleaned_chunks.append(chunk)
    
    # Combine and save
    cleaned_df = pd.concat(cleaned_chunks, ignore_index=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned_df.to_csv(output_path, index=False)
    
    print(f"Total rows: {stats['total_rows']:,}")
    print(f"Valid rows: {stats['valid_rows']:,}")
    print(f"Invalid dates removed: {stats['invalid_date_rows']:,}")
    print(f"Areas standardized: {stats['area_standardized']:,}")
    print(f"Saved to: {output_path}")
    
    return stats


def clean_projects(input_path: Path, output_path: Path, mappings: dict):
    """Clean Projects.csv."""
    print("\n" + "=" * 60)
    print("CLEANING PROJECTS")
    print("=" * 60)
    
    df = pd.read_csv(input_path, low_memory=False)
    print(f"Loaded {len(df):,} projects")
    
    stats = {
        "total_rows": len(df),
        "area_standardized": 0,
        "project_mapped": 0,
    }
    
    # Parse dates
    for date_col in ['project_start_date', 'project_end_date', 'completion_date']:
        if date_col in df.columns:
            df[f'{date_col}_parsed'] = pd.to_datetime(
                df[date_col],
                format='%d-%m-%Y',
                errors='coerce'
            )
    
    # Extract completion year/quarter for future supply analysis
    if 'project_end_date_parsed' in df.columns:
        df['expected_completion_year'] = df['project_end_date_parsed'].dt.year
        df['expected_completion_quarter'] = df['project_end_date_parsed'].dt.quarter
    
    # Apply entity resolution
    if 'areas' in mappings:
        original_areas = df['area_name_en'].copy()
        df['area_name_en'] = df['area_name_en'].apply(
            lambda x: standardize_area_name(x, mappings['areas'])
        )
        stats["area_standardized"] = (original_areas != df['area_name_en']).sum()
    
    if 'projects' in mappings and 'master_project_en' in df.columns:
        original_projects = df['master_project_en'].copy()
        df['master_project_en'] = df['master_project_en'].apply(
            lambda x: apply_project_mapping(x, mappings['projects'])
        )
        stats["project_mapped"] = (original_projects != df['master_project_en']).sum()
    
    # Standardize project status
    if 'project_status' in df.columns:
        df['project_status'] = df['project_status'].str.strip().str.upper()
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"Areas standardized: {stats['area_standardized']:,}")
    print(f"Projects mapped: {stats['project_mapped']:,}")
    print(f"Saved to: {output_path}")
    
    return stats


def clean_units(input_path: Path, output_path: Path, mappings: dict, chunk_size: int = 100_000):
    """Clean Units.csv with proper date parsing."""
    print("\n" + "=" * 60)
    print("CLEANING UNITS")
    print("=" * 60)
    
    stats = {
        "total_rows": 0,
        "area_standardized": 0,
        "dates_parsed": 0,
        "dates_failed": 0,
    }
    
    cleaned_chunks = []
    
    chunk_iter = pd.read_csv(
        input_path,
        chunksize=chunk_size,
        low_memory=False,
        on_bad_lines='warn'
    )
    
    for chunk in tqdm(chunk_iter, desc="Processing chunks"):
        stats["total_rows"] += len(chunk)
        
        # Parse creation_date (DD-MM-YYYY format)
        if 'creation_date' in chunk.columns:
            chunk['creation_date_parsed'] = pd.to_datetime(
                chunk['creation_date'],
                format='%d-%m-%Y',
                errors='coerce'
            )
            chunk['creation_year'] = chunk['creation_date_parsed'].dt.year
            chunk['creation_month'] = chunk['creation_date_parsed'].dt.month
            chunk['creation_quarter'] = chunk['creation_date_parsed'].dt.quarter
            
            # Track parsing stats
            valid_dates = chunk['creation_date_parsed'].notna()
            stats["dates_parsed"] += valid_dates.sum()
            stats["dates_failed"] += (~valid_dates & chunk['creation_date'].notna()).sum()
        
        # Apply entity resolution for areas
        if 'areas' in mappings:
            original_areas = chunk['area_name_en'].copy()
            chunk['area_name_en'] = chunk['area_name_en'].apply(
                lambda x: standardize_area_name(x, mappings['areas'])
            )
            stats["area_standardized"] += (original_areas != chunk['area_name_en']).sum()
        
        # Apply project mapping
        if 'projects' in mappings and 'master_project_en' in chunk.columns:
            chunk['master_project_en'] = chunk['master_project_en'].apply(
                lambda x: apply_project_mapping(x, mappings['projects'])
            )
        
        cleaned_chunks.append(chunk)
    
    # Combine and save
    cleaned_df = pd.concat(cleaned_chunks, ignore_index=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned_df.to_csv(output_path, index=False)
    
    print(f"Total rows: {stats['total_rows']:,}")
    print(f"Dates parsed: {stats['dates_parsed']:,}")
    print(f"Dates failed: {stats['dates_failed']:,}")
    print(f"Areas standardized: {stats['area_standardized']:,}")
    print(f"Saved to: {output_path}")
    
    return stats


def clean_buildings(input_path: Path, output_path: Path, mappings: dict):
    """Clean Buildings.csv with proper date parsing."""
    print("\n" + "=" * 60)
    print("CLEANING BUILDINGS")
    print("=" * 60)
    
    df = pd.read_csv(input_path, low_memory=False)
    print(f"Loaded {len(df):,} buildings")
    
    stats = {
        "total_rows": len(df),
        "area_standardized": 0,
        "dates_parsed": 0,
        "dates_failed": 0,
    }
    
    # Parse creation_date (DD-MM-YYYY format)
    if 'creation_date' in df.columns:
        df['creation_date_parsed'] = pd.to_datetime(
            df['creation_date'],
            format='%d-%m-%Y',
            errors='coerce'
        )
        df['creation_year'] = df['creation_date_parsed'].dt.year
        df['creation_month'] = df['creation_date_parsed'].dt.month
        df['creation_quarter'] = df['creation_date_parsed'].dt.quarter
        
        # Track parsing stats
        valid_dates = df['creation_date_parsed'].notna()
        stats["dates_parsed"] = valid_dates.sum()
        stats["dates_failed"] = (~valid_dates & df['creation_date'].notna()).sum()
    
    # Apply entity resolution for areas
    if 'areas' in mappings and 'area_name_en' in df.columns:
        original_areas = df['area_name_en'].copy()
        df['area_name_en'] = df['area_name_en'].apply(
            lambda x: standardize_area_name(x, mappings['areas'])
        )
        stats["area_standardized"] = (original_areas != df['area_name_en']).sum()
    
    # Apply project mapping
    if 'projects' in mappings and 'master_project_en' in df.columns:
        df['master_project_en'] = df['master_project_en'].apply(
            lambda x: apply_project_mapping(x, mappings['projects'])
        )
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"Dates parsed: {stats['dates_parsed']:,}")
    print(f"Dates failed: {stats['dates_failed']:,}")
    print(f"Areas standardized: {stats['area_standardized']:,}")
    print(f"Saved to: {output_path}")
    
    return stats


def clean_valuation(input_path: Path, output_path: Path, mappings: dict):
    """Clean Valuation.csv with proper date parsing."""
    print("\n" + "=" * 60)
    print("CLEANING VALUATION")
    print("=" * 60)
    
    df = pd.read_csv(input_path, low_memory=False)
    print(f"Loaded {len(df):,} valuations")
    
    stats = {
        "total_rows": len(df),
        "area_standardized": 0,
        "dates_parsed": 0,
        "dates_failed": 0,
    }
    
    # Parse instance_date (DD-MM-YYYY format) - 100% populated!
    if 'instance_date' in df.columns:
        df['instance_date_parsed'] = pd.to_datetime(
            df['instance_date'],
            format='%d-%m-%Y',
            errors='coerce'
        )
        df['valuation_year'] = df['instance_date_parsed'].dt.year
        df['valuation_month'] = df['instance_date_parsed'].dt.month
        df['valuation_quarter'] = df['instance_date_parsed'].dt.quarter
        
        # Track parsing stats
        valid_dates = df['instance_date_parsed'].notna()
        stats["dates_parsed"] = valid_dates.sum()
        stats["dates_failed"] = (~valid_dates & df['instance_date'].notna()).sum()
    
    # Apply entity resolution for areas
    if 'areas' in mappings and 'area_name_en' in df.columns:
        original_areas = df['area_name_en'].copy()
        df['area_name_en'] = df['area_name_en'].apply(
            lambda x: standardize_area_name(x, mappings['areas'])
        )
        stats["area_standardized"] = (original_areas != df['area_name_en']).sum()
    
    # Apply project mapping
    if 'projects' in mappings and 'master_project_en' in df.columns:
        df['master_project_en'] = df['master_project_en'].apply(
            lambda x: apply_project_mapping(x, mappings['projects'])
        )
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"Dates parsed: {stats['dates_parsed']:,}")
    print(f"Dates failed: {stats['dates_failed']:,}")
    print(f"Areas standardized: {stats['area_standardized']:,}")
    print(f"Saved to: {output_path}")
    
    return stats


def main():
    """Run all data cleaning."""
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "Data"
    cleaned_dir = data_dir / "cleaned"
    profiles_dir = project_root / "data_profiles"
    
    print("=" * 60)
    print("MASTER DATA CLEANING")
    print(f"Started at: {datetime.now().isoformat()}")
    print("=" * 60)
    
    # Load entity mappings
    mappings = load_entity_mappings(profiles_dir)
    
    all_stats = {}
    
    # Clean Rent Contracts (using dedicated script)
    from clean_rent_contracts import clean_rent_contracts
    rent_stats = clean_rent_contracts(
        str(data_dir / "Rent_Contracts.csv"),
        str(cleaned_dir / "Rent_Contracts_Cleaned.csv"),
        save_commercial=True
    )
    all_stats['rent_contracts'] = rent_stats
    
    # Clean Transactions
    if (data_dir / "Transactions.csv").exists():
        trans_stats = clean_transactions(
            data_dir / "Transactions.csv",
            cleaned_dir / "Transactions_Cleaned.csv",
            mappings
        )
        all_stats['transactions'] = trans_stats
    
    # Clean Projects
    if (data_dir / "Projects.csv").exists():
        proj_stats = clean_projects(
            data_dir / "Projects.csv",
            cleaned_dir / "Projects_Cleaned.csv",
            mappings
        )
        all_stats['projects'] = proj_stats
    
    # Clean Units
    if (data_dir / "Units.csv").exists():
        unit_stats = clean_units(
            data_dir / "Units.csv",
            cleaned_dir / "Units_Cleaned.csv",
            mappings
        )
        all_stats['units'] = unit_stats
    
    # Clean Buildings (with proper date parsing)
    if (data_dir / "Buildings.csv").exists():
        building_stats = clean_buildings(
            data_dir / "Buildings.csv",
            cleaned_dir / "Buildings_Cleaned.csv",
            mappings
        )
        all_stats['buildings'] = building_stats
    
    # Clean Valuation (with proper date parsing)
    if (data_dir / "Valuation.csv").exists():
        valuation_stats = clean_valuation(
            data_dir / "Valuation.csv",
            cleaned_dir / "Valuation_Cleaned.csv",
            mappings
        )
        all_stats['valuation'] = valuation_stats
    
    # Save overall stats
    stats_path = cleaned_dir / "cleaning_stats.json"
    with open(stats_path, "w") as f:
        json.dump(all_stats, f, indent=2, default=str)
    
    print("\n" + "=" * 60)
    print("CLEANING COMPLETE")
    print(f"Finished at: {datetime.now().isoformat()}")
    print(f"Cleaned files saved to: {cleaned_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()

