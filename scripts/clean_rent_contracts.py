"""
Rent Contracts Data Cleaning Script

Applies all necessary cleaning to Rent_Contracts.csv:
1. Bedroom parsing using the mapping table
2. Date validation (filter invalid years)
3. Residential/commercial filtering
4. Standardized output columns

Output: Cleaned CSV ready for model training
"""

import pandas as pd
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import json

# Import bedroom mapping
from bedroom_mapping import parse_bedrooms, get_bedroom_numeric, is_residential, BEDROOM_MAPPING


def clean_rent_contracts(
    input_path: str,
    output_path: str,
    chunk_size: int = 100_000,
    save_commercial: bool = False
):
    """
    Clean Rent_Contracts.csv for model training.
    
    Args:
        input_path: Path to raw Rent_Contracts.csv
        output_path: Path to save cleaned CSV
        chunk_size: Rows per chunk for memory efficiency
        save_commercial: If True, also save commercial properties separately
    """
    print(f"Cleaning {input_path}...")
    
    # Statistics tracking
    stats = {
        "total_rows": 0,
        "residential_rows": 0,
        "commercial_rows": 0,
        "invalid_date_rows": 0,
        "bedroom_distribution": {},
        "year_distribution": {},
        "unmapped_subtypes": set(),
    }
    
    # Process in chunks
    cleaned_chunks = []
    commercial_chunks = []
    
    chunk_iter = pd.read_csv(
        input_path,
        chunksize=chunk_size,
        low_memory=False,
        on_bad_lines='warn'
    )
    
    for chunk_num, chunk in enumerate(tqdm(chunk_iter, desc="Processing chunks")):
        stats["total_rows"] += len(chunk)
        
        # Apply bedroom parsing
        chunk['bedrooms'] = chunk['ejari_property_sub_type_en'].apply(parse_bedrooms)
        chunk['bedrooms_numeric'] = chunk['bedrooms'].apply(get_bedroom_numeric)
        
        # Track unmapped subtypes
        unmapped = chunk[chunk['bedrooms'].isna() & chunk['ejari_property_sub_type_en'].notna()]
        for subtype in unmapped['ejari_property_sub_type_en'].unique():
            if subtype not in BEDROOM_MAPPING:
                stats["unmapped_subtypes"].add(str(subtype))
        
        # Parse dates
        chunk['contract_start_date_parsed'] = pd.to_datetime(
            chunk['contract_start_date'], 
            format='%d-%m-%Y', 
            errors='coerce'
        )
        chunk['contract_end_date_parsed'] = pd.to_datetime(
            chunk['contract_end_date'], 
            format='%d-%m-%Y', 
            errors='coerce'
        )
        
        # Extract year
        chunk['contract_year'] = chunk['contract_start_date_parsed'].dt.year
        
        # Filter invalid dates (year > 2025 or < 1990)
        valid_dates = (
            (chunk['contract_year'] >= 1990) & 
            (chunk['contract_year'] <= 2025)
        ) | chunk['contract_year'].isna()
        
        invalid_count = (~valid_dates).sum()
        stats["invalid_date_rows"] += invalid_count
        chunk = chunk[valid_dates]
        
        # Split residential vs commercial
        residential_mask = chunk['bedrooms'].notna()
        residential = chunk[residential_mask].copy()
        commercial = chunk[~residential_mask].copy()
        
        stats["residential_rows"] += len(residential)
        stats["commercial_rows"] += len(commercial)
        
        # Track distributions
        for bedroom in residential['bedrooms'].dropna():
            stats["bedroom_distribution"][bedroom] = stats["bedroom_distribution"].get(bedroom, 0) + 1
        
        for year in residential['contract_year'].dropna():
            year = int(year)
            stats["year_distribution"][year] = stats["year_distribution"].get(year, 0) + 1
        
        # Keep essential columns
        output_columns = [
            'contract_start_date',
            'contract_start_date_parsed',
            'contract_end_date',
            'contract_end_date_parsed',
            'contract_year',
            'annual_amount',
            'contract_amount',
            'area_name_en',
            'ejari_property_type_en',
            'ejari_property_sub_type_en',
            'bedrooms',
            'bedrooms_numeric',
            'property_usage_en',
            'ejari_business_type_en',
            'actual_area',
            'nearest_metro_en',
            'nearest_mall_en',
            'nearest_landmark_en',
            'version',
            'ren_contract_num',
        ]
        
        # Filter to existing columns
        output_columns = [c for c in output_columns if c in residential.columns]
        
        cleaned_chunks.append(residential[output_columns])
        if save_commercial:
            commercial_chunks.append(commercial[output_columns])
    
    # Combine chunks
    print("Combining chunks...")
    cleaned_df = pd.concat(cleaned_chunks, ignore_index=True)
    
    # Save cleaned data
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    cleaned_df.to_csv(output_path, index=False)
    print(f"Saved {len(cleaned_df):,} residential records to {output_path}")
    
    if save_commercial and commercial_chunks:
        commercial_df = pd.concat(commercial_chunks, ignore_index=True)
        commercial_path = output_path.parent / "Rent_Contracts_Commercial.csv"
        commercial_df.to_csv(commercial_path, index=False)
        print(f"Saved {len(commercial_df):,} commercial records to {commercial_path}")
    
    # Save statistics (convert numpy types to native Python for JSON serialization)
    stats["unmapped_subtypes"] = list(stats["unmapped_subtypes"])
    # Convert any numpy int64 values in year_distribution
    stats["year_distribution"] = {int(k): int(v) for k, v in stats["year_distribution"].items()}
    stats_path = output_path.parent / "rent_contracts_cleaning_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2, default=lambda x: int(x) if hasattr(x, 'item') else str(x))
    print(f"Saved cleaning statistics to {stats_path}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("CLEANING SUMMARY")
    print("=" * 60)
    print(f"Total rows processed:     {stats['total_rows']:,}")
    print(f"Residential rows:         {stats['residential_rows']:,} ({stats['residential_rows']/stats['total_rows']*100:.1f}%)")
    print(f"Commercial rows:          {stats['commercial_rows']:,} ({stats['commercial_rows']/stats['total_rows']*100:.1f}%)")
    print(f"Invalid date rows:        {stats['invalid_date_rows']:,}")
    print()
    print("Bedroom Distribution:")
    for bedroom, count in sorted(stats["bedroom_distribution"].items(), key=lambda x: -x[1]):
        print(f"  {bedroom:12s}: {count:>10,}")
    print()
    
    if stats["unmapped_subtypes"]:
        print(f"WARNING: {len(stats['unmapped_subtypes'])} unmapped property subtypes found:")
        for subtype in stats["unmapped_subtypes"][:10]:
            print(f"  - {subtype}")
        if len(stats["unmapped_subtypes"]) > 10:
            print(f"  ... and {len(stats['unmapped_subtypes']) - 10} more")
    
    return stats


if __name__ == "__main__":
    # Default paths
    project_root = Path(__file__).parent.parent
    input_path = project_root / "Data" / "Rent_Contracts.csv"
    output_path = project_root / "Data" / "cleaned" / "Rent_Contracts_Cleaned.csv"
    
    stats = clean_rent_contracts(
        str(input_path),
        str(output_path),
        save_commercial=True
    )

