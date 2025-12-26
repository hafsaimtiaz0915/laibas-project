#!/usr/bin/env python3
"""
Entity Resolution Pipeline for Dubai Real Estate Data

This script applies entity resolution mappings to standardize:
- Area names (area_name_en)
- Master project names (master_project_en)

This enables proper joins between Transactions, Units, Rent_Contracts, and Projects.
"""

import pandas as pd
import json
from pathlib import Path
import re

# ============================================
# CANONICAL MAPPINGS
# ============================================

# Area name corrections (case sensitivity)
AREA_MAPPINGS = {
    "AL QUSAIS": "Al Qusais",
    "BUSINESS BAY": "Business Bay",
    "BusinessBay": "Business Bay",
}

# Master project to parent project mappings
# Sub-communities mapped to their parent master project
PROJECT_MAPPINGS = {
    # Arabian Ranches sub-communities
    "Arabian Ranches - Golf Homes": "Arabian Ranches - 1",
    "Arabian Ranches - Polo Homes": "Arabian Ranches - 1",
    "Arabian Ranches II - AZALEA": "Arabian Ranches II",
    "Arabian Ranches II - CASA": "Arabian Ranches II",
    "Arabian Ranches II - Camelia": "Arabian Ranches II",
    "Arabian Ranches II - LILA": "Arabian Ranches II",
    "Arabian Ranches II - PALMA": "Arabian Ranches II",
    "Arabian Ranches II - RASHA": "Arabian Ranches II",
    "Arabian Ranches II - ROSA": "Arabian Ranches II",
    "Arabian Ranches II - Reem Community": "Arabian Ranches II",
    "Arabian Ranches II - SAMARA": "Arabian Ranches II",
    "Arabian Ranches II - YASMIN": "Arabian Ranches II",
    
    # Dubai Hills sub-communities
    "DUBAI HILLS": "Dubai Hills Estate",
    "DUBAI HILLS - CLUB VILLAS": "Dubai Hills Estate",
    "DUBAI HILLS - EMERALD HILLS": "Dubai Hills Estate",
    "DUBAI HILLS - FAIRWAY VISTAS": "Dubai Hills Estate",
    "DUBAI HILLS - FAIRWAYS": "Dubai Hills Estate",
    "DUBAI HILLS - GOLF GROVE": "Dubai Hills Estate",
    "DUBAI HILLS - GOLF PLACE": "Dubai Hills Estate",
    "DUBAI HILLS - GOLF TERRACES": "Dubai Hills Estate",
    "DUBAI HILLS - HILLS GROVE": "Dubai Hills Estate",
    "DUBAI HILLS - HILLS VIEW": "Dubai Hills Estate",
    "DUBAI HILLS - LAMBOURGHINI": "Dubai Hills Estate",
    "DUBAI HILLS - MAPLE 1": "Dubai Hills Estate",
    "DUBAI HILLS - MAPLE 2": "Dubai Hills Estate",
    "DUBAI HILLS - MAPLE 3": "Dubai Hills Estate",
    "DUBAI HILLS - PALM HILLS": "Dubai Hills Estate",
    "DUBAI HILLS - PARKWAY VISTAS": "Dubai Hills Estate",
    "DUBAI HILLS - PARKWAYS": "Dubai Hills Estate",
    "DUBAI HILLS - SIDRA 1": "Dubai Hills Estate",
    "DUBAI HILLS - SIDRA 2": "Dubai Hills Estate",
    "DUBAI HILLS - SIDRA 3": "Dubai Hills Estate",
    "DUBAI HILLS - PARK": "Dubai Hills Estate",
    
    # The Lakes sub-communities (NOT Jumeirah Lakes Towers!)
    "Lakes - Deema 1": "The Lakes",
    "Lakes - Deema 2": "The Lakes",
    "Lakes - Deema 3": "The Lakes",
    "Lakes - Deema 4": "The Lakes",
    "Lakes - Forat": "The Lakes",
    "Lakes - Ghadeer": "The Lakes",
    "Lakes - Hattan I": "The Lakes",
    "Lakes - Hattan II": "The Lakes",
    "Lakes - Hattan III": "The Lakes",
    "Lakes - Maeen": "The Lakes",
    
    # Meadows sub-communities
    "Meadows 1": "The Meadows",
    "Meadows 2": "The Meadows",
    "Meadows 3": "The Meadows",
    "Meadows 4": "The Meadows",
    "Meadows 5": "The Meadows",
    "Meadows 6": "The Meadows",
    "Meadows 7": "The Meadows",
    "Meadows 8": "The Meadows",
    "Meadows 9": "The Meadows",
    
    # Springs sub-communities
    "Springs - 1": "The Springs",
    "Springs - 2": "The Springs",
    "Springs - 3": "The Springs",
    "Springs - 4": "The Springs",
    "Springs - 5": "The Springs",
    "Springs - 6": "The Springs",
    "Springs - 7": "The Springs",
    
    # Golf estates
    "Jumeirah Golf Estates - Phase B": "Jumeirah Golf Estates",
    
    # Liwan
    "Liwan": "Liwan1",
}


def resolve_area_name(name):
    """Resolve area name to canonical form."""
    if pd.isna(name):
        return name
    
    # Check exact mapping first
    if name in AREA_MAPPINGS:
        return AREA_MAPPINGS[name]
    
    # Normalize: Title Case
    return str(name).strip()


def resolve_project_name(name):
    """Resolve project name to parent project."""
    if pd.isna(name):
        return name
    
    # Check exact mapping first
    if name in PROJECT_MAPPINGS:
        return PROJECT_MAPPINGS[name]
    
    return str(name).strip()


def apply_entity_resolution(df, columns=None):
    """
    Apply entity resolution to a DataFrame.
    
    Args:
        df: pandas DataFrame
        columns: list of columns to resolve, or None for auto-detect
    
    Returns:
        DataFrame with resolved entities
    """
    df = df.copy()
    
    if columns is None:
        columns = df.columns.tolist()
    
    # Resolve area names
    if 'area_name_en' in columns and 'area_name_en' in df.columns:
        df['area_name_en'] = df['area_name_en'].apply(resolve_area_name)
    
    # Resolve project names
    if 'master_project_en' in columns and 'master_project_en' in df.columns:
        df['master_project_en'] = df['master_project_en'].apply(resolve_project_name)
    
    return df


def process_file_in_chunks(filepath, output_path, chunksize=100000):
    """Process large CSV file in chunks with entity resolution."""
    
    print(f"Processing: {filepath}")
    
    first_chunk = True
    total_rows = 0
    
    for chunk in pd.read_csv(filepath, chunksize=chunksize, low_memory=False):
        # Apply entity resolution
        chunk = apply_entity_resolution(chunk)
        
        # Write to output
        chunk.to_csv(
            output_path, 
            mode='w' if first_chunk else 'a',
            header=first_chunk,
            index=False
        )
        
        first_chunk = False
        total_rows += len(chunk)
        print(f"  Processed {total_rows:,} rows...")
    
    print(f"✓ Saved to: {output_path}")
    return total_rows


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Apply entity resolution to CSV files')
    parser.add_argument('--file', type=str, help='Single file to process')
    parser.add_argument('--all', action='store_true', help='Process all data files')
    parser.add_argument('--output-dir', type=str, default='Data/cleaned', help='Output directory')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.file:
        input_path = Path(args.file)
        output_path = output_dir / f"{input_path.stem}_resolved{input_path.suffix}"
        process_file_in_chunks(input_path, output_path)
    
    elif args.all:
        files = [
            'Data/Transactions.csv',
            'Data/Units.csv', 
            'Data/Rent_Contracts.csv',
            'Data/Buildings.csv',
            'Data/Projects.csv',
        ]
        
        for f in files:
            input_path = Path(f)
            if input_path.exists():
                output_path = output_dir / f"{input_path.stem}_resolved.csv"
                process_file_in_chunks(input_path, output_path)
    
    else:
        print("Usage: python apply_entity_resolution.py --file <path> | --all")
        print("\nAvailable mappings:")
        print(f"  Area name mappings: {len(AREA_MAPPINGS)}")
        print(f"  Project mappings: {len(PROJECT_MAPPINGS)}")

