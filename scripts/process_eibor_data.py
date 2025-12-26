"""
EIBOR Data Processing Script

Processes raw EIBOR Excel files into cleaned time series data.
Follows "Clean Only, Never Guide" philosophy - NO buckets, NO thresholds.

Output:
- Data/cleaned/eibor_daily.csv: Daily rates with derived features
- Data/cleaned/eibor_monthly.csv: Monthly averages for transaction joining

Usage:
    python scripts/process_eibor_data.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


# Column name mappings for different EIBOR file formats
# Note: Some column names have trailing spaces in the Excel files
# Note: 2018-2019 files use "ON", "3 Month" (singular) format
COLUMN_MAPPINGS = {
    # English variations
    'date': 'date',
    'Date': 'date',
    'DATE': 'date',
    
    # Overnight (including "ON" format from 2018-2019)
    'overnight': 'overnight',
    'Overnight': 'overnight',
    'O/N': 'overnight',
    'ON': 'overnight',  # Used in 2018-2019 files
    'OVERNIGHT': 'overnight',
    
    # 1 Week
    '1 week': '1_week',
    '1 Week': '1_week',
    '1W': '1_week',
    '1 WEEK': '1_week',
    
    # 1 Month (with and without trailing spaces)
    '1 month': '1_month',
    '1 Month': '1_month',
    '1 Month ': '1_month',  # Trailing space in some files
    '1M': '1_month',
    '1 MONTH': '1_month',
    
    # 3 Months (most important for mortgages) - including singular form
    '3 months': '3_month',
    '3 Months': '3_month',
    '3 Month': '3_month',  # Singular - used in 2018-2019 files
    '3M': '3_month',
    '3 MONTHS': '3_month',
    
    # 6 Months (with and without trailing spaces) - including singular form
    '6 months': '6_month',
    '6 Months': '6_month',
    '6 Month': '6_month',  # Singular - used in 2018-2019 files
    '6 Months ': '6_month',  # Trailing space in some files
    '6M': '6_month',
    '6 MONTHS': '6_month',
    
    # 12 Months / 1 Year
    '12 months': '12_month',
    '12 Months': '12_month',
    '1Y': '12_month',
    '1 Year': '12_month',
    '12 MONTHS': '12_month',
    
    # Arabic mappings
    'التاريخ': 'date',
    'ليلة واحدة': 'overnight',
    'أسبوع واحد': '1_week',
    'شهر واحد': '1_month',
    '3 أشهر': '3_month',
    'ثلاثة أشهر': '3_month',
    '6 أشهر': '6_month',
    'ستة أشهر': '6_month',
    '12 شهر': '12_month',
    'سنة': '12_month',
}


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names to English."""
    rename_map = {}
    
    for col in df.columns:
        col_str = str(col)
        col_clean = col_str.strip()
        
        # First try exact match (with original spacing)
        if col_str in COLUMN_MAPPINGS:
            rename_map[col] = COLUMN_MAPPINGS[col_str]
        # Then try stripped version
        elif col_clean in COLUMN_MAPPINGS:
            rename_map[col] = COLUMN_MAPPINGS[col_clean]
        else:
            # Try partial matching for variations
            col_lower = col_clean.lower()
            for key, value in COLUMN_MAPPINGS.items():
                if key.lower() == col_lower or key.lower().strip() == col_lower:
                    rename_map[col] = value
                    break
    
    return df.rename(columns=rename_map)


def parse_single_sheet(filepath: Path, sheet_name, engine: str, header_row: int = 0) -> pd.DataFrame:
    """Parse a single sheet from an Excel file."""
    try:
        df = pd.read_excel(filepath, sheet_name=sheet_name, header=header_row, engine=engine)
        df = standardize_columns(df)
        
        if 'date' not in df.columns or '3_month' not in df.columns:
            return pd.DataFrame()
        
        # Handle date column
        if not pd.api.types.is_datetime64_any_dtype(df['date']):
            df['date'] = pd.to_datetime(df['date'], errors='coerce', dayfirst=True)
        
        # Drop invalid dates
        df = df.dropna(subset=['date'])
        df = df[(df['date'].dt.year >= 2000) & (df['date'].dt.year <= 2030)]
        
        # Ensure numeric columns
        rate_columns = ['overnight', '1_week', '1_month', '3_month', '6_month', '12_month']
        for col in rate_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Keep only relevant columns
        keep_cols = ['date'] + [c for c in rate_columns if c in df.columns]
        return df[keep_cols].copy()
        
    except Exception:
        return pd.DataFrame()


def parse_eibor_file(filepath: Path) -> pd.DataFrame:
    """
    Parse a single EIBOR Excel file with ALL sheets.
    
    Handles:
    - .xls (old format) and .xlsx (new format)
    - MULTIPLE SHEETS per file (one per month)
    - Different header positions
    - Arabic and English column names
    - Date format variations (string DD-MM-YYYY or datetime objects)
    """
    print(f"  Processing: {filepath.name}")
    
    # Determine file format
    if filepath.suffix == '.xls':
        engine = 'xlrd'
    else:
        engine = 'openpyxl'
    
    all_dfs = []
    
    try:
        xl = pd.ExcelFile(filepath, engine=engine)
        sheet_names = xl.sheet_names
        print(f"    Found {len(sheet_names)} sheets")
        
        # Process each sheet
        for sheet_name in sheet_names:
            # Try different header positions
            for header_row in [0, 1, 2, 3]:
                df = parse_single_sheet(filepath, sheet_name, engine, header_row)
                if len(df) > 0:
                    all_dfs.append(df)
                    break
        
    except Exception as e:
        print(f"    WARNING: Could not open {filepath.name}: {e}")
        return pd.DataFrame()
    
    if not all_dfs:
        print(f"    WARNING: No valid data found in {filepath.name}")
        return pd.DataFrame()
    
    # Combine all sheets
    combined = pd.concat(all_dfs, ignore_index=True)
    combined = combined.drop_duplicates(subset=['date'], keep='last')
    combined = combined.sort_values('date').reset_index(drop=True)
    
    if len(combined) > 0:
        print(f"    Parsed {len(combined)} rows total, date range: {combined['date'].min().date()} to {combined['date'].max().date()}")
    
    return combined


def consolidate_eibor_files(eibor_dir: str) -> pd.DataFrame:
    """
    Consolidate all EIBOR files into a single time series.
    """
    eibor_path = Path(eibor_dir)
    all_dfs = []
    
    print("=" * 60)
    print("EIBOR Data Processing")
    print("=" * 60)
    
    # Process each file
    for filepath in sorted(eibor_path.glob('*.xls*')):
        df = parse_eibor_file(filepath)
        if len(df) > 0:
            all_dfs.append(df)
    
    if not all_dfs:
        raise ValueError("No EIBOR files could be parsed!")
    
    # Concatenate all
    combined = pd.concat(all_dfs, ignore_index=True)
    
    # Remove duplicates (same date from overlapping files)
    combined = combined.drop_duplicates(subset=['date'], keep='last')
    
    # Sort by date
    combined = combined.sort_values('date').reset_index(drop=True)
    
    # Fill missing rate columns with forward fill (weekends/holidays)
    # This is DATA CLEANING (filling gaps), not MODEL GUIDANCE
    rate_columns = ['overnight', '1_week', '1_month', '3_month', '6_month', '12_month']
    for col in rate_columns:
        if col in combined.columns:
            combined[col] = combined[col].ffill()
    
    return combined


def compute_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute derived EIBOR features for model training.
    
    ⚠️ CRITICAL: Only compute FACTUAL derived values.
    NO buckets, NO thresholds, NO categorical judgments.
    The model learns all relationships from continuous values.
    """
    df = df.copy()
    
    # Primary rate (3-month EIBOR is most common mortgage reference)
    df['eibor_primary'] = df['3_month']
    
    # Rate changes over different horizons (FACTUAL - mathematical differences)
    df['eibor_change_1m'] = df['3_month'].diff(periods=21)   # ~21 trading days
    df['eibor_change_3m'] = df['3_month'].diff(periods=63)   # ~63 trading days
    df['eibor_change_6m'] = df['3_month'].diff(periods=126)  # ~126 trading days
    df['eibor_change_12m'] = df['3_month'].diff(periods=252) # ~252 trading days
    
    # Rate momentum (FACTUAL - percentage change calculation)
    df['eibor_momentum_3m'] = df['3_month'].pct_change(periods=63) * 100
    df['eibor_momentum_6m'] = df['3_month'].pct_change(periods=126) * 100
    df['eibor_momentum_12m'] = df['3_month'].pct_change(periods=252) * 100
    
    # Yield curve slope (FACTUAL - mathematical difference)
    if '12_month' in df.columns:
        df['yield_curve_slope'] = df['12_month'] - df['3_month']
    
    # Volatility (FACTUAL - statistical measure)
    df['eibor_volatility_3m'] = df['3_month'].rolling(window=63).std()
    df['eibor_volatility_12m'] = df['3_month'].rolling(window=252).std()
    
    # ════════════════════════════════════════════════════════════════════
    # ❌ REMOVED: The following were HUMAN JUDGMENTS, not facts:
    # 
    # df['eibor_level'] = pd.cut(...)    # Who decides what's "Low" vs "High"?
    # df['eibor_direction'] = pd.cut(...) # Who decides 0.25 is the threshold?
    #
    # The MODEL will learn what EIBOR levels correlate with outcomes.
    # We don't tell it - it discovers from the continuous values.
    # ════════════════════════════════════════════════════════════════════
    
    return df


def create_monthly_summary(daily_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create monthly summary for joining with transaction data.
    
    Most transactions only have month-level dates, so we need
    monthly averages.
    """
    df = daily_df.copy()
    df['year_month'] = df['date'].dt.to_period('M')
    
    # Aggregate to monthly - all continuous values
    agg_dict = {
        'overnight': 'mean',
        '1_week': 'mean',
        '1_month': 'mean',
        '3_month': 'mean',
        '6_month': 'mean',
        '12_month': 'mean',
        'eibor_primary': 'mean',
    }
    
    # Add derived features if they exist
    for col in ['eibor_change_3m', 'eibor_change_6m', 'eibor_change_12m', 
                'eibor_momentum_3m', 'yield_curve_slope', 
                'eibor_volatility_3m', 'eibor_volatility_12m']:
        if col in df.columns:
            agg_dict[col] = 'last'  # Take end-of-month value
    
    monthly = df.groupby('year_month').agg(agg_dict).reset_index()
    
    # Convert period to date (first of month)
    monthly['year_month'] = monthly['year_month'].dt.to_timestamp()
    
    return monthly


def main():
    """Main processing pipeline."""
    project_root = Path(__file__).parent.parent
    EIBOR_DIR = project_root / "Data" / "EIBOR_RAW_DATA"
    OUTPUT_DIR = project_root / "Data" / "cleaned"
    
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Step 1: Consolidate all files
    print("\nStep 1: Consolidating EIBOR files...")
    daily_df = consolidate_eibor_files(str(EIBOR_DIR))
    
    # Step 2: Compute derived features
    print("\nStep 2: Computing derived features (continuous values only)...")
    daily_df = compute_derived_features(daily_df)
    
    # Step 3: Create monthly summary
    print("\nStep 3: Creating monthly summary...")
    monthly_df = create_monthly_summary(daily_df)
    
    # Step 4: Save outputs
    print("\nStep 4: Saving outputs...")
    
    # Daily data (for precise lookups)
    daily_output = OUTPUT_DIR / "eibor_daily.csv"
    daily_df.to_csv(daily_output, index=False)
    print(f"  Daily data saved: {daily_output}")
    
    # Monthly data (for joining with transactions)
    monthly_output = OUTPUT_DIR / "eibor_monthly.csv"
    monthly_df.to_csv(monthly_output, index=False)
    print(f"  Monthly data saved: {monthly_output}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("EIBOR PROCESSING SUMMARY")
    print("=" * 60)
    print(f"Date range: {daily_df['date'].min().date()} to {daily_df['date'].max().date()}")
    print(f"Total daily observations: {len(daily_df):,}")
    print(f"Total monthly observations: {len(monthly_df):,}")
    print(f"\n3-Month EIBOR statistics (continuous values):")
    print(f"  Min: {daily_df['3_month'].min():.2f}%")
    print(f"  Max: {daily_df['3_month'].max():.2f}%")
    print(f"  Current: {daily_df['3_month'].iloc[-1]:.2f}%")
    print(f"  Volatility (3m): {daily_df['eibor_volatility_3m'].iloc[-1]:.4f}")
    
    print("\n⚠️ NOTE: All features are CONTINUOUS. NO categorical buckets.")
    print("   The model will learn what EIBOR levels matter from outcomes.")
    
    return daily_df, monthly_df


if __name__ == "__main__":
    daily_df, monthly_df = main()

