"""
Tourism Data Cleaning Script

Cleans and consolidates the 54 Excel files from Data/Toursim Data/ into 
structured time series CSVs for ML model training.

Outputs:
- Data/cleaned/tourism_visitors.csv: Visitor counts by region over time
- Data/cleaned/tourism_inventory.csv: Hotel/apartment inventory over time
"""

import pandas as pd
import numpy as np
from pathlib import Path
import re
import json
from datetime import datetime
from typing import Tuple, Optional, Dict, List


# Arabic quarter mapping
ARABIC_QUARTERS = {
    'الأول': 'Q1',
    'الثاني': 'Q2', 
    'الثالث': 'Q3',
    'الرابع': 'Q4',
}

# File type detection patterns
FILE_PATTERNS = {
    'visitors': [
        'الزوار حسب المنطقة الجغرافية',  # Visitors by region
    ],
    'visitors_distribution': [
        'التوزيع النسبي للزوار',  # Percentage distribution of visitors
    ],
    'hotel_inventory': [
        'الفنادق و الشقق الفندقية',  # Hotels and apartments
        'الفنادق و مبانى الشقق الفندقية',  # Hotels and apartment buildings
        'الغرف الفندقية والشقق الفندقية',  # Hotel rooms and apartments
    ],
    'hotel_occupancy': [
        'الفنادق ومتوسط إشغال الغرف',  # Hotels and average room occupancy
        'الشقق الفندقية ومتوسط إشغال',  # Hotel apartments and average occupancy
    ],
}

# Region name mapping (Arabic -> English)
REGION_MAPPING = {
    'أوروبا الغربية': 'Western Europe',
    'جنوب آسيا': 'South Asia',
    'رابطة الدول المستقلة وأوروبا الشرقية': 'CIS & Eastern Europe',
    'رابطة الدول المستقلة وأوروبا الشرقية**': 'CIS & Eastern Europe',
    'دول مجلس التعاون الخليجي': 'GCC',
    'منطقة الشرق الأوسط وشمال أفريقيا': 'MENA',
    'الأمريكتان': 'Americas',
    'أفريقيا': 'Africa',
    'شرق آسيا': 'East Asia',
    'جنوب شرق آسيا': 'Southeast Asia',
    'أوقيانوسيا': 'Oceania',
    'الصين': 'China',
    'الهند': 'India',
    'روسيا': 'Russia',
    'المملكة المتحدة': 'United Kingdom',
    'ألمانيا': 'Germany',
    'الإمارات العربية المتحدة': 'UAE',
    'المملكة العربية السعودية': 'Saudi Arabia',
    'الإجمالي': 'Total',
    'مناطق أخرى': 'Other Regions',
}


def extract_year_quarter(filename: str) -> Tuple[Optional[int], Optional[str]]:
    """Extract year and quarter from Arabic filename."""
    # Extract year (4 digits)
    year_match = re.search(r'20\d{2}', filename)
    year = int(year_match.group()) if year_match else None
    
    # Extract quarter
    quarter = None
    for arabic_q, english_q in ARABIC_QUARTERS.items():
        if arabic_q in filename:
            quarter = english_q
            break
    
    return year, quarter


def detect_file_type(filename: str) -> Optional[str]:
    """Detect the type of tourism data file."""
    for file_type, patterns in FILE_PATTERNS.items():
        for pattern in patterns:
            if pattern in filename:
                return file_type
    return None


def parse_visitors_file(filepath: Path, year: int, quarter: Optional[str]) -> Optional[pd.DataFrame]:
    """Parse a visitors Excel file into structured data."""
    try:
        df = pd.read_excel(filepath)
        
        # The data structure typically has:
        # - Header rows with title info
        # - A row with 'البيان' / 'Title' indicating column headers
        # - Data rows with regions and values
        
        # Find the title row
        title_row = None
        for idx, row in df.iterrows():
            row_str = ' '.join(str(v) for v in row.values if pd.notna(v))
            if 'البيان' in row_str or 'Title' in row_str:
                title_row = idx
                break
        
        if title_row is None:
            return None
        
        # Get data rows (after title row)
        data_start = title_row + 1
        data = []
        
        for idx in range(data_start, len(df)):
            row = df.iloc[idx]
            
            # Skip empty rows or source/footer rows
            row_str = ' '.join(str(v) for v in row.values if pd.notna(v))
            if not row_str or 'المصدر' in row_str or 'Source' in row_str:
                continue
            
            # Get region name (first or last column usually has English)
            region_ar = str(row.iloc[0]) if pd.notna(row.iloc[0]) else None
            region_en = str(row.iloc[-1]) if pd.notna(row.iloc[-1]) else None
            
            # Map Arabic to English if needed
            if region_ar in REGION_MAPPING:
                region = REGION_MAPPING[region_ar]
            elif region_en and region_en != 'nan':
                region = region_en.strip()
            elif region_ar:
                region = region_ar.strip()
            else:
                continue
            
            # Get visitor count (usually in middle columns)
            value = None
            for col_idx in range(1, len(row) - 1):
                val = row.iloc[col_idx]
                if pd.notna(val) and isinstance(val, (int, float)):
                    value = val
                    break
                elif pd.notna(val):
                    try:
                        # Try to parse as number (might have commas)
                        val_str = str(val).replace(',', '').replace(' ', '')
                        value = float(val_str)
                        break
                    except ValueError:
                        continue
            
            if value is not None:
                data.append({
                    'year': year,
                    'quarter': quarter,
                    'period': f"{year}-{quarter}" if quarter else str(year),
                    'region': region,
                    'visitors_thousands': value,
                })
        
        if data:
            return pd.DataFrame(data)
        return None
        
    except Exception as e:
        print(f"  Error parsing {filepath.name}: {e}")
        return None


def parse_inventory_file(filepath: Path, year: int, quarter: Optional[str]) -> Optional[pd.DataFrame]:
    """Parse a hotel inventory Excel file into structured data."""
    try:
        df = pd.read_excel(filepath)
        
        # Find the title row
        title_row = None
        for idx, row in df.iterrows():
            row_str = ' '.join(str(v) for v in row.values if pd.notna(v))
            if 'البيان' in row_str or 'Title' in row_str:
                title_row = idx
                break
        
        if title_row is None:
            return None
        
        # Get data rows (after title row)
        data_start = title_row + 1
        metrics = {}
        
        # Metric name mapping
        METRIC_MAPPING = {
            'عدد الفنادق': 'num_hotels',
            'Number of Hotels': 'num_hotels',
            'عدد الغرف الفندقية': 'num_hotel_rooms',
            'Number of Hotel rooms': 'num_hotel_rooms',
            'عدد مباني الشقق الفندقية': 'num_apartment_buildings',
            'Number of Hotel Apartment Bldgs': 'num_apartment_buildings',
            'عدد الشقق الفندقية': 'num_hotel_apartments',
            'Number of Hotel Apartments': 'num_hotel_apartments',
        }
        
        for idx in range(data_start, len(df)):
            row = df.iloc[idx]
            
            # Skip empty rows or source rows
            row_str = ' '.join(str(v) for v in row.values if pd.notna(v))
            if not row_str or 'المصدر' in row_str or 'Source' in row_str:
                continue
            
            # Get metric name from first or last column
            metric_ar = str(row.iloc[0]).strip() if pd.notna(row.iloc[0]) else None
            metric_en = str(row.iloc[-1]).strip() if pd.notna(row.iloc[-1]) else None
            
            # Find the metric key
            metric_key = None
            for name, key in METRIC_MAPPING.items():
                if metric_ar and name in metric_ar:
                    metric_key = key
                    break
                if metric_en and name in metric_en:
                    metric_key = key
                    break
            
            if not metric_key:
                continue
            
            # Get values from middle columns (might be multiple years)
            for col_idx in range(1, len(row) - 1):
                val = row.iloc[col_idx]
                if pd.notna(val):
                    try:
                        if isinstance(val, (int, float)):
                            metrics[metric_key] = int(val)
                            break
                        else:
                            val_str = str(val).replace(',', '').replace(' ', '')
                            metrics[metric_key] = int(float(val_str))
                            break
                    except ValueError:
                        continue
        
        if metrics:
            record = {
                'year': year,
                'quarter': quarter,
                'period': f"{year}-{quarter}" if quarter else str(year),
                **metrics
            }
            return pd.DataFrame([record])
        return None
        
    except Exception as e:
        print(f"  Error parsing {filepath.name}: {e}")
        return None


def clean_tourism_data(input_dir: Path, output_dir: Path) -> Dict:
    """Clean all tourism data files and output consolidated CSVs."""
    print("\n" + "=" * 60)
    print("CLEANING TOURISM DATA")
    print("=" * 60)
    
    stats = {
        "total_files": 0,
        "processed_files": 0,
        "failed_files": 0,
        "visitors_records": 0,
        "inventory_records": 0,
    }
    
    visitors_data = []
    inventory_data = []
    
    # Get all Excel files
    excel_files = list(input_dir.glob("*.xlsx"))
    stats["total_files"] = len(excel_files)
    print(f"Found {len(excel_files)} Excel files")
    
    for filepath in excel_files:
        filename = filepath.name
        print(f"\nProcessing: {filename[:50]}...")
        
        # Extract year and quarter
        year, quarter = extract_year_quarter(filename)
        if year is None:
            print(f"  Skipping - couldn't extract year")
            stats["failed_files"] += 1
            continue
        
        # Detect file type
        file_type = detect_file_type(filename)
        
        if file_type == 'visitors':
            result = parse_visitors_file(filepath, year, quarter)
            if result is not None:
                visitors_data.append(result)
                stats["processed_files"] += 1
                print(f"  ✓ Parsed {len(result)} visitor records")
            else:
                stats["failed_files"] += 1
                print(f"  ✗ Failed to parse")
                
        elif file_type in ['hotel_inventory']:
            result = parse_inventory_file(filepath, year, quarter)
            if result is not None:
                inventory_data.append(result)
                stats["processed_files"] += 1
                print(f"  ✓ Parsed inventory record")
            else:
                stats["failed_files"] += 1
                print(f"  ✗ Failed to parse")
                
        elif file_type in ['visitors_distribution', 'hotel_occupancy']:
            # These have different structures - skip for now
            print(f"  Skipping {file_type} (not implemented)")
            stats["failed_files"] += 1
        else:
            print(f"  Skipping - unknown file type")
            stats["failed_files"] += 1
    
    # Consolidate and save visitors data
    if visitors_data:
        visitors_df = pd.concat(visitors_data, ignore_index=True)
        
        # Remove duplicates (same period + region)
        visitors_df = visitors_df.drop_duplicates(subset=['period', 'region'])
        
        # Sort by period and region
        visitors_df = visitors_df.sort_values(['year', 'quarter', 'region'])
        
        output_path = output_dir / "tourism_visitors.csv"
        visitors_df.to_csv(output_path, index=False)
        stats["visitors_records"] = len(visitors_df)
        print(f"\n✓ Saved {len(visitors_df)} visitor records to {output_path}")
    else:
        print("\n⚠ No visitor data extracted")
    
    # Consolidate and save inventory data
    if inventory_data:
        inventory_df = pd.concat(inventory_data, ignore_index=True)
        
        # Remove duplicates
        inventory_df = inventory_df.drop_duplicates(subset=['period'])
        
        # Sort by period
        inventory_df = inventory_df.sort_values(['year', 'quarter'])
        
        output_path = output_dir / "tourism_inventory.csv"
        inventory_df.to_csv(output_path, index=False)
        stats["inventory_records"] = len(inventory_df)
        print(f"✓ Saved {len(inventory_df)} inventory records to {output_path}")
    else:
        print("⚠ No inventory data extracted")
    
    # Summary
    print("\n" + "-" * 40)
    print("TOURISM DATA CLEANING SUMMARY")
    print("-" * 40)
    print(f"Total files: {stats['total_files']}")
    print(f"Processed: {stats['processed_files']}")
    print(f"Failed/Skipped: {stats['failed_files']}")
    print(f"Visitor records: {stats['visitors_records']}")
    print(f"Inventory records: {stats['inventory_records']}")
    
    return stats


def main():
    """Run tourism data cleaning."""
    project_root = Path(__file__).parent.parent
    input_dir = project_root / "Data" / "Toursim Data"
    output_dir = project_root / "Data" / "cleaned"
    
    print("=" * 60)
    print("TOURISM DATA CLEANING")
    print(f"Started at: {datetime.now().isoformat()}")
    print("=" * 60)
    
    if not input_dir.exists():
        print(f"ERROR: Input directory not found: {input_dir}")
        return
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    stats = clean_tourism_data(input_dir, output_dir)
    
    # Save stats
    stats_path = output_dir / "tourism_cleaning_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    
    print("\n" + "=" * 60)
    print("TOURISM CLEANING COMPLETE")
    print(f"Finished at: {datetime.now().isoformat()}")
    print("=" * 60)


if __name__ == "__main__":
    main()

