#!/usr/bin/env python3
"""
Generate detailed profiling reports for smaller CSV files.
Uses ydata-profiling for comprehensive analysis if available.

Usage:
    python scripts/profile_small_csv.py
    python scripts/profile_small_csv.py --file Data/Projects.csv
"""

import pandas as pd
from pathlib import Path
import argparse
import sys
import json

try:
    from ydata_profiling import ProfileReport
    YDATA_AVAILABLE = True
except ImportError:
    YDATA_AVAILABLE = False
    print("Note: ydata-profiling not installed. Using basic profiling.")
    print("Install with: pip install ydata-profiling")


def basic_profile(df: pd.DataFrame, title: str, output_path: Path):
    """Generate basic profile without ydata-profiling."""
    profile = {
        "title": title,
        "rows": len(df),
        "columns": len(df.columns),
        "memory_mb": round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2),
        "column_profiles": {}
    }
    
    for col in df.columns:
        col_profile = {
            "dtype": str(df[col].dtype),
            "null_count": int(df[col].isna().sum()),
            "null_pct": round(df[col].isna().mean() * 100, 2),
            "unique_count": int(df[col].nunique()),
            "sample_values": [str(v) for v in df[col].dropna().head(5).tolist()],
        }
        
        if pd.api.types.is_numeric_dtype(df[col]):
            non_null = df[col].dropna()
            if len(non_null) > 0:
                col_profile["min"] = float(non_null.min())
                col_profile["max"] = float(non_null.max())
                col_profile["mean"] = float(non_null.mean())
                col_profile["std"] = float(non_null.std())
        
        profile["column_profiles"][col] = col_profile
    
    # Save as JSON
    json_path = output_path.with_suffix('.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(profile, f, indent=2, ensure_ascii=False)
    
    print(f"  Basic profile saved to: {json_path}")
    
    # Also create a simple text report
    txt_path = output_path.with_suffix('.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(f"Data Profile: {title}\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Rows: {profile['rows']:,}\n")
        f.write(f"Columns: {profile['columns']}\n")
        f.write(f"Memory: {profile['memory_mb']:.2f} MB\n\n")
        
        f.write("Column Details:\n")
        f.write("-" * 60 + "\n")
        f.write(f"{'Column':<30} {'Null%':>8} {'Unique':>10} {'Type':<15}\n")
        f.write("-" * 60 + "\n")
        
        for col, stats in profile["column_profiles"].items():
            f.write(f"{col[:30]:<30} {stats['null_pct']:>7.1f}% {stats['unique_count']:>10,} {stats['dtype']:<15}\n")
    
    print(f"  Text report saved to: {txt_path}")
    return profile


def full_profile(df: pd.DataFrame, title: str, output_path: Path):
    """Generate full profile with ydata-profiling."""
    profile = ProfileReport(
        df,
        title=title,
        explorative=True,
        minimal=False,
    )
    
    profile.to_file(output_path)
    print(f"  Full HTML profile saved to: {output_path}")
    return profile


def profile_file(filepath: Path, output_dir: Path):
    """Profile a single file."""
    print(f"\nProfiling: {filepath.name}")
    print("-" * 40)
    
    try:
        df = pd.read_csv(filepath, low_memory=False)
        print(f"  Rows: {len(df):,}")
        print(f"  Columns: {len(df.columns)}")
        print(f"  Memory: {df.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")
        
        output_path = output_dir / f"{filepath.stem}_profile.html"
        
        if YDATA_AVAILABLE:
            full_profile(df, f"Data Profile: {filepath.name}", output_path)
        
        # Always generate basic profile (faster to parse programmatically)
        basic_profile(df, f"Data Profile: {filepath.name}", output_dir / f"{filepath.stem}_profile")
        
        return True
        
    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Profile small/medium CSV files")
    parser.add_argument("--file", help="Path to specific file to profile")
    parser.add_argument("--output", default="data_profiles/detailed_reports", help="Output directory")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.file:
        filepath = Path(args.file)
        if not filepath.exists():
            print(f"Error: {filepath} not found")
            sys.exit(1)
        profile_file(filepath, output_dir)
    else:
        # Profile all small files
        small_files = [
            "Data/Buildings.csv",
            "Data/Projects.csv",
            "Data/Valuation.csv",
        ]
        
        print("="*60)
        print("Profiling Small/Medium CSV Files")
        print("="*60)
        
        success = 0
        for filepath_str in small_files:
            filepath = Path(filepath_str)
            if filepath.exists():
                if profile_file(filepath, output_dir):
                    success += 1
            else:
                print(f"\nSkipping {filepath_str} - not found")
        
        print(f"\n{'='*60}")
        print(f"Completed: {success}/{len(small_files)} files profiled")
        print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()

