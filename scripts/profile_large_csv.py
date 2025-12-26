#!/usr/bin/env python3
"""
Profile large CSV files without loading entirely into memory.
Outputs: data_profiles/{filename}_profile.json

Usage:
    python scripts/profile_large_csv.py Data/Transactions.csv
    python scripts/profile_large_csv.py --all
"""

import pandas as pd
import json
import argparse
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import re
from datetime import datetime
import sys


class LargeCSVProfiler:
    """Profile large CSV files in chunks."""
    
    def __init__(self, filepath: str, chunk_size: int = 100_000):
        self.filepath = Path(filepath)
        self.chunk_size = chunk_size
        self.profile = {
            "filename": self.filepath.name,
            "filepath": str(self.filepath),
            "profiled_at": datetime.now().isoformat(),
            "total_rows": 0,
            "columns": {},
            "sample_values": {},
            "entity_resolution_candidates": {},
        }
        
    def profile_file(self):
        """Run full profiling pipeline."""
        print(f"\n{'='*60}")
        print(f"Profiling: {self.filepath.name}")
        print(f"{'='*60}")
        
        # First pass: Get columns and basic stats
        print("Reading schema...")
        first_chunk = pd.read_csv(self.filepath, nrows=5)
        columns = list(first_chunk.columns)
        print(f"Found {len(columns)} columns")
        
        # Initialize column stats
        for col in columns:
            self.profile["columns"][col] = {
                "dtype": str(first_chunk[col].dtype),
                "null_count": 0,
                "total_count": 0,
                "unique_values": set(),
                "sample_values": [],
                "min": None,
                "max": None,
            }
        
        # Count total rows first (for progress bar)
        print("Counting rows...")
        with open(self.filepath, 'r', encoding='utf-8', errors='ignore') as f:
            total_lines = sum(1 for _ in f) - 1  # Subtract header
        print(f"Total rows: {total_lines:,}")
        
        # Process in chunks
        print("Processing chunks...")
        chunk_iter = pd.read_csv(
            self.filepath, 
            chunksize=self.chunk_size,
            low_memory=False,
            on_bad_lines='warn'
        )
        
        num_chunks = (total_lines // self.chunk_size) + 1
        for chunk_num, chunk in enumerate(tqdm(chunk_iter, total=num_chunks, desc="Chunks")):
            self.profile["total_rows"] += len(chunk)
            self._process_chunk(chunk, chunk_num)
        
        # Finalize profile
        self._finalize_profile()
        return self.profile
    
    def _process_chunk(self, chunk: pd.DataFrame, chunk_num: int):
        """Process a single chunk."""
        for col in chunk.columns:
            if col not in self.profile["columns"]:
                continue
                
            col_stats = self.profile["columns"][col]
            
            # Count nulls
            col_stats["null_count"] += int(chunk[col].isna().sum())
            col_stats["total_count"] += len(chunk)
            
            # Track unique values (cap at 10000 to avoid memory issues)
            if len(col_stats["unique_values"]) < 10000:
                non_null = chunk[col].dropna()
                if len(non_null) > 0:
                    new_uniques = set(non_null.astype(str).unique())
                    col_stats["unique_values"].update(new_uniques)
            
            # Sample values from first chunk only
            if chunk_num == 0:
                non_null = chunk[col].dropna()
                if len(non_null) > 0:
                    col_stats["sample_values"] = [str(v) for v in non_null.head(5).tolist()]
            
            # Min/Max for numeric columns
            if pd.api.types.is_numeric_dtype(chunk[col]):
                non_null = chunk[col].dropna()
                if len(non_null) > 0:
                    chunk_min = non_null.min()
                    chunk_max = non_null.max()
                    if col_stats["min"] is None or chunk_min < col_stats["min"]:
                        col_stats["min"] = float(chunk_min)
                    if col_stats["max"] is None or chunk_max > col_stats["max"]:
                        col_stats["max"] = float(chunk_max)
    
    def _finalize_profile(self):
        """Calculate final statistics."""
        print("Finalizing profile...")
        
        for col, stats in self.profile["columns"].items():
            # Convert set to count
            stats["unique_count"] = len(stats["unique_values"])
            
            # Calculate null percentage
            if stats["total_count"] > 0:
                stats["null_pct"] = round(
                    stats["null_count"] / stats["total_count"] * 100, 2
                )
            else:
                stats["null_pct"] = 0
            
            # For text columns with few unique values, store for entity resolution
            if stats["unique_count"] < 1000 and stats["unique_count"] > 0:
                self.profile["entity_resolution_candidates"][col] = sorted(
                    list(stats["unique_values"])
                )[:500]  # Cap for JSON size
            
            # Clean up for JSON serialization
            stats["unique_values"] = None  # Remove the set
    
    def save_profile(self, output_dir: str = "data_profiles"):
        """Save profile to JSON."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        output_file = output_path / f"{self.filepath.stem}_profile.json"
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(self.profile, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"\nProfile saved to: {output_file}")
        return output_file
    
    def print_summary(self):
        """Print a summary of the profile."""
        print(f"\n{'='*60}")
        print("PROFILE SUMMARY")
        print(f"{'='*60}")
        print(f"File: {self.profile['filename']}")
        print(f"Total Rows: {self.profile['total_rows']:,}")
        print(f"Total Columns: {len(self.profile['columns'])}")
        
        print(f"\n{'Column Statistics':^60}")
        print("-"*60)
        print(f"{'Column':<30} {'Null%':>8} {'Unique':>10} {'Type':<10}")
        print("-"*60)
        
        for col, stats in sorted(self.profile["columns"].items()):
            null_pct = stats.get("null_pct", 0)
            unique = stats.get("unique_count", 0)
            dtype = stats.get("dtype", "unknown")[:10]
            
            # Highlight high null rates
            null_str = f"{null_pct:>7.1f}%"
            if null_pct > 50:
                null_str = f"⚠️{null_pct:>5.1f}%"
            elif null_pct > 20:
                null_str = f"!{null_pct:>6.1f}%"
            
            print(f"{col[:30]:<30} {null_str:>8} {unique:>10,} {dtype:<10}")


class TransactionsProfiler(LargeCSVProfiler):
    """Extended profiler for Transactions.csv with domain-specific checks."""
    
    def __init__(self, filepath: str):
        super().__init__(filepath)
        self.profile["domain_analysis"] = {
            "price_distribution": {},
            "date_parsing": {"success": 0, "failure": 0, "sample_failures": []},
            "procedure_types": {},
            "registration_types": {},
            "area_name_count": 0,
        }
    
    def _process_chunk(self, chunk: pd.DataFrame, chunk_num: int):
        super()._process_chunk(chunk, chunk_num)
        
        # Domain-specific analysis
        
        # 1. Price distribution
        if "actual_worth" in chunk.columns:
            prices = pd.to_numeric(chunk["actual_worth"], errors="coerce")
            valid_prices = prices.dropna()
            if len(valid_prices) > 0:
                self._update_price_stats(valid_prices)
        
        # 2. Date parsing
        if "instance_date" in chunk.columns:
            self._analyze_dates(chunk["instance_date"], chunk_num)
        
        # 3. Procedure types (Sales, Mortgages, Gifts, etc.)
        if "procedure_name_en" in chunk.columns:
            for proc, count in chunk["procedure_name_en"].value_counts().items():
                self.profile["domain_analysis"]["procedure_types"][str(proc)] = \
                    self.profile["domain_analysis"]["procedure_types"].get(str(proc), 0) + int(count)
        
        # 4. Registration types (Off-Plan vs Ready)
        if "reg_type_en" in chunk.columns:
            for reg_type, count in chunk["reg_type_en"].value_counts().items():
                self.profile["domain_analysis"]["registration_types"][str(reg_type)] = \
                    self.profile["domain_analysis"]["registration_types"].get(str(reg_type), 0) + int(count)
    
    def _analyze_dates(self, date_series: pd.Series, chunk_num: int):
        """Analyze date parsing success."""
        sample_size = 1000 if chunk_num == 0 else 100  # More samples from first chunk
        
        for date_str in date_series.dropna().head(sample_size):
            try:
                # Try DD-MM-YYYY format (Dubai standard)
                pd.to_datetime(str(date_str), format="%d-%m-%Y")
                self.profile["domain_analysis"]["date_parsing"]["success"] += 1
            except:
                try:
                    # Try other common formats
                    pd.to_datetime(str(date_str), dayfirst=True)
                    self.profile["domain_analysis"]["date_parsing"]["success"] += 1
                except:
                    self.profile["domain_analysis"]["date_parsing"]["failure"] += 1
                    # Keep sample of failures for debugging
                    failures = self.profile["domain_analysis"]["date_parsing"]["sample_failures"]
                    if len(failures) < 20:
                        failures.append(str(date_str))
    
    def _update_price_stats(self, prices: pd.Series):
        """Track price distribution."""
        stats = self.profile["domain_analysis"]["price_distribution"]
        
        buckets = [
            (0, 100_000, "0-100K"),
            (100_000, 500_000, "100K-500K"),
            (500_000, 1_000_000, "500K-1M"),
            (1_000_000, 2_000_000, "1M-2M"),
            (2_000_000, 5_000_000, "2M-5M"),
            (5_000_000, 10_000_000, "5M-10M"),
            (10_000_000, 50_000_000, "10M-50M"),
            (50_000_000, float('inf'), "50M+"),
        ]
        
        for low, high, label in buckets:
            count = int(((prices >= low) & (prices < high)).sum())
            stats[label] = stats.get(label, 0) + count


class RentContractsProfiler(LargeCSVProfiler):
    """Extended profiler for Rent_Contracts.csv with RERA-specific checks."""
    
    BEDROOM_PATTERNS = {
        r'studio': 'Studio',
        r'1\s*bed|1\s*br|1br|one\s*bed': '1BR',
        r'2\s*bed|2\s*br|2br|two\s*bed': '2BR',
        r'3\s*bed|3\s*br|3br|three\s*bed': '3BR',
        r'4\s*bed|4\s*br|4br|four\s*bed': '4BR',
        r'5\s*bed|5\s*br|5br|five\s*bed': '5BR+',
        r'6\s*bed|6\s*br|6br': '5BR+',
        r'7\s*bed|7\s*br|7br': '5BR+',
    }
    
    def __init__(self, filepath: str):
        super().__init__(filepath)
        self.profile["domain_analysis"] = {
            "bedroom_parsing": {
                "success": 0, 
                "failure": 0, 
                "parsed_values": {},
                "unparseable_samples": []
            },
            "rent_distribution": {},
            "property_usage_types": {},
            "date_coverage": {"min_date": None, "max_date": None},
            "contract_types": {},
        }
    
    def _process_chunk(self, chunk: pd.DataFrame, chunk_num: int):
        super()._process_chunk(chunk, chunk_num)
        
        # Bedroom parsing analysis
        bedroom_col = None
        for col in ["ejari_property_sub_type_en", "property_sub_type_en", "rooms_en"]:
            if col in chunk.columns:
                bedroom_col = col
                break
        
        if bedroom_col:
            self._analyze_bedroom_parsing(chunk[bedroom_col])
        
        # Rent distribution
        if "annual_amount" in chunk.columns:
            rents = pd.to_numeric(chunk["annual_amount"], errors="coerce")
            self._update_rent_stats(rents.dropna())
        
        # Property usage types (Residential vs Commercial)
        for col in ["property_usage_en", "property_type_en"]:
            if col in chunk.columns:
                for ptype, count in chunk[col].value_counts().items():
                    self.profile["domain_analysis"]["property_usage_types"][str(ptype)] = \
                        self.profile["domain_analysis"]["property_usage_types"].get(str(ptype), 0) + int(count)
        
        # Date coverage
        for col in ["contract_start_date", "start_date"]:
            if col in chunk.columns:
                self._update_date_coverage(chunk[col])
        
        # Contract types (new vs renewal)
        if "is_renewal" in chunk.columns or "contract_type" in chunk.columns:
            type_col = "is_renewal" if "is_renewal" in chunk.columns else "contract_type"
            for ctype, count in chunk[type_col].value_counts().items():
                self.profile["domain_analysis"]["contract_types"][str(ctype)] = \
                    self.profile["domain_analysis"]["contract_types"].get(str(ctype), 0) + int(count)
    
    def _analyze_bedroom_parsing(self, series: pd.Series):
        """Test bedroom parsing patterns."""
        for val in series.dropna().unique():
            val_str = str(val).lower()
            matched = False
            
            for pattern, label in self.BEDROOM_PATTERNS.items():
                if re.search(pattern, val_str):
                    self.profile["domain_analysis"]["bedroom_parsing"]["success"] += 1
                    parsed = self.profile["domain_analysis"]["bedroom_parsing"]["parsed_values"]
                    parsed[label] = parsed.get(label, 0) + 1
                    matched = True
                    break
            
            if not matched:
                self.profile["domain_analysis"]["bedroom_parsing"]["failure"] += 1
                samples = self.profile["domain_analysis"]["bedroom_parsing"]["unparseable_samples"]
                if len(samples) < 50 and str(val) not in samples:
                    samples.append(str(val))
    
    def _update_rent_stats(self, rents: pd.Series):
        """Track rent distribution."""
        stats = self.profile["domain_analysis"]["rent_distribution"]
        
        buckets = [
            (0, 20_000, "0-20K"),
            (20_000, 40_000, "20K-40K"),
            (40_000, 60_000, "40K-60K"),
            (60_000, 80_000, "60K-80K"),
            (80_000, 100_000, "80K-100K"),
            (100_000, 150_000, "100K-150K"),
            (150_000, 200_000, "150K-200K"),
            (200_000, 300_000, "200K-300K"),
            (300_000, 500_000, "300K-500K"),
            (500_000, float('inf'), "500K+"),
        ]
        
        for low, high, label in buckets:
            count = int(((rents >= low) & (rents < high)).sum())
            stats[label] = stats.get(label, 0) + count
    
    def _update_date_coverage(self, date_series: pd.Series):
        """Track date range coverage."""
        try:
            dates = pd.to_datetime(date_series, format="%d-%m-%Y", errors="coerce")
            valid_dates = dates.dropna()
            
            if len(valid_dates) > 0:
                chunk_min = valid_dates.min()
                chunk_max = valid_dates.max()
                
                current_min = self.profile["domain_analysis"]["date_coverage"]["min_date"]
                current_max = self.profile["domain_analysis"]["date_coverage"]["max_date"]
                
                if current_min is None or str(chunk_min) < current_min:
                    self.profile["domain_analysis"]["date_coverage"]["min_date"] = str(chunk_min.date())
                if current_max is None or str(chunk_max) > current_max:
                    self.profile["domain_analysis"]["date_coverage"]["max_date"] = str(chunk_max.date())
        except Exception as e:
            pass


def main():
    parser = argparse.ArgumentParser(description="Profile large CSV files")
    parser.add_argument("filepath", nargs="?", help="Path to CSV file")
    parser.add_argument("--all", action="store_true", help="Profile all large files")
    parser.add_argument("--output", default="data_profiles", help="Output directory")
    
    args = parser.parse_args()
    
    if args.all:
        # Profile all large files
        large_files = {
            "Data/Transactions.csv": TransactionsProfiler,
            "Data/Rent_Contracts.csv": RentContractsProfiler,
            "Data/Units.csv": LargeCSVProfiler,
        }
        
        for filepath, profiler_class in large_files.items():
            if Path(filepath).exists():
                profiler = profiler_class(filepath)
                profiler.profile_file()
                profiler.print_summary()
                profiler.save_profile(args.output)
            else:
                print(f"Warning: {filepath} not found, skipping")
    
    elif args.filepath:
        # Profile single file
        filepath = Path(args.filepath)
        
        if not filepath.exists():
            print(f"Error: {filepath} not found")
            sys.exit(1)
        
        # Select appropriate profiler
        if "transaction" in filepath.name.lower():
            profiler = TransactionsProfiler(str(filepath))
        elif "rent" in filepath.name.lower():
            profiler = RentContractsProfiler(str(filepath))
        else:
            profiler = LargeCSVProfiler(str(filepath))
        
        profiler.profile_file()
        profiler.print_summary()
        profiler.save_profile(args.output)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

