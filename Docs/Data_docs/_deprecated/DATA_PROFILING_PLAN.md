# Data Profiling & Cleaning Plan

**Document Version:** 1.0  
**Created:** December 9, 2024  
**Status:** Planning  
**Owner:** Data Engineering Team

---

## 1. Executive Summary

Before building any ML models or ETL pipelines, we must understand the actual quality of our data. This document outlines a systematic approach to profile all data sources, identify quality issues, and create a cleaning strategy.

### Why This Matters

The PRD assumes data quality that has not been verified:
- **Rent_Contracts.csv** has ~9.5M rows - parsing success rates are unknown
- **Entity resolution** (matching "Dubai Marina" across files) has unknown complexity
- **Tourism data** has 54 Excel files with Arabic headers - translation coverage unknown
- **Date formats**, null rates, outliers - all undocumented

**Risk**: Starting development without this analysis could waste weeks on issues discovered late.

---

## 2. Data Inventory

### 2.1 Core Data Files

| File | Estimated Size | Estimated Rows | Profiling Strategy |
|------|---------------|----------------|-------------------|
| `Transactions.csv` | >200MB | ~1.6M+ | Chunked profiling |
| `Units.csv` | >200MB | ~2.3M+ | Chunked profiling |
| `Buildings.csv` | ~50MB | ~239K | Full profiling |
| `Projects.csv` | ~2MB | ~3K | Full profiling |
| `Rent_Contracts.csv` | ~4.2GB | ~9.5M | Chunked profiling |
| `Valuation.csv` | ~20MB | ~87K | Full profiling |

### 2.2 Tourism Data Files

| Category | File Count | Format | Language |
|----------|-----------|--------|----------|
| Visitors by Geographic Region | 14 | xlsx | Arabic |
| Hotel/Apartment Occupancy | 16 | xlsx | Arabic |
| Hotel Rooms & Apartments Inventory | 12 | xlsx | Arabic |
| Hotels & Apartment Buildings | 12 | xlsx | Arabic |
| FDI Reports | 4 | pdf | English |
| **Total** | **58** | - | - |

### 2.3 Initial Schema Discovery

Based on file sampling:

**Buildings.csv Columns (45 columns):**
```
property_id, area_id, zone_id, area_name_ar, area_name_en, land_number, 
land_sub_number, building_number, common_area, actual_common_area, floors, 
rooms, rooms_ar, rooms_en, car_parks, built_up_area, bld_levels, shops, 
flats, offices, swimming_pools, elevators, actual_area, property_type_id, 
property_type_ar, property_type_en, property_sub_type_id, property_sub_type_ar, 
property_sub_type_en, parent_property_id, creation_date, parcel_id, is_free_hold, 
is_lease_hold, is_registered, pre_registration_number, master_project_id, 
master_project_en, master_project_ar, project_id, project_name_ar, project_name_en, 
land_type_id, land_type_ar, land_type_en
```

**Projects.csv Columns (37 columns):**
```
project_id, project_number, project_name, developer_id, developer_number, 
developer_name, master_developer_id, master_developer_number, master_developer_name, 
project_start_date, project_end_date, project_type_id, project_type_ar, 
project_classification_id, project_classification_ar, escrow_agent_id, 
escrow_agent_name, project_status, project_status_ar, percent_completed, 
completion_date, cancellation_date, project_description_ar, project_description_en, 
property_id, area_id, area_name_ar, area_name_en, master_project_ar, master_project_en, 
zoning_authority_id, zoning_authority_ar, zoning_authority_en, no_of_lands, 
no_of_buildings, no_of_villas, no_of_units
```

**Valuation.csv Columns (20 columns):**
```
procedure_id, procedure_name_ar, procedure_name_en, procedure_year, 
procedure_number, instance_date, actual_worth, row_status_code, procedure_area, 
property_type_id, property_type_ar, property_type_en, property_sub_type_id, 
property_sub_type_ar, property_sub_type_en, area_id, area_name_ar, area_name_en, 
actual_area, property_total_value
```

---

## 3. Profiling Execution Plan

### 3.1 Timeline Overview

| Day | Phase | Tasks | Deliverables |
|-----|-------|-------|--------------|
| 1 | Setup | Environment setup, install dependencies | `scripts/` folder, `requirements-profiling.txt` |
| 2 | Core Profiling | Profile `Transactions.csv` | `data_profiles/transactions_profile.json` |
| 3 | Core Profiling | Profile `Rent_Contracts.csv`, `Units.csv` | Profile JSONs |
| 3 | Core Profiling | Profile small files | HTML reports |
| 4 | Entity Analysis | Run entity resolution analysis | Entity mapping files |
| 5 | Tourism | Profile tourism Excel files | Tourism catalog |
| 6 | Tourism | Manual translation review | Updated translations |
| 7 | Reporting | Generate final quality report | `DATA_QUALITY_REPORT.md` |

### 3.2 Phase Details

#### Day 1: Environment Setup

**Objective:** Create profiling infrastructure

**Tasks:**
- [ ] Create `scripts/` directory structure
- [ ] Create `requirements-profiling.txt`
- [ ] Set up `data_profiles/` output directory
- [ ] Test chunked CSV reading on sample

**Success Criteria:**
- Can read 100K rows from each large file
- All dependencies installed

---

#### Days 2-3: Core Data Profiling

**Objective:** Profile all CSV files and generate quality metrics

**Tasks:**
- [ ] Run `TransactionsProfiler` on Transactions.csv
- [ ] Run `RentContractsProfiler` on Rent_Contracts.csv  
- [ ] Run `LargeCSVProfiler` on Units.csv
- [ ] Generate ydata-profiling reports for Buildings, Projects, Valuation

**Key Metrics to Extract:**

| Metric | Why It Matters |
|--------|----------------|
| Null rates per column | Identifies unusable columns |
| Date parsing success | Validates temporal analysis feasibility |
| Bedroom parsing success | Critical for RERA lookups |
| Price distribution | Identifies outliers for filtering |
| Unique area names | Scopes entity resolution work |
| Unique project names | Scopes entity resolution work |

---

#### Day 4: Entity Resolution Analysis

**Objective:** Understand cross-dataset matching complexity

**Tasks:**
- [ ] Extract all unique area names from all files
- [ ] Extract all unique project names from all files
- [ ] Identify spelling variations and conflicts
- [ ] Generate canonical master list candidates
- [ ] Flag entries requiring manual review

**Deliverables:**
- `master_entity_list.json` - Proposed canonical names
- `conflicts.json` - Same entity with different spellings
- `similar_entities.json` - Fuzzy match candidates (>85% similarity)

---

#### Days 5-6: Tourism Data Processing

**Objective:** Catalog and translate Arabic tourism files

**Tasks:**
- [ ] Categorize all 54 Excel files by type
- [ ] Extract all unique column headers
- [ ] Apply Arabic→English translation mapping
- [ ] Identify untranslated columns for manual review
- [ ] Document file structure patterns

**Arabic Translation Reference:**

| Arabic | English | Context |
|--------|---------|---------|
| الزوار | Visitors | File category |
| المنطقة الجغرافية | Geographic Region | Column header |
| التوزيع النسبي | Percentage Distribution | File category |
| الفنادق | Hotels | File category |
| الشقق الفندقية | Hotel Apartments | File category |
| الغرف الفندقية | Hotel Rooms | Column header |
| إشغال | Occupancy | Metric type |
| درجة التصنيف | Classification/Rating | Column header |
| الربع الأول/الثاني/الثالث/الرابع | Q1/Q2/Q3/Q4 | Time period |

---

#### Day 7: Final Reporting

**Objective:** Synthesize findings into actionable report

**Tasks:**
- [ ] Compile all profiling results
- [ ] Calculate overall quality scores
- [ ] Prioritize cleaning actions
- [ ] Update PRD timeline if needed
- [ ] Document blocking issues

---

## 4. Profiling Scripts

### 4.1 Directory Structure

```
Properly/
├── Data/
│   ├── Buildings.csv
│   ├── Projects.csv
│   ├── Rent_Contracts.csv
│   ├── Toursim Data/
│   │   └── [58 files]
│   ├── Transactions.csv
│   ├── Units.csv
│   └── Valuation.csv
├── Docs/
│   ├── DATA_PROFILING_PLAN.md (this file)
│   └── ...
├── scripts/
│   ├── __init__.py
│   ├── profile_large_csv.py
│   ├── profile_small_csv.py
│   ├── analyze_entity_resolution.py
│   ├── profile_tourism_data.py
│   └── generate_quality_report.py
├── data_profiles/
│   ├── transactions_profile.json
│   ├── rent_contracts_profile.json
│   ├── units_profile.json
│   ├── detailed_reports/
│   │   ├── Buildings_profile.html
│   │   ├── Projects_profile.html
│   │   └── Valuation_profile.html
│   ├── entity_resolution/
│   │   ├── master_entity_list.json
│   │   ├── conflicts.json
│   │   ├── similar_entities.json
│   │   └── summary.json
│   └── tourism/
│       ├── tourism_catalog.json
│       ├── column_translations.json
│       ├── files_by_category.json
│       └── summary.json
└── requirements-profiling.txt
```

### 4.2 Requirements File

```
# requirements-profiling.txt
# Data Profiling Dependencies

# Core
pandas>=2.0.0
numpy>=1.24.0

# Large file handling
dask[complete]>=2024.1.0
pyarrow>=14.0.0

# Excel processing
openpyxl>=3.1.0
xlrd>=2.0.1

# Profiling
ydata-profiling>=4.6.0

# Arabic text handling
arabic-reshaper>=3.0.0
python-bidi>=0.4.2

# Progress and visualization
tqdm>=4.66.0
matplotlib>=3.8.0
seaborn>=0.13.0

# Fuzzy matching
python-Levenshtein>=0.23.0
rapidfuzz>=3.5.0
```

### 4.3 Main Profiling Script: Large CSV Files

```python
# scripts/profile_large_csv.py
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
```

### 4.4 Small File Profiler Script

```python
# scripts/profile_small_csv.py
"""
Generate detailed profiling reports for smaller CSV files.
Uses ydata-profiling for comprehensive analysis.

Usage:
    python scripts/profile_small_csv.py
    python scripts/profile_small_csv.py --file Data/Projects.csv
"""

import pandas as pd
from pathlib import Path
import argparse
import sys

try:
    from ydata_profiling import ProfileReport
    YDATA_AVAILABLE = True
except ImportError:
    YDATA_AVAILABLE = False
    print("Warning: ydata-profiling not installed. Using basic profiling.")


def basic_profile(df: pd.DataFrame, title: str, output_path: Path):
    """Generate basic profile without ydata-profiling."""
    profile = {
        "title": title,
        "rows": len(df),
        "columns": len(df.columns),
        "memory_mb": df.memory_usage(deep=True).sum() / 1024 / 1024,
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
            col_profile["min"] = float(df[col].min()) if not df[col].isna().all() else None
            col_profile["max"] = float(df[col].max()) if not df[col].isna().all() else None
            col_profile["mean"] = float(df[col].mean()) if not df[col].isna().all() else None
        
        profile["column_profiles"][col] = col_profile
    
    # Save as JSON
    import json
    with open(output_path.with_suffix('.json'), 'w', encoding='utf-8') as f:
        json.dump(profile, f, indent=2, ensure_ascii=False)
    
    print(f"Basic profile saved to: {output_path.with_suffix('.json')}")
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
    print(f"Full profile saved to: {output_path}")
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
        else:
            basic_profile(df, f"Data Profile: {filepath.name}", output_path)
        
        return True
        
    except Exception as e:
        print(f"  Error: {e}")
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
```

### 4.5 Entity Resolution Analysis Script

```python
# scripts/analyze_entity_resolution.py
"""
Analyze entity resolution requirements across datasets.
Identifies matching challenges for area names, project names, etc.

Usage:
    python scripts/analyze_entity_resolution.py
"""

import pandas as pd
import json
from pathlib import Path
from collections import defaultdict
import re
from difflib import SequenceMatcher
from tqdm import tqdm


class EntityResolutionAnalyzer:
    """Analyze entity matching requirements across datasets."""
    
    def __init__(self):
        self.entities = defaultdict(lambda: defaultdict(set))
        # entities[entity_type][normalized_form] = {(source_file, original_value), ...}
        
    def normalize(self, text: str) -> str:
        """Normalize text for matching."""
        if pd.isna(text):
            return ""
        text = str(text).lower().strip()
        # Remove special characters
        text = re.sub(r'[^\w\s]', '', text)
        # Collapse whitespace
        text = re.sub(r'\s+', ' ', text)
        return text
    
    def extract_entities_from_df(self, df: pd.DataFrame, source_file: str, entity_columns: dict):
        """
        Extract entities from a DataFrame.
        
        entity_columns: {entity_type: [column_names]}
        Example: {"area": ["area_name_en", "master_project_en"]}
        """
        for entity_type, columns in entity_columns.items():
            for col in columns:
                if col not in df.columns:
                    continue
                
                for val in df[col].dropna().unique():
                    normalized = self.normalize(val)
                    if normalized:
                        self.entities[entity_type][normalized].add((source_file, str(val)))
    
    def find_conflicts(self) -> dict:
        """Find entities with multiple spellings."""
        conflicts = {}
        
        for entity_type, normalized_map in self.entities.items():
            type_conflicts = {}
            
            for normalized, sources in normalized_map.items():
                # Get unique spellings
                spellings = list(set(s[1] for s in sources))
                if len(spellings) > 1:
                    type_conflicts[normalized] = {
                        "spellings": spellings,
                        "sources": list(set(s[0] for s in sources)),
                        "count": len(spellings)
                    }
            
            if type_conflicts:
                # Sort by number of conflicts
                conflicts[entity_type] = dict(
                    sorted(type_conflicts.items(), key=lambda x: -x[1]["count"])
                )
        
        return conflicts
    
    def find_similar_entities(self, threshold: float = 0.85) -> dict:
        """Find potentially duplicate entities using fuzzy matching."""
        similar = {}
        
        for entity_type, normalized_map in self.entities.items():
            normalized_keys = list(normalized_map.keys())
            type_similar = []
            
            print(f"  Analyzing {entity_type}: {len(normalized_keys)} unique entities...")
            
            # Limit comparisons for performance
            max_comparisons = min(len(normalized_keys), 500)
            
            for i, key1 in enumerate(tqdm(normalized_keys[:max_comparisons], 
                                          desc=f"    {entity_type}", leave=False)):
                for key2 in normalized_keys[i+1:max_comparisons]:
                    # Quick length check first (optimization)
                    if abs(len(key1) - len(key2)) / max(len(key1), len(key2)) > 0.3:
                        continue
                    
                    ratio = SequenceMatcher(None, key1, key2).ratio()
                    if ratio >= threshold and ratio < 1.0:
                        type_similar.append({
                            "entity_1": key1,
                            "entity_2": key2,
                            "similarity": round(ratio, 3),
                            "spellings_1": list(set(s[1] for s in normalized_map[key1]))[:3],
                            "spellings_2": list(set(s[1] for s in normalized_map[key2]))[:3],
                        })
            
            if type_similar:
                similar[entity_type] = sorted(type_similar, key=lambda x: -x["similarity"])[:100]
        
        return similar
    
    def generate_master_list(self) -> dict:
        """Generate a canonical master list for each entity type."""
        master_list = {}
        
        for entity_type, normalized_map in self.entities.items():
            entries = []
            
            for normalized, sources in normalized_map.items():
                # Pick the most common spelling as canonical
                spellings = [s[1] for s in sources]
                spelling_counts = defaultdict(int)
                for s in spellings:
                    spelling_counts[s] += 1
                
                canonical = max(spelling_counts, key=spelling_counts.get)
                
                entries.append({
                    "canonical": canonical,
                    "normalized": normalized,
                    "alternate_spellings": list(set(spellings) - {canonical}),
                    "sources": list(set(s[0] for s in sources)),
                })
            
            master_list[entity_type] = sorted(entries, key=lambda x: x["canonical"])
        
        return master_list


def run_entity_analysis():
    """Run full entity resolution analysis."""
    DATA_DIR = Path("Data")
    OUTPUT_DIR = Path("data_profiles/entity_resolution")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    analyzer = EntityResolutionAnalyzer()
    
    print("="*60)
    print("Entity Resolution Analysis")
    print("="*60)
    
    # Define entity columns for each file
    file_entity_map = {
        "Buildings.csv": {
            "area": ["area_name_en"],
            "project": ["master_project_en", "project_name_en"],
        },
        "Projects.csv": {
            "area": ["area_name_en"],
            "project": ["master_project_en", "project_name"],
            "developer": ["developer_name", "master_developer_name"],
        },
        "Valuation.csv": {
            "area": ["area_name_en"],
        },
    }
    
    # Process smaller files fully
    print("\nProcessing small files...")
    for filename, entity_columns in file_entity_map.items():
        filepath = DATA_DIR / filename
        if filepath.exists():
            print(f"  Loading {filename}...")
            df = pd.read_csv(filepath, low_memory=False)
            analyzer.extract_entities_from_df(df, filename, entity_columns)
        else:
            print(f"  Skipping {filename} - not found")
    
    # Process large files in chunks
    large_file_entities = {
        "Transactions.csv": {
            "area": ["area_name_en"],
            "project": ["master_project_en", "project_name_en"],
        },
        "Rent_Contracts.csv": {
            "area": ["area_name_en"],
            "project": ["master_project_en"],
        },
    }
    
    print("\nProcessing large files (chunked)...")
    for filename, entity_columns in large_file_entities.items():
        filepath = DATA_DIR / filename
        if filepath.exists():
            print(f"  Processing {filename}...")
            chunk_count = 0
            for chunk in pd.read_csv(filepath, chunksize=100_000, low_memory=False):
                analyzer.extract_entities_from_df(chunk, filename, entity_columns)
                chunk_count += 1
                if chunk_count >= 10:  # Limit chunks for entity analysis
                    break
            print(f"    Processed {chunk_count} chunks")
        else:
            print(f"  Skipping {filename} - not found")
    
    # Generate reports
    print("\nGenerating reports...")
    
    # 1. Summary
    summary = {
        "entity_counts": {k: len(v) for k, v in analyzer.entities.items()},
    }
    
    # 2. Conflicts
    print("  Finding spelling conflicts...")
    conflicts = analyzer.find_conflicts()
    summary["conflict_counts"] = {k: len(v) for k, v in conflicts.items()}
    
    with open(OUTPUT_DIR / "conflicts.json", "w", encoding="utf-8") as f:
        json.dump(conflicts, f, indent=2, ensure_ascii=False)
    
    # 3. Similar entities
    print("  Finding similar entities (fuzzy matching)...")
    similar = analyzer.find_similar_entities(threshold=0.85)
    summary["similar_pair_counts"] = {k: len(v) for k, v in similar.items()}
    
    with open(OUTPUT_DIR / "similar_entities.json", "w", encoding="utf-8") as f:
        json.dump(similar, f, indent=2, ensure_ascii=False)
    
    # 4. Master list
    print("  Generating master entity list...")
    master_list = analyzer.generate_master_list()
    
    with open(OUTPUT_DIR / "master_entity_list.json", "w", encoding="utf-8") as f:
        json.dump(master_list, f, indent=2, ensure_ascii=False)
    
    # 5. Save summary
    with open(OUTPUT_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("ENTITY RESOLUTION SUMMARY")
    print("="*60)
    print(f"\n{'Entity Type':<20} {'Unique':<10} {'Conflicts':<10} {'Similar Pairs':<15}")
    print("-"*60)
    for entity_type in analyzer.entities.keys():
        unique = summary["entity_counts"].get(entity_type, 0)
        conf = summary["conflict_counts"].get(entity_type, 0)
        sim = summary["similar_pair_counts"].get(entity_type, 0)
        print(f"{entity_type:<20} {unique:<10} {conf:<10} {sim:<15}")
    
    print(f"\nResults saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    run_entity_analysis()
```

### 4.6 Tourism Data Profiler Script

```python
# scripts/profile_tourism_data.py
"""
Profile and catalog Tourism data files.
Handles Arabic column headers and generates translation mapping.

Usage:
    python scripts/profile_tourism_data.py
"""

import pandas as pd
from pathlib import Path
import json
import re
from collections import defaultdict
from datetime import datetime


# Arabic to English translation mapping
ARABIC_TRANSLATIONS = {
    # File name components
    "الزوار": "Visitors",
    "المنطقة الجغرافية": "Geographic_Region",
    "التوزيع النسبي": "Percentage_Distribution",
    "الفنادق": "Hotels",
    "الشقق الفندقية": "Hotel_Apartments",
    "الغرف الفندقية": "Hotel_Rooms",
    "إشغال": "Occupancy",
    "درجة التصنيف": "Classification",
    "مبانى": "Buildings",
    "متوسط": "Average",
    "حسب": "By",
    "و": "And",
    "ومتوسط": "And_Average",
    
    # Quarter names
    "الربع الأول": "Q1",
    "الربع الثاني": "Q2",
    "الربع الثالث": "Q3",
    "الربع الرابع": "Q4",
    
    # Column headers (common ones)
    "المنطقة": "Region",
    "الدولة": "Country",
    "الجنسية": "Nationality",
    "عدد الزوار": "Visitor_Count",
    "عدد": "Count",
    "النسبة": "Percentage",
    "نسبة الإشغال": "Occupancy_Rate",
    "عدد الغرف": "Room_Count",
    "عدد الفنادق": "Hotel_Count",
    "عدد الشقق": "Apartment_Count",
    "التصنيف": "Classification",
    "الفئة": "Category",
    "نجوم": "Stars",
    "5 نجوم": "5_Star",
    "4 نجوم": "4_Star",
    "3 نجوم": "3_Star",
    "2 نجوم": "2_Star",
    "1 نجمة": "1_Star",
    "ديلوكس": "Deluxe",
    "ستاندرد": "Standard",
    "الشهر": "Month",
    "السنة": "Year",
    "المجموع": "Total",
    "الإجمالي": "Grand_Total",
}


class TourismDataProfiler:
    """Profile and catalog tourism data files."""
    
    def __init__(self, tourism_dir: str):
        self.tourism_dir = Path(tourism_dir)
        self.catalog = []
        self.column_translations = {}
        self.file_categories = defaultdict(list)
        self.all_columns = set()
        
    def categorize_filename(self, filename: str) -> dict:
        """Extract metadata from Arabic filename."""
        info = {
            "original_name": filename,
            "category": "unknown",
            "subcategory": None,
            "quarter": None,
            "year": None,
            "english_name": None,
        }
        
        # Extract year
        year_match = re.search(r'20\d{2}', filename)
        if year_match:
            info["year"] = int(year_match.group())
        
        # Extract quarter
        if "الربع الأول" in filename:
            info["quarter"] = "Q1"
        elif "الربع الثاني" in filename:
            info["quarter"] = "Q2"
        elif "الربع الثالث" in filename:
            info["quarter"] = "Q3"
        elif "الربع الرابع" in filename:
            info["quarter"] = "Q4"
        
        # Categorize by content
        if "الزوار" in filename:
            info["category"] = "visitors"
            if "التوزيع النسبي" in filename:
                info["subcategory"] = "percentage_distribution"
                info["english_name"] = f"Visitor_Percentage_Distribution_{info['quarter']}_{info['year']}"
            else:
                info["subcategory"] = "absolute_numbers"
                info["english_name"] = f"Visitors_By_Region_{info['quarter']}_{info['year']}"
                
        elif "إشغال" in filename:
            info["category"] = "occupancy"
            if "الشقق" in filename:
                info["subcategory"] = "hotel_apartments"
                info["english_name"] = f"Hotel_Apartment_Occupancy_{info['quarter']}_{info['year']}"
            else:
                info["subcategory"] = "hotels"
                info["english_name"] = f"Hotel_Occupancy_{info['quarter']}_{info['year']}"
                
        elif "الغرف الفندقية" in filename or "الشقق الفندقية" in filename:
            info["category"] = "inventory"
            info["subcategory"] = "rooms_and_apartments"
            info["english_name"] = f"Room_Apartment_Inventory_{info['quarter']}_{info['year']}"
            
        elif "الفنادق" in filename and "مبانى" in filename:
            info["category"] = "inventory"
            info["subcategory"] = "hotel_buildings"
            info["english_name"] = f"Hotel_Building_Inventory_{info['quarter']}_{info['year']}"
        
        elif "الفنادق" in filename and "الشقق" in filename:
            info["category"] = "inventory"
            info["subcategory"] = "combined"
            info["english_name"] = f"Hotels_Apartments_Combined_{info['year']}"
            
        elif "fdi" in filename.lower():
            info["category"] = "fdi_report"
            info["subcategory"] = "pdf"
            info["english_name"] = filename
        
        return info
    
    def translate_column(self, arabic_col: str) -> str:
        """Translate Arabic column header to English."""
        if pd.isna(arabic_col):
            return "EMPTY"
        
        col_str = str(arabic_col).strip()
        
        # Check direct translation
        if col_str in ARABIC_TRANSLATIONS:
            return ARABIC_TRANSLATIONS[col_str]
        
        # Try partial matches (longest first)
        english_parts = []
        remaining = col_str
        
        for ar, en in sorted(ARABIC_TRANSLATIONS.items(), key=lambda x: -len(x[0])):
            if ar in remaining:
                english_parts.append(en)
                remaining = remaining.replace(ar, " ")
        
        remaining = remaining.strip()
        
        if english_parts:
            result = "_".join(english_parts)
            if remaining:
                result += f"_{remaining[:20]}"
            return result
        
        # Check if it's already English/numeric
        if re.match(r'^[a-zA-Z0-9\s_-]+$', col_str):
            return col_str.replace(" ", "_")
        
        return f"UNTRANSLATED_{col_str[:30]}"
    
    def profile_excel_file(self, filepath: Path) -> dict:
        """Profile a single Excel file."""
        profile = {
            "filepath": str(filepath.name),
            "file_info": self.categorize_filename(filepath.name),
            "sheets": [],
            "errors": [],
        }
        
        try:
            xlsx = pd.ExcelFile(filepath)
            
            for sheet_name in xlsx.sheet_names:
                sheet_profile = {
                    "name": sheet_name,
                    "columns": [],
                    "rows": 0,
                }
                
                try:
                    df = pd.read_excel(xlsx, sheet_name=sheet_name)
                    sheet_profile["rows"] = len(df)
                    
                    for col in df.columns:
                        col_str = str(col)
                        translated = self.translate_column(col_str)
                        
                        col_info = {
                            "original": col_str,
                            "translated": translated,
                            "dtype": str(df[col].dtype),
                            "null_pct": round(df[col].isna().mean() * 100, 2),
                        }
                        
                        # Add sample values for non-empty columns
                        non_null = df[col].dropna()
                        if len(non_null) > 0:
                            col_info["sample_values"] = [str(v) for v in non_null.head(3).tolist()]
                        
                        sheet_profile["columns"].append(col_info)
                        
                        # Track all columns and translations
                        self.all_columns.add(col_str)
                        self.column_translations[col_str] = translated
                    
                except Exception as e:
                    sheet_profile["error"] = str(e)
                
                profile["sheets"].append(sheet_profile)
                
        except Exception as e:
            profile["errors"].append(str(e))
        
        return profile
    
    def profile_all_files(self) -> dict:
        """Profile all tourism data files."""
        print("="*60)
        print("Tourism Data Profiling")
        print("="*60)
        
        # Get all files
        excel_files = sorted(self.tourism_dir.glob("*.xlsx"))
        pdf_files = sorted(self.tourism_dir.glob("*.pdf"))
        
        print(f"\nFound {len(excel_files)} Excel files and {len(pdf_files)} PDF files")
        
        # Profile Excel files
        print("\nProcessing Excel files...")
        for filepath in excel_files:
            short_name = filepath.name[:50] + "..." if len(filepath.name) > 50 else filepath.name
            print(f"  {short_name}")
            
            profile = self.profile_excel_file(filepath)
            self.catalog.append(profile)
            
            # Organize by category
            category = profile["file_info"]["category"]
            self.file_categories[category].append({
                "file": filepath.name,
                "english_name": profile["file_info"]["english_name"],
                "year": profile["file_info"]["year"],
                "quarter": profile["file_info"]["quarter"],
            })
        
        # Note PDF files
        print("\nPDF files (require manual extraction):")
        for filepath in pdf_files:
            print(f"  {filepath.name}")
            profile = {
                "filepath": filepath.name,
                "file_info": self.categorize_filename(filepath.name),
                "note": "PDF file - requires manual extraction or OCR",
            }
            self.catalog.append(profile)
            self.file_categories["pdf"].append(filepath.name)
        
        return self._generate_summary()
    
    def _generate_summary(self) -> dict:
        """Generate summary of tourism data."""
        # Count untranslated columns
        untranslated = [k for k, v in self.column_translations.items() 
                       if v.startswith("UNTRANSLATED_")]
        
        return {
            "profiled_at": datetime.now().isoformat(),
            "total_files": len(self.catalog),
            "excel_files": len([c for c in self.catalog if c["filepath"].endswith(".xlsx")]),
            "pdf_files": len([c for c in self.catalog if c["filepath"].endswith(".pdf")]),
            "categories": {
                k: {
                    "count": len(v),
                    "files": v if k == "pdf" else None
                } 
                for k, v in self.file_categories.items()
            },
            "year_coverage": self._get_year_coverage(),
            "unique_columns_found": len(self.all_columns),
            "untranslated_column_count": len(untranslated),
            "translation_coverage_pct": round(
                (len(self.all_columns) - len(untranslated)) / len(self.all_columns) * 100, 1
            ) if self.all_columns else 0,
        }
    
    def _get_year_coverage(self) -> list:
        """Get years covered in the data."""
        years = set()
        for profile in self.catalog:
            year = profile["file_info"].get("year")
            if year:
                years.add(year)
        return sorted(years)
    
    def save_results(self, output_dir: str = "data_profiles/tourism"):
        """Save profiling results."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Full catalog
        with open(output_path / "tourism_catalog.json", "w", encoding="utf-8") as f:
            json.dump(self.catalog, f, indent=2, ensure_ascii=False, default=str)
        
        # Column translations
        with open(output_path / "column_translations.json", "w", encoding="utf-8") as f:
            json.dump(dict(sorted(self.column_translations.items())), 
                     f, indent=2, ensure_ascii=False)
        
        # Files by category
        with open(output_path / "files_by_category.json", "w", encoding="utf-8") as f:
            json.dump(dict(self.file_categories), f, indent=2, ensure_ascii=False)
        
        # Untranslated columns (for manual review)
        untranslated = {k: v for k, v in self.column_translations.items() 
                       if v.startswith("UNTRANSLATED_")}
        with open(output_path / "untranslated_columns.json", "w", encoding="utf-8") as f:
            json.dump(untranslated, f, indent=2, ensure_ascii=False)
        
        # Summary
        summary = self._generate_summary()
        with open(output_path / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        # Print summary
        print("\n" + "="*60)
        print("TOURISM DATA SUMMARY")
        print("="*60)
        print(f"\nFiles processed: {summary['total_files']}")
        print(f"  - Excel: {summary['excel_files']}")
        print(f"  - PDF: {summary['pdf_files']}")
        print(f"\nYear coverage: {summary['year_coverage']}")
        print(f"\nColumn translation:")
        print(f"  - Unique columns found: {summary['unique_columns_found']}")
        print(f"  - Translation coverage: {summary['translation_coverage_pct']}%")
        print(f"  - Untranslated: {summary['untranslated_column_count']}")
        print(f"\nCategories:")
        for cat, info in summary['categories'].items():
            print(f"  - {cat}: {info['count']} files")
        
        print(f"\nResults saved to: {output_path}")


def main():
    tourism_dir = Path("Data/Toursim Data")
    
    if not tourism_dir.exists():
        print(f"Error: Tourism data directory not found: {tourism_dir}")
        return
    
    profiler = TourismDataProfiler(str(tourism_dir))
    profiler.profile_all_files()
    profiler.save_results()


if __name__ == "__main__":
    main()
```

### 4.7 Quality Report Generator

```python
# scripts/generate_quality_report.py
"""
Generate final data quality report from all profiling outputs.

Usage:
    python scripts/generate_quality_report.py
"""

import json
from pathlib import Path
from datetime import datetime


def load_json(filepath: Path) -> dict:
    """Load JSON file if exists."""
    if filepath.exists():
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def generate_report():
    """Generate comprehensive data quality report."""
    PROFILE_DIR = Path("data_profiles")
    OUTPUT_FILE = Path("Docs/DATA_QUALITY_REPORT.md")
    
    # Load all profiles
    transactions = load_json(PROFILE_DIR / "transactions_profile.json")
    rent_contracts = load_json(PROFILE_DIR / "rent_contracts_profile.json")
    units = load_json(PROFILE_DIR / "units_profile.json")
    entity_summary = load_json(PROFILE_DIR / "entity_resolution" / "summary.json")
    tourism_summary = load_json(PROFILE_DIR / "tourism" / "summary.json")
    
    # Generate report
    report = f"""# Data Quality Report

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M")}  
**Status:** {'Complete' if transactions else 'Partial - Run profiling scripts first'}

---

## 1. Executive Summary

"""
    
    # Dataset summary table
    report += "### Dataset Overview\n\n"
    report += "| Dataset | Rows | Columns | Quality Notes |\n"
    report += "|---------|------|---------|---------------|\n"
    
    if transactions:
        rows = transactions.get('total_rows', 'Unknown')
        cols = len(transactions.get('columns', {}))
        report += f"| Transactions | {rows:,} | {cols} | ✅ Profiled |\n"
    else:
        report += "| Transactions | - | - | ⚠️ Not yet profiled |\n"
    
    if rent_contracts:
        rows = rent_contracts.get('total_rows', 'Unknown')
        cols = len(rent_contracts.get('columns', {}))
        report += f"| Rent_Contracts | {rows:,} | {cols} | ✅ Profiled |\n"
    else:
        report += "| Rent_Contracts | - | - | ⚠️ Not yet profiled |\n"
    
    if units:
        rows = units.get('total_rows', 'Unknown')
        cols = len(units.get('columns', {}))
        report += f"| Units | {rows:,} | {cols} | ✅ Profiled |\n"
    else:
        report += "| Units | - | - | ⚠️ Not yet profiled |\n"
    
    report += "\n"
    
    # Critical findings
    report += "## 2. Critical Findings\n\n"
    
    # Transactions domain analysis
    if transactions and 'domain_analysis' in transactions:
        da = transactions['domain_analysis']
        report += "### 2.1 Transactions Analysis\n\n"
        
        if 'date_parsing' in da:
            dp = da['date_parsing']
            total = dp.get('success', 0) + dp.get('failure', 0)
            if total > 0:
                success_rate = dp['success'] / total * 100
                report += f"**Date Parsing Success Rate:** {success_rate:.1f}%\n\n"
                if dp.get('sample_failures'):
                    report += "Sample parsing failures:\n"
                    for f in dp['sample_failures'][:5]:
                        report += f"- `{f}`\n"
                    report += "\n"
        
        if 'procedure_types' in da:
            report += "**Transaction Types:**\n\n"
            for proc, count in sorted(da['procedure_types'].items(), key=lambda x: -x[1])[:10]:
                report += f"- {proc}: {count:,}\n"
            report += "\n"
        
        if 'price_distribution' in da:
            report += "**Price Distribution:**\n\n"
            report += "| Range | Count |\n"
            report += "|-------|-------|\n"
            for bucket, count in da['price_distribution'].items():
                report += f"| {bucket} | {count:,} |\n"
            report += "\n"
    
    # Rent contracts domain analysis
    if rent_contracts and 'domain_analysis' in rent_contracts:
        da = rent_contracts['domain_analysis']
        report += "### 2.2 Rent Contracts Analysis\n\n"
        
        if 'bedroom_parsing' in da:
            bp = da['bedroom_parsing']
            total = bp.get('success', 0) + bp.get('failure', 0)
            if total > 0:
                success_rate = bp['success'] / total * 100
                report += f"**Bedroom Parsing Success Rate:** {success_rate:.1f}%\n\n"
                
                if bp.get('parsed_values'):
                    report += "Parsed bedroom distribution:\n"
                    for br, count in sorted(bp['parsed_values'].items()):
                        report += f"- {br}: {count:,}\n"
                    report += "\n"
                
                if bp.get('unparseable_samples'):
                    report += "⚠️ Unparseable samples (need manual mapping):\n"
                    for s in bp['unparseable_samples'][:10]:
                        report += f"- `{s}`\n"
                    report += "\n"
        
        if 'date_coverage' in da:
            dc = da['date_coverage']
            report += f"**Date Coverage:** {dc.get('min_date', 'Unknown')} to {dc.get('max_date', 'Unknown')}\n\n"
    
    # Entity resolution
    if entity_summary:
        report += "### 2.3 Entity Resolution\n\n"
        
        if 'entity_counts' in entity_summary:
            report += "**Unique Entities Found:**\n\n"
            for entity_type, count in entity_summary['entity_counts'].items():
                report += f"- {entity_type}: {count:,}\n"
            report += "\n"
        
        if 'conflict_counts' in entity_summary:
            report += "**Spelling Conflicts (same entity, different spellings):**\n\n"
            for entity_type, count in entity_summary['conflict_counts'].items():
                report += f"- {entity_type}: {count:,} conflicts\n"
            report += "\n"
    
    # Tourism data
    if tourism_summary:
        report += "### 2.4 Tourism Data\n\n"
        report += f"- **Files processed:** {tourism_summary.get('total_files', 0)}\n"
        report += f"- **Year coverage:** {tourism_summary.get('year_coverage', [])}\n"
        report += f"- **Translation coverage:** {tourism_summary.get('translation_coverage_pct', 0)}%\n"
        report += f"- **Untranslated columns:** {tourism_summary.get('untranslated_column_count', 0)}\n\n"
    
    # Recommendations
    report += """## 3. Recommended Actions

### Priority 1 (Blocking)

1. **Fix date parsing** - Ensure DD-MM-YYYY format is consistently parsed
2. **Create bedroom mapping** - Map all unique bedroom strings to standard labels
3. **Resolve entity conflicts** - Review and approve canonical entity names

### Priority 2 (Important)

4. **Complete tourism translations** - Review untranslated Arabic columns
5. **Handle outliers** - Define filtering rules for extreme price values
6. **Document data gaps** - Identify temporal/geographic coverage gaps

### Priority 3 (Enhancement)

7. **Create data dictionary** - Document all column meanings and valid values
8. **Build validation tests** - Automated checks for data quality

---

## 4. Next Steps

1. [ ] Review this report
2. [ ] Address Priority 1 items
3. [ ] Re-run profiling to verify fixes
4. [ ] Proceed to Phase 1 (Database Setup)

"""
    
    # Write report
    OUTPUT_FILE.parent.mkdir(exist_ok=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"Report generated: {OUTPUT_FILE}")


if __name__ == "__main__":
    generate_report()
```

---

## 5. Expected Outputs

After running all profiling scripts:

```
data_profiles/
├── transactions_profile.json       # Transactions analysis
├── rent_contracts_profile.json     # Rent contracts analysis  
├── units_profile.json              # Units analysis
├── detailed_reports/
│   ├── Buildings_profile.html      # Interactive report
│   ├── Projects_profile.html
│   └── Valuation_profile.html
├── entity_resolution/
│   ├── master_entity_list.json     # Canonical names
│   ├── conflicts.json              # Spelling conflicts
│   ├── similar_entities.json       # Fuzzy matches
│   └── summary.json
└── tourism/
    ├── tourism_catalog.json        # All file profiles
    ├── column_translations.json    # Arabic→English
    ├── untranslated_columns.json   # For manual review
    ├── files_by_category.json
    └── summary.json
```

---

## 6. Success Criteria

| Metric | Target | Measured By |
|--------|--------|-------------|
| All core CSVs profiled | 100% | Profile JSONs exist |
| Date parsing success | >95% | domain_analysis.date_parsing |
| Bedroom parsing success | >90% | domain_analysis.bedroom_parsing |
| Entity conflicts identified | 100% | conflicts.json complete |
| Tourism files cataloged | 100% | tourism_catalog.json complete |
| Column translation coverage | >80% | summary.json metric |

---

## 7. Risk Mitigation

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| Large file processing fails | Medium | Use chunked processing, increase chunk size |
| Arabic translation incomplete | High | Flag untranslated for manual review |
| Entity resolution too complex | Medium | Focus on top 100 areas/projects first |
| Profiling takes too long | Medium | Limit chunks processed for entity analysis |

---

## 8. Dependencies on This Phase

The following phases **cannot start** until this profiling is complete:

- **Phase 1 (Foundation)**: Schema design depends on understanding actual column types/nulls
- **Phase 2 (ML Training)**: Feature engineering requires knowing data quality
- **Rent Aggregation**: Bedroom parsing must work before aggregating

---

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2024-12-09 | System | Initial plan created |

