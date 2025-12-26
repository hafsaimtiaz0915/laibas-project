#!/usr/bin/env python3
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
    
    # Load all profiles (match actual filenames)
    transactions = load_json(PROFILE_DIR / "Transactions_profile.json")
    rent_contracts = load_json(PROFILE_DIR / "Rent_Contracts_profile.json")
    units = load_json(PROFILE_DIR / "Units_profile.json")
    entity_summary = load_json(PROFILE_DIR / "entity_resolution" / "summary.json")
    tourism_summary = load_json(PROFILE_DIR / "tourism" / "summary.json")
    
    # Load detailed reports if available
    buildings_profile = load_json(PROFILE_DIR / "detailed_reports" / "Buildings_profile.json")
    projects_profile = load_json(PROFILE_DIR / "detailed_reports" / "Projects_profile.json")
    valuation_profile = load_json(PROFILE_DIR / "detailed_reports" / "Valuation_profile.json")
    
    # Generate report
    report = f"""# Data Quality Report

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M")}  
**Status:** {'Complete' if transactions else 'Partial - Run profiling scripts first'}

---

## 1. Executive Summary

"""
    
    # Dataset summary table
    report += "### Dataset Overview\n\n"
    report += "| Dataset | Rows | Columns | Null Rate | Status |\n"
    report += "|---------|------|---------|-----------|--------|\n"
    
    def format_row(name, profile_data, key="total_rows"):
        if profile_data:
            rows = profile_data.get(key, profile_data.get('rows', 'Unknown'))
            if isinstance(rows, int):
                rows = f"{rows:,}"
            # Handle both integer 'columns' and dict 'columns'/'column_profiles'
            cols = profile_data.get('columns', 0)
            if isinstance(cols, dict):
                cols = len(cols)
            elif not isinstance(cols, int):
                cols = len(profile_data.get('column_profiles', {}))
            return f"| {name} | {rows} | {cols} | - | ✅ Profiled |\n"
        return f"| {name} | - | - | - | ⚠️ Not profiled |\n"
    
    report += format_row("Transactions", transactions)
    report += format_row("Rent_Contracts", rent_contracts)
    report += format_row("Units", units)
    report += format_row("Buildings", buildings_profile, "rows")
    report += format_row("Projects", projects_profile, "rows")
    report += format_row("Valuation", valuation_profile, "rows")
    
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
                    report += "Sample parsing failures:\n```\n"
                    for f in dp['sample_failures'][:5]:
                        report += f"{f}\n"
                    report += "```\n\n"
        
        if 'procedure_types' in da and da['procedure_types']:
            report += "**Transaction Types:**\n\n"
            report += "| Type | Count |\n"
            report += "|------|-------|\n"
            for proc, count in sorted(da['procedure_types'].items(), key=lambda x: -x[1])[:10]:
                report += f"| {proc} | {count:,} |\n"
            report += "\n"
        
        if 'price_distribution' in da and da['price_distribution']:
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
                    report += "Parsed bedroom distribution:\n\n"
                    report += "| Bedroom Type | Count |\n"
                    report += "|--------------|-------|\n"
                    for br, count in sorted(bp['parsed_values'].items()):
                        report += f"| {br} | {count:,} |\n"
                    report += "\n"
                
                if bp.get('unparseable_samples'):
                    report += "⚠️ **Unparseable samples (need manual mapping):**\n```\n"
                    for s in bp['unparseable_samples'][:10]:
                        report += f"{s}\n"
                    report += "```\n\n"
        
        if 'date_coverage' in da:
            dc = da['date_coverage']
            report += f"**Date Coverage:** {dc.get('min_date', 'Unknown')} to {dc.get('max_date', 'Unknown')}\n\n"
        
        if 'rent_distribution' in da and da['rent_distribution']:
            report += "**Rent Distribution (Annual):**\n\n"
            report += "| Range | Count |\n"
            report += "|-------|-------|\n"
            for bucket, count in da['rent_distribution'].items():
                report += f"| {bucket} AED | {count:,} |\n"
            report += "\n"
    
    # Entity resolution
    if entity_summary:
        report += "### 2.3 Entity Resolution\n\n"
        
        if 'entity_counts' in entity_summary:
            report += "**Unique Entities Found:**\n\n"
            report += "| Entity Type | Count |\n"
            report += "|-------------|-------|\n"
            for entity_type, count in entity_summary['entity_counts'].items():
                report += f"| {entity_type} | {count:,} |\n"
            report += "\n"
        
        if 'conflict_counts' in entity_summary:
            report += "**Spelling Conflicts (same entity, different spellings):**\n\n"
            report += "| Entity Type | Conflicts |\n"
            report += "|-------------|----------|\n"
            for entity_type, count in entity_summary['conflict_counts'].items():
                report += f"| {entity_type} | {count:,} |\n"
            report += "\n"
    
    # Tourism data
    if tourism_summary:
        report += "### 2.4 Tourism Data\n\n"
        report += f"- **Files processed:** {tourism_summary.get('total_files', 0)}\n"
        report += f"- **Year coverage:** {tourism_summary.get('year_coverage', [])}\n"
        report += f"- **Translation coverage:** {tourism_summary.get('translation_coverage_pct', 0)}%\n"
        report += f"- **Untranslated columns:** {tourism_summary.get('untranslated_column_count', 0)}\n\n"
    
    # Column analysis for small files
    if buildings_profile or projects_profile or valuation_profile:
        report += "### 2.5 Small File Analysis\n\n"
        
        for name, profile in [("Buildings", buildings_profile), 
                               ("Projects", projects_profile), 
                               ("Valuation", valuation_profile)]:
            if profile and 'column_profiles' in profile:
                high_null_cols = [
                    (col, stats['null_pct']) 
                    for col, stats in profile['column_profiles'].items() 
                    if stats.get('null_pct', 0) > 50
                ]
                if high_null_cols:
                    report += f"**{name}.csv - High Null Columns (>50%):**\n"
                    for col, pct in sorted(high_null_cols, key=lambda x: -x[1])[:5]:
                        report += f"- `{col}`: {pct:.1f}%\n"
                    report += "\n"
    
    # Recommendations
    report += """## 3. Recommended Actions

### Priority 1 (Blocking)

1. **Fix date parsing** - Ensure DD-MM-YYYY format is consistently parsed
2. **Create bedroom mapping** - Map all unique bedroom strings to standard labels (Studio, 1BR, 2BR, 3BR, 4BR+)
3. **Resolve entity conflicts** - Review and approve canonical entity names for areas and projects

### Priority 2 (Important)

4. **Complete tourism translations** - Review untranslated Arabic columns
5. **Handle outliers** - Define filtering rules for extreme price/rent values
6. **Document data gaps** - Identify temporal/geographic coverage gaps

### Priority 3 (Enhancement)

7. **Create data dictionary** - Document all column meanings and valid values
8. **Build validation tests** - Automated checks for data quality

---

## 4. Profiling Artifacts

| Artifact | Path | Description |
|----------|------|-------------|
| Transaction Profile | `data_profiles/transactions_profile.json` | Chunked analysis of transactions |
| Rent Contracts Profile | `data_profiles/rent_contracts_profile.json` | RERA-focused analysis |
| Entity Master List | `data_profiles/entity_resolution/master_entity_list.json` | Canonical entity names |
| Entity Conflicts | `data_profiles/entity_resolution/conflicts.json` | Spelling variations |
| Tourism Catalog | `data_profiles/tourism/tourism_catalog.json` | All tourism files profiled |
| Column Translations | `data_profiles/tourism/column_translations.json` | Arabic→English mapping |

---

## 5. Next Steps

- [ ] Review this report
- [ ] Address Priority 1 items
- [ ] Create manual mappings for unparseable values
- [ ] Re-run profiling to verify fixes
- [ ] Update PRD timeline if significant issues found
- [ ] Proceed to Phase 1 (Database Setup)

"""
    
    # Write report
    OUTPUT_FILE.parent.mkdir(exist_ok=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"Report generated: {OUTPUT_FILE}")
    print("\nSummary:")
    print(f"  - Transactions profiled: {'Yes' if transactions else 'No'}")
    print(f"  - Rent Contracts profiled: {'Yes' if rent_contracts else 'No'}")
    print(f"  - Entity resolution done: {'Yes' if entity_summary else 'No'}")
    print(f"  - Tourism data cataloged: {'Yes' if tourism_summary else 'No'}")


if __name__ == "__main__":
    generate_report()

