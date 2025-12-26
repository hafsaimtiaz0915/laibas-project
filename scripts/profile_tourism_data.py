#!/usr/bin/env python3
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
                (len(self.all_columns) - len(untranslated)) / max(len(self.all_columns), 1) * 100, 1
            ),
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

