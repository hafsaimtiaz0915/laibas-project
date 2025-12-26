"""
Trend Lookup Service

Retrieves historical trends and statistics for:
- Developer performance
- Area price trends
- Rent benchmarks

Data formats:
- developer_stats.csv: Uses Arabic developer names
- area_stats.csv: Area names with spaces
- rent_benchmarks.csv: Bedrooms as "1BR", "2BR", "Studio", etc.
"""

import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import pandas as pd
from pydantic import BaseModel
from rapidfuzz import fuzz, process

from ..core.config import get_settings

logger = logging.getLogger(__name__)

# Conversion factor: DLD data is in square meters, display in square feet
# 1 sqm = 10.764 sqft, so AED/sqm ÷ 10.764 = AED/sqft
SQM_TO_SQFT = 10.764


class DeveloperStats(BaseModel):
    """Developer statistics."""
    developer_name: str
    projects_total: int = 0
    projects_completed: int = 0
    projects_active: int = 0
    total_units: int = 0
    avg_completion_percent: float = 0.0
    completion_rate: float = 0.0
    avg_duration_months: float = 0.0
    avg_delay_months: float = 0.0


class AreaStats(BaseModel):
    """Area statistics."""
    area_name: str
    current_median_sqft: Optional[float] = None
    price_change_12m: Optional[float] = None
    price_change_36m: Optional[float] = None
    transaction_count_12m: int = 0
    supply_pipeline: int = 0


class RentBenchmark(BaseModel):
    """Rent benchmark for area + bedroom."""
    area_name: str
    bedrooms: str
    median_annual_rent: Optional[float] = None
    rent_count: int = 0
    median_rent_sqft: Optional[float] = None


class TrendLookupService:
    """
    Service for looking up historical trends and statistics.
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.lookups_path = Path(self.settings.lookups_path)
        
        # Load data
        self.developer_stats: Optional[pd.DataFrame] = None
        self.area_stats: Optional[pd.DataFrame] = None
        self.rent_benchmarks: Optional[pd.DataFrame] = None
        
        # Developer name mapping (English -> Arabic)
        self._english_to_arabic: Dict[str, str] = {}
        
        # Cache for fuzzy matching
        self._developer_names: list = []
        self._area_names: list = []
        
        self._load_data()
    
    def _load_data(self):
        """Load lookup CSV files."""
        # Load developer mapping (English -> Arabic)
        mapping_path = self.lookups_path / "developer_mapping.json"
        if mapping_path.exists():
            try:
                with open(mapping_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                for m in data.get('mappings', []):
                    arabic = m.get('arabic', '')
                    english = m.get('english', '')
                    aliases = m.get('aliases', [])
                    
                    # Map English name to Arabic
                    if english and arabic:
                        self._english_to_arabic[english.lower()] = arabic
                    # Map aliases to Arabic
                    for alias in aliases:
                        if alias:
                            self._english_to_arabic[alias.lower()] = arabic
                logger.info(f"Loaded {len(self._english_to_arabic)} developer name mappings")
            except Exception as e:
                logger.error(f"Error loading developer mapping: {e}")

        # Also load developer_reference.csv (this is populated even when developer_mapping.json is empty)
        # Columns: developer_name (Arabic), english_name, common_aliases
        dev_ref_path = self.lookups_path / "developer_reference.csv"
        if dev_ref_path.exists():
            try:
                ref = pd.read_csv(dev_ref_path, low_memory=False)
                for _, r in ref.iterrows():
                    arabic = str(r.get("developer_name") or "").strip()
                    english = str(r.get("english_name") or "").strip()
                    aliases = str(r.get("common_aliases") or "").strip()
                    if english and arabic:
                        self._english_to_arabic.setdefault(english.lower(), arabic)
                    if aliases:
                        for a in [x.strip() for x in aliases.split(",") if x.strip()]:
                            self._english_to_arabic.setdefault(a.lower(), arabic)
                logger.info(f"Loaded developer_reference.csv mappings (total english->arabic: {len(self._english_to_arabic)})")
            except Exception as e:
                logger.error(f"Error loading developer_reference.csv: {e}")
        
        # Developer stats
        dev_path = self.lookups_path / "developer_stats.csv"
        if dev_path.exists():
            try:
                self.developer_stats = pd.read_csv(dev_path)
                self._developer_names = self.developer_stats['developer_name'].tolist()
                logger.info(f"Loaded {len(self.developer_stats)} developer stats")
            except Exception as e:
                logger.error(f"Error loading developer stats: {e}")
        else:
            logger.warning(f"Developer stats not found: {dev_path}")
        
        # Area stats
        area_path = self.lookups_path / "area_stats.csv"
        if area_path.exists():
            try:
                self.area_stats = pd.read_csv(area_path)
                self._area_names = self.area_stats['area_name'].tolist()
                logger.info(f"Loaded {len(self.area_stats)} area stats")
            except Exception as e:
                logger.error(f"Error loading area stats: {e}")
        else:
            logger.warning(f"Area stats not found: {area_path}")
        
        # Rent benchmarks
        rent_path = self.lookups_path / "rent_benchmarks.csv"
        if rent_path.exists():
            try:
                self.rent_benchmarks = pd.read_csv(rent_path)
                logger.info(f"Loaded {len(self.rent_benchmarks)} rent benchmarks")
            except Exception as e:
                logger.error(f"Error loading rent benchmarks: {e}")
        else:
            logger.warning(f"Rent benchmarks not found: {rent_path}")
    
    def _fuzzy_match_area(self, area_name: str) -> Optional[str]:
        """Fuzzy match area name against known areas."""
        if not self._area_names or not area_name:
            return None
        
        # Try exact match first (case-insensitive)
        for known_area in self._area_names:
            if known_area.lower() == area_name.lower():
                return known_area
        
        # Fuzzy match
        match = process.extractOne(area_name, self._area_names, scorer=fuzz.WRatio)
        if match and match[1] >= 70:
            return match[0]
        
        return None

    def _match_area_with_audit(self, area_name: str) -> Dict[str, Any]:
        """
        Match area name against known areas and return audit metadata.
        """
        audit: Dict[str, Any] = {
            "input": area_name,
            "matched": None,
            "score": 0.0,
            "method": "none",
        }
        if not self._area_names or not area_name:
            return audit

        # Exact match first (case-insensitive)
        for known_area in self._area_names:
            if known_area.lower() == area_name.lower():
                audit.update({"matched": known_area, "score": 100.0, "method": "exact"})
                return audit

        # Fuzzy match
        match = process.extractOne(area_name, self._area_names, scorer=fuzz.WRatio)
        if match:
            audit.update({"matched": match[0], "score": float(match[1]), "method": "fuzzy"})
        return audit
    
    def _fuzzy_match_developer(self, developer_name: str) -> Optional[str]:
        """Fuzzy match developer name against known developers."""
        if not self._developer_names or not developer_name:
            return None
        
        # Skip "Unknown" developer
        if developer_name.lower() == "unknown":
            return None
        
        # Try exact match first
        for known_dev in self._developer_names:
            if known_dev == developer_name:
                return known_dev
        
        # Try partial match (developer name might be substring)
        for known_dev in self._developer_names:
            if developer_name in known_dev or known_dev in developer_name:
                return known_dev
        
        # Fuzzy match
        match = process.extractOne(developer_name, self._developer_names, scorer=fuzz.partial_ratio)
        if match and match[1] >= 60:  # Lower threshold for Arabic names
            return match[0]
        
        return None

    def _match_developer_with_audit(self, developer_name: str) -> Dict[str, Any]:
        """
        Match developer name against known developers and return audit metadata.
        """
        audit: Dict[str, Any] = {
            "input": developer_name,
            "mapped_to_arabic": None,
            "matched": None,
            "score": 0.0,
            "method": "none",
        }
        if not self._developer_names or not developer_name:
            return audit

        if developer_name.lower() == "unknown":
            audit["method"] = "unknown"
            return audit

        # Try to convert English name to Arabic using mapping (record mapping)
        arabic_name = self._english_to_arabic.get(developer_name.lower())
        if arabic_name:
            audit["mapped_to_arabic"] = arabic_name
            developer_name = arabic_name

        # Exact match first (case-sensitive in data)
        for known_dev in self._developer_names:
            if known_dev == developer_name:
                audit.update({"matched": known_dev, "score": 100.0, "method": "exact"})
                return audit

        # Partial substring match
        for known_dev in self._developer_names:
            if developer_name in known_dev or known_dev in developer_name:
                audit.update({"matched": known_dev, "score": 90.0, "method": "substring"})
                return audit

        # Fuzzy match (lower threshold for Arabic)
        match = process.extractOne(developer_name, self._developer_names, scorer=fuzz.partial_ratio)
        if match:
            score = float(match[1])
            # Only treat as a real "match" if the score clears the same bar used for returning stats.
            audit.update({"score": score, "method": "fuzzy"})
            if score >= 60:
                audit["matched"] = match[0]
        return audit

    def get_developer_stats_with_audit(self, developer_name: str) -> Dict[str, Any]:
        """
        Get developer stats and include matching audit.
        """
        audit = self._match_developer_with_audit(developer_name)
        stats = None
        if self.developer_stats is None or not developer_name:
            return {"stats": None, "audit": audit}

        matched_name = audit.get("matched")
        score = float(audit.get("score") or 0.0)
        if not matched_name or score < 60:
            return {"stats": None, "audit": audit}

        match = self.developer_stats[self.developer_stats["developer_name"] == matched_name]
        if match.empty:
            return {"stats": None, "audit": audit}

        row = match.iloc[0]
        stats = DeveloperStats(
            developer_name=row["developer_name"],
            projects_total=int(row.get("projects_total", 0)),
            projects_completed=int(row.get("projects_completed", 0)),
            projects_active=int(row.get("projects_active", 0)),
            total_units=int(row.get("total_units", 0)),
            avg_completion_percent=float(row.get("avg_completion_percent", 0)),
            completion_rate=float(row.get("completion_rate", 0)),
            avg_duration_months=float(row.get("avg_duration_months", 0)),
            avg_delay_months=float(row.get("avg_delay_months", 0)),
        )
        return {"stats": stats, "audit": audit}

    def get_area_stats_with_audit(self, area_name: str) -> Dict[str, Any]:
        """
        Get area stats and include matching audit.
        """
        audit = self._match_area_with_audit(area_name)
        stats = None
        if self.area_stats is None or not area_name:
            return {"stats": None, "audit": audit}

        matched_name = audit.get("matched")
        score = float(audit.get("score") or 0.0)
        if not matched_name or score < 70:
            return {"stats": None, "audit": audit}

        match = self.area_stats[self.area_stats["area_name"] == matched_name]
        if match.empty:
            return {"stats": None, "audit": audit}

        row = match.iloc[0]

        current_median_sqm = float(row["current_median_sqft"]) if pd.notna(row.get("current_median_sqft")) else None
        current_median_sqft = current_median_sqm / SQM_TO_SQFT if current_median_sqm else None

        stats = AreaStats(
            area_name=row["area_name"],
            current_median_sqft=current_median_sqft,
            price_change_12m=float(row["price_change_12m"]) if pd.notna(row.get("price_change_12m")) else None,
            price_change_36m=float(row["price_change_36m"]) if pd.notna(row.get("price_change_36m")) else None,
            transaction_count_12m=int(row.get("transaction_count_12m", 0)),
            supply_pipeline=int(row.get("supply_pipeline", 0)),
        )
        return {"stats": stats, "audit": audit}

    def get_rent_benchmark_with_audit(self, area_name: str, bedroom: str) -> Dict[str, Any]:
        """
        Get rent benchmark and include matching audit (matched area + bedroom).
        """
        audit: Dict[str, Any] = {
            "area_input": area_name,
            "bedroom_input": bedroom,
            "matched_area": None,
            "matched_bedroom": None,
            "area_score": 0.0,
            "area_method": "none",
        }
        if self.rent_benchmarks is None or not area_name or not bedroom:
            return {"stats": None, "audit": audit}

        bedroom_normalized = self._normalize_bedroom_for_lookup(bedroom)
        audit["matched_bedroom"] = bedroom_normalized

        area_audit = self._match_area_with_audit(area_name)
        audit["matched_area"] = area_audit.get("matched")
        audit["area_score"] = float(area_audit.get("score") or 0.0)
        audit["area_method"] = area_audit.get("method")

        if not audit["matched_area"] or audit["area_score"] < 70:
            return {"stats": None, "audit": audit}

        matched_area = str(audit["matched_area"])
        match = self.rent_benchmarks[
            (self.rent_benchmarks["area_name"].str.upper() == matched_area.upper())
            & (self.rent_benchmarks["bedrooms"].str.upper() == str(bedroom_normalized).upper())
        ]
        if match.empty:
            return {"stats": None, "audit": audit}

        row = match.iloc[0]
        median_rent_sqm = float(row["median_rent_sqft"]) if pd.notna(row.get("median_rent_sqft")) else None
        median_rent_sqft = median_rent_sqm / SQM_TO_SQFT if median_rent_sqm else None

        stats = RentBenchmark(
            area_name=row["area_name"],
            bedrooms=str(row["bedrooms"]),
            median_annual_rent=float(row["median_annual_rent"]) if pd.notna(row.get("median_annual_rent")) else None,
            rent_count=int(row.get("rent_count", 0)),
            median_rent_sqft=median_rent_sqft,
        )
        return {"stats": stats, "audit": audit}
    
    def get_developer_stats(self, developer_name: str) -> Optional[DeveloperStats]:
        """
        Get statistics for a developer.
        
        Args:
            developer_name: Developer name (English or Arabic)
            
        Returns:
            DeveloperStats or None if not found
        """
        if self.developer_stats is None or not developer_name:
            return None
        
        if developer_name.lower() == "unknown":
            return None
        
        # Try to convert English name to Arabic using mapping
        arabic_name = self._english_to_arabic.get(developer_name.lower())
        if arabic_name:
            logger.debug(f"Mapped '{developer_name}' -> '{arabic_name}'")
            developer_name = arabic_name
        
        # Find matching developer
        matched_name = self._fuzzy_match_developer(developer_name)
        if not matched_name:
            return None
        
        match = self.developer_stats[
            self.developer_stats['developer_name'] == matched_name
        ]
        
        if match.empty:
            return None
        
        row = match.iloc[0]
        
        return DeveloperStats(
            developer_name=row['developer_name'],
            projects_total=int(row.get('projects_total', 0)),
            projects_completed=int(row.get('projects_completed', 0)),
            projects_active=int(row.get('projects_active', 0)),
            total_units=int(row.get('total_units', 0)),
            avg_completion_percent=float(row.get('avg_completion_percent', 0)),
            completion_rate=float(row.get('completion_rate', 0)),
            avg_duration_months=float(row.get('avg_duration_months', 0)),
            avg_delay_months=float(row.get('avg_delay_months', 0)),
        )
    
    def get_area_stats(self, area_name: str) -> Optional[AreaStats]:
        """
        Get statistics for an area.
        
        Args:
            area_name: Area name (with spaces)
            
        Returns:
            AreaStats or None if not found
        """
        if self.area_stats is None or not area_name:
            return None
        
        # Find matching area
        matched_name = self._fuzzy_match_area(area_name)
        if not matched_name:
            return None
        
        match = self.area_stats[
            self.area_stats['area_name'] == matched_name
        ]
        
        if match.empty:
            return None
        
        row = match.iloc[0]
        
        # Convert price from AED/sqm to AED/sqft
        current_median_sqm = float(row['current_median_sqft']) if pd.notna(row.get('current_median_sqft')) else None
        current_median_sqft = current_median_sqm / SQM_TO_SQFT if current_median_sqm else None
        
        return AreaStats(
            area_name=row['area_name'],
            current_median_sqft=current_median_sqft,
            price_change_12m=float(row['price_change_12m']) if pd.notna(row.get('price_change_12m')) else None,
            price_change_36m=float(row['price_change_36m']) if pd.notna(row.get('price_change_36m')) else None,
            transaction_count_12m=int(row.get('transaction_count_12m', 0)),
            supply_pipeline=int(row.get('supply_pipeline', 0)),
        )
    
    def get_rent_benchmark(self, area_name: str, bedroom: str) -> Optional[RentBenchmark]:
        """
        Get rent benchmark for area + bedroom combination.
        
        Args:
            area_name: Area name (with spaces)
            bedroom: Bedroom count (e.g., "2BR", "Studio")
            
        Returns:
            RentBenchmark or None if not found
        """
        if self.rent_benchmarks is None or not area_name or not bedroom:
            return None
        
        # Normalize bedroom to match CSV format (e.g., "2BR", "Studio", "Penthouse")
        bedroom_normalized = self._normalize_bedroom_for_lookup(bedroom)
        
        # Try to match area name
        matched_area = self._fuzzy_match_area(area_name)
        
        if matched_area:
            # Try exact match on area + bedroom
            match = self.rent_benchmarks[
                (self.rent_benchmarks['area_name'].str.upper() == matched_area.upper()) &
                (self.rent_benchmarks['bedrooms'].str.upper() == bedroom_normalized.upper())
            ]
            
            if match.empty:
                # Try case-insensitive partial area match
                match = self.rent_benchmarks[
                    (self.rent_benchmarks['area_name'].str.upper().str.contains(matched_area.upper(), regex=False)) &
                    (self.rent_benchmarks['bedrooms'].str.upper() == bedroom_normalized.upper())
                ]
        else:
            # Direct search by area name contains
            match = self.rent_benchmarks[
                (self.rent_benchmarks['area_name'].str.upper().str.contains(area_name.upper(), regex=False)) &
                (self.rent_benchmarks['bedrooms'].str.upper() == bedroom_normalized.upper())
            ]
        
        if match.empty:
            return None
        
        row = match.iloc[0]
        
        # Convert rent per sqm to rent per sqft
        median_rent_sqm = float(row['median_rent_sqft']) if pd.notna(row.get('median_rent_sqft')) else None
        median_rent_sqft = median_rent_sqm / SQM_TO_SQFT if median_rent_sqm else None
        
        return RentBenchmark(
            area_name=row['area_name'],
            bedrooms=str(row['bedrooms']),
            median_annual_rent=float(row['median_annual_rent']) if pd.notna(row.get('median_annual_rent')) else None,
            rent_count=int(row.get('rent_count', 0)),
            median_rent_sqft=median_rent_sqft,
        )
    
    def _normalize_bedroom_for_lookup(self, bedroom: str) -> str:
        """
        Normalize bedroom string for lookup in rent_benchmarks.csv.
        
        The CSV has values like: "1BR", "2BR", "Studio", "Penthouse", "6BR+"
        """
        if not bedroom:
            return "2BR"  # Default
        
        bedroom_upper = bedroom.upper().strip()
        
        # Already in correct format
        if bedroom_upper in ["STUDIO", "PENTHOUSE", "ROOM", "1BR", "2BR", "3BR", "4BR", "5BR", "6BR+"]:
            return bedroom
        
        # Handle variations
        mapping = {
            "0BR": "Studio",
            "0 BR": "Studio",
            "STUDIO": "Studio",
            "PH": "Penthouse",
            "1BR": "1BR",
            "1 BR": "1BR",
            "2BR": "2BR",
            "2 BR": "2BR",
            "3BR": "3BR",
            "3 BR": "3BR",
            "4BR": "4BR",
            "4 BR": "4BR",
            "5BR": "5BR",
            "5 BR": "5BR",
            "6BR": "6BR+",
            "6BR+": "6BR+",
        }
        
        if bedroom_upper in mapping:
            return mapping[bedroom_upper]
        
        # Extract number if present
        import re
        num_match = re.search(r'(\d+)', bedroom)
        if num_match:
            num = int(num_match.group(1))
            if num == 0:
                return "Studio"
            elif num >= 6:
                return "6BR+"
            else:
                return f"{num}BR"
        
        return bedroom  # Return as-is
    
    def get_all_trends(
        self,
        developer_arabic: Optional[str],
        area_name: Optional[str],
        bedroom: Optional[str]
    ) -> Dict[str, Any]:
        """
        Get all relevant trends for a property query.
        
        Returns:
            Dict with developer_stats, area_stats, rent_benchmark
        """
        result: Dict[str, Any] = {
            "developer_stats": None,
            "area_stats": None,
            "rent_benchmark": None,
            "lookup_audit": {
                "developer": None,
                "area": None,
                "rent": None,
            },
        }
        
        try:
            if developer_arabic:
                dev_res = self.get_developer_stats_with_audit(developer_arabic)
                result["developer_stats"] = dev_res.get("stats")
                result["lookup_audit"]["developer"] = dev_res.get("audit")
        except Exception as e:
            logger.error(f"Error getting developer stats: {e}")
        
        try:
            if area_name:
                area_res = self.get_area_stats_with_audit(area_name)
                result["area_stats"] = area_res.get("stats")
                result["lookup_audit"]["area"] = area_res.get("audit")
                
                if bedroom:
                    rent_res = self.get_rent_benchmark_with_audit(area_name, bedroom)
                    result["rent_benchmark"] = rent_res.get("stats")
                    result["lookup_audit"]["rent"] = rent_res.get("audit")
        except Exception as e:
            logger.error(f"Error getting area/rent stats: {e}")
        
        return result


# Singleton instance
_service: Optional[TrendLookupService] = None


def get_trend_service() -> TrendLookupService:
    """Get singleton TrendLookupService instance."""
    global _service
    if _service is None:
        _service = TrendLookupService()
    return _service


async def get_trends(
    developer_arabic: Optional[str] = None,
    area_name: Optional[str] = None,
    bedroom: Optional[str] = None
) -> Dict[str, Any]:
    """
    Convenience function to get all trends.
    """
    service = get_trend_service()
    return service.get_all_trends(developer_arabic, area_name, bedroom)
