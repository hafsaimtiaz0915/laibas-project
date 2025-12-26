"""
Entity Validator Service

Validates and resolves entity names:
- Developer: English → Arabic resolution (TFT uses Arabic names)
- Area: Abbreviation expansion and fuzzy matching
- Bedroom: Normalization to standard format

IMPORTANT: Group ID format must match build_tft_data.py exactly:
    {area}_{property_type}_{bedroom}_{reg_type}_{developer}
    
    With these transformations:
    - area: spaces → underscores, remove apostrophes
    - developer: spaces → underscores, remove apostrophes/commas/periods
"""

import json
import logging
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from pydantic import BaseModel
from rapidfuzz import fuzz, process

from ..core.config import get_settings

logger = logging.getLogger(__name__)


class ValidatedEntities(BaseModel):
    """Validated and resolved entity names."""
    developer_english: Optional[str] = None
    developer_arabic: Optional[str] = None
    developer_confidence: float = 0.0
    developer_resolution_method: Optional[str] = None  # exact|alias|fuzzy|unknown|building_dev
    area_name: Optional[str] = None  # DLD area name (for model)
    area_name_clean: Optional[str] = None  # Underscore format for group_id
    area_display_name: Optional[str] = None  # User-friendly area name (e.g., "JVC" instead of "Al Barsha South Fourth")
    area_confidence: float = 0.0
    area_resolution_method: Optional[str] = None  # abbrev|exact|fuzzy|fallback
    bedroom: Optional[str] = None
    property_type: str = "Unit"
    reg_type: Optional[str] = None
    group_id: Optional[str] = None  # Constructed group_id for TFT
    
    # Building developer tracking (for developers without own data)
    is_building_developer: bool = False  # True if developer doesn't have own data in training
    building_developer_name: Optional[str] = None  # User-facing developer name (e.g., "Binghatti")
    developer_data_caveat: Optional[str] = None  # Caveat message for user


class EntityValidator:
    """Validates and resolves property entities."""
    
    def __init__(self):
        self.settings = get_settings()
        self.lookups_path = Path(self.settings.lookups_path)
        
        # Load mappings
        self.developer_mapping: List[Dict] = []
        self.area_mapping: Dict = {}
        self.all_areas: List[str] = []
        self.dld_to_common: Dict[str, str] = {}  # DLD name → user-friendly name
        
        # Building developers (without own data in training)
        self.building_developers: Dict = {}
        self.building_developer_aliases: Dict[str, str] = {}  # alias → developer name
        
        self._load_mappings()
    
    def _load_mappings(self):
        """Load mapping files from lookups directory."""
        # Load developer mapping (English → Arabic)
        dev_path = self.lookups_path / "developer_mapping.json"
        if dev_path.exists():
            try:
                with open(dev_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.developer_mapping = data.get("mappings", [])
                logger.info(f"Loaded {len(self.developer_mapping)} developer mappings")
            except Exception as e:
                logger.error(f"Error loading developer mapping: {e}")
        else:
            logger.warning(f"Developer mapping not found: {dev_path}")
        
        # Load area mapping (abbreviations + all areas + dld_to_common)
        area_path = self.lookups_path / "area_mapping.json"
        if area_path.exists():
            try:
                with open(area_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.area_mapping = data.get("abbreviations", {})
                    self.all_areas = data.get("all_areas", [])
                    self.dld_to_common = data.get("dld_to_common", {})
                    self.ambiguous_area_aliases = data.get("ambiguous_aliases", {})
                logger.info(f"Loaded {len(self.all_areas)} areas, {len(self.area_mapping)} abbreviations, {len(self.dld_to_common)} display mappings")
            except Exception as e:
                logger.error(f"Error loading area mapping: {e}")
        else:
            logger.warning(f"Area mapping not found: {area_path}")
        
        # Load building developers (without own data in training)
        building_dev_path = self.lookups_path / "building_developers.json"
        if building_dev_path.exists():
            try:
                with open(building_dev_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.building_developers = data.get("building_developers_without_own_data", {})
                    # Build alias lookup
                    for dev_name, dev_info in self.building_developers.items():
                        for alias in dev_info.get("aliases", []):
                            self.building_developer_aliases[alias.lower()] = dev_name
                logger.info(f"Loaded {len(self.building_developers)} building developers without own data")
            except Exception as e:
                logger.error(f"Error loading building developers: {e}")
        else:
            logger.warning(f"Building developers file not found: {building_dev_path}")
    
    def resolve_developer(self, english_name: str) -> Tuple[Optional[str], Optional[str], float]:
        """
        Resolve English developer name to Arabic.
        
        TFT model uses Arabic developer names in group_id.
        
        Returns:
            Tuple of (canonical_english_name, arabic_name, confidence)
        """
        if not english_name or not self.developer_mapping:
            return None, None, 0.0
        
        english_lower = english_name.lower().strip()
        
        # 1. Try exact match on English name or aliases
        for mapping in self.developer_mapping:
            if mapping.get("english", "").lower() == english_lower:
                return mapping["english"], mapping["arabic"], 100.0
            
            for alias in mapping.get("aliases", []):
                if alias.lower() == english_lower:
                    return mapping["english"], mapping["arabic"], 100.0
        
        # 2. Try fuzzy match on English names and aliases
        all_names = []
        name_to_mapping = {}
        
        for mapping in self.developer_mapping:
            english = mapping.get("english")
            if english:
                all_names.append(english)
                name_to_mapping[english.lower()] = mapping
            
            for alias in mapping.get("aliases", []):
                all_names.append(alias)
                name_to_mapping[alias.lower()] = mapping
        
        if not all_names:
            return None, None, 0.0
        
        # Fuzzy match with WRatio (weighted ratio - handles partial matches well)
        match = process.extractOne(english_name, all_names, scorer=fuzz.WRatio)
        if match and match[1] >= 70:  # 70% threshold
            matched_name = match[0].lower()
            mapping = name_to_mapping.get(matched_name)
            if mapping:
                return mapping["english"], mapping["arabic"], match[1]
        
        # 3. Return "Unknown" for unrecognized developers
        return english_name, "Unknown", 50.0

    def resolve_developer_with_method(self, english_name: str) -> Tuple[Optional[str], Optional[str], float, str]:
        """
        Resolve developer and return (canonical_english, arabic, confidence, method).
        """
        if not english_name or not self.developer_mapping:
            return None, None, 0.0, "none"

        english_lower = english_name.lower().strip()

        # 1. Try exact match on English name or aliases
        for mapping in self.developer_mapping:
            if mapping.get("english", "").lower() == english_lower:
                return mapping["english"], mapping["arabic"], 100.0, "exact"
            for alias in mapping.get("aliases", []):
                if alias.lower() == english_lower:
                    return mapping["english"], mapping["arabic"], 100.0, "alias"

        # 2. Fuzzy match on English names and aliases
        all_names = []
        name_to_mapping = {}
        for mapping in self.developer_mapping:
            english = mapping.get("english")
            if english:
                all_names.append(english)
                name_to_mapping[english.lower()] = mapping
            for alias in mapping.get("aliases", []):
                all_names.append(alias)
                name_to_mapping[alias.lower()] = mapping

        if all_names:
            match = process.extractOne(english_name, all_names, scorer=fuzz.WRatio)
            if match and match[1] >= 70:
                matched_name = match[0].lower()
                mapping = name_to_mapping.get(matched_name)
                if mapping:
                    return mapping["english"], mapping["arabic"], float(match[1]), "fuzzy"

        return english_name, "Unknown", 50.0, "unknown"
    
    def check_building_developer(self, developer_name: str, area_name: Optional[str] = None) -> Tuple[bool, Optional[str], Optional[str], Optional[str]]:
        """
        Check if a developer is a "building developer" without own data in training.
        
        These developers (like Binghatti, Danube, Azizi) have their projects registered
        under master developers in DLD, so their data is aggregated with other developers.
        
        Args:
            developer_name: The developer name to check
            area_name: The area (used to determine which master developer to use for multi-area developers)
            
        Returns:
            Tuple of (is_building_developer, building_developer_name, registered_under_arabic, caveat_message)
        """
        if not developer_name:
            return False, None, None, None
        
        dev_lower = developer_name.lower().strip()
        
        # Check if this is a known building developer (by alias)
        matched_dev_name = self.building_developer_aliases.get(dev_lower)
        
        # Also check against the developer keys directly
        if not matched_dev_name:
            for dev_name in self.building_developers.keys():
                if dev_name.lower() == dev_lower:
                    matched_dev_name = dev_name
                    break
        
        if not matched_dev_name:
            return False, None, None, None
        
        dev_info = self.building_developers.get(matched_dev_name, {})
        
        # Determine which master developer they're registered under
        registered_under = dev_info.get("registered_under")
        
        # Some developers have different master developers by area
        if not registered_under and area_name:
            by_area = dev_info.get("registered_under_by_area", {})
            registered_under = by_area.get(area_name)
        
        # Build caveat message
        area_display = self.dld_to_common.get(area_name, area_name) if area_name else "the area"
        caveat = f"Developer-specific prediction data is not available for {matched_dev_name}. The prediction is based on {area_display} area market data, which includes all developers in that location."
        
        return True, matched_dev_name, registered_under, caveat
    
    def get_area_display_name(self, dld_area_name: str) -> str:
        """Get user-friendly display name for a DLD area name."""
        return self.dld_to_common.get(dld_area_name, dld_area_name)
    
    def resolve_area(self, area_input: str) -> Tuple[Optional[str], float]:
        """
        Resolve area name, expanding abbreviations and fuzzy matching.
        
        Returns:
            Tuple of (area_name_with_spaces, confidence)
        """
        if not area_input:
            return None, 0.0
        
        area_clean = area_input.strip()
        
        # 1. Check abbreviations (case-insensitive)
        for key, value in self.area_mapping.items():
            if key.lower() == area_clean.lower():
                return value, 100.0
        
        # 2. Check exact match in all_areas (case-insensitive)
        for area in self.all_areas:
            if area.lower() == area_clean.lower():
                return area, 100.0
        
        # 3. Fuzzy match against all_areas
        if self.all_areas:
            match = process.extractOne(area_input, self.all_areas, scorer=fuzz.WRatio)
            if match and match[1] >= 70:  # 70% threshold
                return match[0], match[1]
        
        # 4. Return original input (let TFT handle unknown areas via fuzzy matching)
        return area_input, 50.0

    def resolve_area_with_method(self, area_input: str) -> Tuple[Optional[str], float, str]:
        """
        Resolve area name and return (resolved_dld_area, confidence, method).
        """
        if not area_input:
            return None, 0.0, "none"

        area_clean = area_input.strip()

        # 1. Abbreviations (case-insensitive)
        for key, value in self.area_mapping.items():
            if key.lower() == area_clean.lower():
                return value, 100.0, "abbrev"

        # 1b. Ambiguous alias (do not return 100% confidence)
        # Example: keys like "Dubai Water Canal" may map to multiple DLD areas.
        if hasattr(self, "ambiguous_area_aliases") and self.ambiguous_area_aliases:
            for key, info in self.ambiguous_area_aliases.items():
                if key.lower() == area_clean.lower():
                    dominant = (info or {}).get("dominant")
                    if dominant:
                        return dominant, 60.0, "ambiguous"

        # 1c. Fuzzy match against abbreviation keys (marketing names + known aliases)
        # This catches cases like parser returning "Jumeirah Lake Towers" vs mapping key "Jumeirah Lakes Towers".
        if self.area_mapping:
            try:
                keys = list(self.area_mapping.keys())
                match = process.extractOne(area_input, keys, scorer=fuzz.WRatio)
                if match and match[1] >= 85:
                    matched_key = match[0]
                    # If the matched key is ambiguous, return dominant with reduced confidence
                    if hasattr(self, "ambiguous_area_aliases") and self.ambiguous_area_aliases:
                        amb = self.ambiguous_area_aliases.get(matched_key)
                        if amb and amb.get("dominant"):
                            return amb["dominant"], 60.0, "ambiguous"
                    return self.area_mapping.get(matched_key), float(match[1]), "abbrev_fuzzy"
            except Exception:
                pass

        # 2. Exact match in all_areas (case-insensitive)
        for area in self.all_areas:
            if area.lower() == area_clean.lower():
                return area, 100.0, "exact"

        # 3. Fuzzy match against all_areas
        if self.all_areas:
            match = process.extractOne(area_input, self.all_areas, scorer=fuzz.WRatio)
            if match and match[1] >= 70:
                return match[0], float(match[1]), "fuzzy"

        # 4. Fallback
        return area_input, 50.0, "fallback"
    
    def normalize_bedroom(self, bedroom_input: str) -> Optional[str]:
        """
        Normalize bedroom string to standard TFT format.
        
        Valid values: Studio, 1BR, 2BR, 3BR, 4BR, 5BR, 6BR+, Penthouse, Room
        """
        if not bedroom_input:
            return None
        
        bedroom_lower = bedroom_input.lower().strip()
        
        # Direct mappings
        mapping = {
            "studio": "Studio",
            "0": "Studio",
            "0br": "Studio",
            "0 bed": "Studio",
            "1": "1BR",
            "1br": "1BR",
            "1 bed": "1BR",
            "1 bedroom": "1BR",
            "one bed": "1BR",
            "2": "2BR",
            "2br": "2BR",
            "2 bed": "2BR",
            "2 bedroom": "2BR",
            "two bed": "2BR",
            "3": "3BR",
            "3br": "3BR",
            "3 bed": "3BR",
            "3 bedroom": "3BR",
            "three bed": "3BR",
            "4": "4BR",
            "4br": "4BR",
            "4 bed": "4BR",
            "4 bedroom": "4BR",
            "four bed": "4BR",
            "5": "5BR",
            "5br": "5BR",
            "5 bed": "5BR",
            "5 bedroom": "5BR",
            "five bed": "5BR",
            "6": "6BR+",
            "6br": "6BR+",
            "6br+": "6BR+",
            "6 bed": "6BR+",
            "7": "6BR+",
            "7br": "6BR+",
            "penthouse": "Penthouse",
            "ph": "Penthouse",
            "room": "Room",
            "single room": "Room",
        }
        
        # Remove common suffixes for matching
        cleaned = bedroom_lower.replace("bedrooms", "").replace("bedroom", "").strip()
        
        # Try exact match
        if bedroom_lower in mapping:
            return mapping[bedroom_lower]
        
        if cleaned in mapping:
            return mapping[cleaned]
        
        # Try to extract number
        import re
        num_match = re.search(r'(\d+)', bedroom_input)
        if num_match:
            num = int(num_match.group(1))
            if num == 0:
                return "Studio"
            elif num >= 6:
                return "6BR+"
            else:
                return f"{num}BR"
        
        # Check for keywords
        if "studio" in bedroom_lower:
            return "Studio"
        if "penthouse" in bedroom_lower or "ph" in bedroom_lower:
            return "Penthouse"
        
        # Return None if can't normalize
        return None
    
    def construct_group_id(
        self,
        area: str,
        property_type: str,
        bedroom: str,
        reg_type: str,
        developer_arabic: str
    ) -> str:
        """
        Construct group_id matching exact format from build_tft_data.py.
        
        Format: {area}_{property_type}_{bedroom}_{reg_type}_{developer}
        
        Transformations (must match build_tft_data.py lines 919-925):
        - area: replace spaces with _, remove apostrophes
        - property_type: replace spaces with _
        - bedroom: replace spaces with _
        - reg_type: as-is
        - developer: replace spaces with _, remove apostrophes/commas/periods
        """
        area_clean = area.replace(' ', '_').replace("'", "")
        prop_clean = property_type.replace(' ', '_')
        bed_clean = bedroom.replace(' ', '_')
        dev_clean = developer_arabic.replace(' ', '_').replace("'", "").replace(",", "").replace(".", "")
        
        return f"{area_clean}_{prop_clean}_{bed_clean}_{reg_type}_{dev_clean}"
    
    def validate(
        self,
        developer: Optional[str],
        area: Optional[str],
        bedroom: Optional[str],
        property_type: Optional[str],
        reg_type: Optional[str]
    ) -> ValidatedEntities:
        """
        Validate and resolve all entities.
        
        Args:
            developer: English developer name (will be resolved to Arabic for TFT)
            area: Area name (may be abbreviation like "JVC")
            bedroom: Bedroom count (e.g., "2BR", "2 bed", "two bedroom")
            property_type: "Unit" or "Villa"
            reg_type: "OffPlan" or "Ready"
            
        Returns:
            ValidatedEntities with resolved names, confidence scores, and group_id
        """
        result = ValidatedEntities()
        
        # Resolve area first (needed for building developer check)
        if area:
            resolved_area, conf, method = self.resolve_area_with_method(area)
            result.area_name = resolved_area
            result.area_name_clean = resolved_area.replace(' ', '_').replace("'", "") if resolved_area else None
            result.area_confidence = conf
            result.area_resolution_method = method
            # Set user-friendly display name
            result.area_display_name = self.get_area_display_name(resolved_area) if resolved_area else None
        
        # Check if developer is a "building developer" without own data
        if developer:
            is_building_dev, building_dev_name, registered_under, caveat = self.check_building_developer(
                developer, result.area_name
            )
            
            if is_building_dev:
                # This developer doesn't have own data - use "Unknown" for model
                # but preserve the user-facing name
                result.is_building_developer = True
                result.building_developer_name = building_dev_name
                result.developer_data_caveat = caveat
                result.developer_english = building_dev_name
                result.developer_arabic = "Unknown"  # Use Unknown for model lookup (area-based prediction)
                result.developer_confidence = 100.0  # We recognized the developer
                result.developer_resolution_method = "building_dev"
            else:
                # Normal developer resolution
                eng, arabic, conf, method = self.resolve_developer_with_method(developer)
                result.developer_english = eng
                result.developer_arabic = arabic or "Unknown"
                result.developer_confidence = conf
                result.developer_resolution_method = method
        else:
            result.developer_arabic = "Unknown"
            result.developer_confidence = 0.0
            result.developer_resolution_method = "none"
        
        # Normalize bedroom
        if bedroom:
            result.bedroom = self.normalize_bedroom(bedroom)
        
        if not result.bedroom:
            result.bedroom = "2BR"  # Default to 2BR if not specified
        
        # Validate property type
        if property_type in ["Unit", "Villa"]:
            result.property_type = property_type
        elif property_type:
            prop_lower = property_type.lower()
            if "villa" in prop_lower or "house" in prop_lower or "townhouse" in prop_lower:
                result.property_type = "Villa"
            else:
                result.property_type = "Unit"
        else:
            result.property_type = "Unit"  # Default
        
        # Validate reg_type
        if reg_type in ["OffPlan", "Ready"]:
            result.reg_type = reg_type
        elif reg_type:
            reg_lower = reg_type.lower()
            if "off" in reg_lower or "plan" in reg_lower or "new" in reg_lower:
                result.reg_type = "OffPlan"
            elif "ready" in reg_lower or "complet" in reg_lower or "resale" in reg_lower:
                result.reg_type = "Ready"
        
        if not result.reg_type:
            result.reg_type = "OffPlan"  # Default for off-plan analysis tool
        
        # Construct group_id for TFT lookup
        if result.area_name and result.bedroom:
            result.group_id = self.construct_group_id(
                area=result.area_name,
                property_type=result.property_type,
                bedroom=result.bedroom,
                reg_type=result.reg_type,
                developer_arabic=result.developer_arabic
            )
        
        return result


# Singleton instance
_validator: Optional[EntityValidator] = None


def get_entity_validator() -> EntityValidator:
    """Get singleton EntityValidator instance."""
    global _validator
    if _validator is None:
        _validator = EntityValidator()
    return _validator


async def validate_entities(
    developer: Optional[str] = None,
    area: Optional[str] = None,
    bedroom: Optional[str] = None,
    property_type: Optional[str] = None,
    reg_type: Optional[str] = None
) -> ValidatedEntities:
    """
    Convenience function to validate entities.
    """
    validator = get_entity_validator()
    return validator.validate(developer, area, bedroom, property_type, reg_type)
