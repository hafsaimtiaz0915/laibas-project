"""
Bedroom Mapping for Rent_Contracts.csv

Maps ejari_property_sub_type_en values to standardized bedroom categories.
- Residential properties map to bedroom counts (Studio, 1BR, 2BR, etc.)
- Commercial/non-residential properties map to None (for filtering)

Usage:
    from bedroom_mapping import BEDROOM_MAPPING, parse_bedrooms
    df['bedrooms'] = df['ejari_property_sub_type_en'].apply(parse_bedrooms)
    df_residential = df[df['bedrooms'].notna()]
"""

from typing import Optional
import pandas as pd

# Complete mapping of all 86+ property subtypes
BEDROOM_MAPPING = {
    # === RESIDENTIAL WITH BEDROOM COUNT ===
    'Studio': 'Studio',
    '1bed room+Hall': '1BR',
    '2 bed rooms+hall': '2BR',
    '2 bed rooms+hall+Maids Room': '2BR',
    '3 bed rooms+hall': '3BR',
    '3 bed rooms+hall+Maids Room': '3BR',
    '4 bed rooms+hall': '4BR',
    '4 bed rooms+hall+Maids Room': '4BR',
    '5 bed rooms+hall': '5BR',
    '5 bed rooms+hall+Maids Room': '5BR',
    '6 bed rooms+hall': '6BR+',
    '6 bed rooms+hall+Maids Room': '6BR+',
    '7 bed rooms+hall': '6BR+',
    '7 bed rooms+hall+Maids Room': '6BR+',
    '8 bed rooms+hall': '6BR+',
    '9 bed rooms+hall': '6BR+',
    '10 bed rooms+hall': '6BR+',
    '11 bed rooms+hall': '6BR+',
    '15 bed room+hall': '6BR+',
    'Duplex': 'Duplex',
    'Penthouse': 'Penthouse',
    'Room': 'Room',  # Single room (labor housing, shared accommodation)
    'Villa': 'Villa',  # Villa without specific bedroom count
    
    # === COMMERCIAL / NON-RESIDENTIAL (map to None to filter out) ===
    'Office': None,
    'Shop': None,
    'Shop with a mezzanine': None,
    'Showroom': None,
    'Showroom with mezzanine': None,
    'Warehouse': None,
    'Warehouse with a mezzanine': None,
    'Storage': None,
    'Store': None,
    'Store cooled': None,
    'Workshop': None,
    'WorkShop with a mezzanine': None,
    'Factory': None,
    'Labor Camp': None,
    'Room in labor Camp': None,
    'Staff Accommodatoion': None,
    'Staff Accommodation': None,  # Alternate spelling
    'Portacabin Rooms': None,
    'Hotel': None,
    'Hotel apartments': None,
    'hotel building': None,
    'Restaurant': None,
    'Coffe shop': None,
    'Coffee shop': None,  # Alternate spelling
    'Coffe shop,Fast Food': None,
    'Internet cafe , Fast Food': None,
    'Food court': None,
    'Supermarket': None,
    'Grocery': None,
    'Pharmacy': None,
    'Bank': None,
    'ATM': None,
    'Clinic': None,
    'Hospital': None,
    'Nursery': None,
    'School': None,
    'College': None,
    'GYM': None,
    'Gym': None,  # Alternate case
    'Health club': None,
    'Spa': None,
    'swimming pool': None,
    'Swimming Pool': None,  # Alternate case
    'Cinema': None,
    'Hall weddings': None,
    'Laundry': None,
    'Car wash': None,
    'Garage': None,
    'Parking': None,
    'Pertol Station': None,
    'Petrol Station': None,  # Alternate spelling
    'Service Center': None,
    'Kiosk': None,
    'Counter Printing': None,
    'Desk': None,
    'Mezzanine': None,
    'Boardroom': None,
    'Lab': None,
    'laboratory': None,
    'Laboratory': None,  # Alternate case
    'kitchen': None,
    'Kitchen': None,  # Alternate case
    'Machine': None,
    'Sign board': None,
    'Open Land': None,
    'Open space': None,
    'OPEN STORAGE SHED': None,
    'Garden': None,
    'Farm': None,
    'Poultry Farm': None,
    'Horse stable': None,
    'Fish counter': None,
    'Commercial villa': None,
    'Ladies Saloon': None,
    'Hat Saloon': None,
    'church': None,
    'Church': None,  # Alternate case
    'Mosque': None,
    'Other': None,
}

# Numeric mapping for ML models
# ⚠️ NOTE: Only use numeric values where they are FACTUAL.
# For types without clear bedroom count (Duplex, Penthouse, Villa), 
# we keep them as categorical - the MODEL learns their characteristics.
BEDROOM_NUMERIC = {
    'Studio': 0,
    '1BR': 1,
    '2BR': 2,
    '3BR': 3,
    '4BR': 4,
    '5BR': 5,
    '6BR+': 6,
    'Room': 0,  # Single room - factual
    # ❌ REMOVED: These were ASSUMPTIONS, not facts:
    # 'Duplex': 3,    # WHO said duplexes are 3BR? Model should learn this.
    # 'Penthouse': 4, # WHO said penthouses are 4BR? Model should learn this.
    # 'Villa': 4,     # WHO said villas are 4BR? Model should learn this.
    # These should remain as categorical features for the model to interpret.
}


def parse_bedrooms(property_subtype: str) -> Optional[str]:
    """
    Parse bedroom count from ejari_property_sub_type_en.
    
    Args:
        property_subtype: The raw property subtype string
        
    Returns:
        Standardized bedroom category (Studio, 1BR, 2BR, etc.) or None for commercial
    """
    if pd.isna(property_subtype):
        return None
    
    cleaned = str(property_subtype).strip()
    return BEDROOM_MAPPING.get(cleaned, None)


def get_bedroom_numeric(bedroom_category: str) -> Optional[int]:
    """
    Convert bedroom category to numeric value for ML models.
    
    Args:
        bedroom_category: Standardized bedroom category (Studio, 1BR, etc.)
        
    Returns:
        Numeric bedroom count or None
    """
    if pd.isna(bedroom_category):
        return None
    return BEDROOM_NUMERIC.get(bedroom_category, None)


def is_residential(property_subtype: str) -> bool:
    """
    Check if a property subtype is residential.
    
    Args:
        property_subtype: The raw property subtype string
        
    Returns:
        True if residential, False if commercial/other
    """
    return parse_bedrooms(property_subtype) is not None


# Statistics for data quality reporting
def get_mapping_stats() -> dict:
    """Get statistics about the bedroom mapping."""
    residential = [k for k, v in BEDROOM_MAPPING.items() if v is not None]
    commercial = [k for k, v in BEDROOM_MAPPING.items() if v is None]
    
    return {
        "total_mapped": len(BEDROOM_MAPPING),
        "residential_types": len(residential),
        "commercial_types": len(commercial),
        "bedroom_categories": list(set(v for v in BEDROOM_MAPPING.values() if v is not None)),
    }


if __name__ == "__main__":
    # Print mapping statistics
    stats = get_mapping_stats()
    print("Bedroom Mapping Statistics")
    print("=" * 40)
    print(f"Total property types mapped: {stats['total_mapped']}")
    print(f"Residential types: {stats['residential_types']}")
    print(f"Commercial types: {stats['commercial_types']}")
    print(f"Bedroom categories: {stats['bedroom_categories']}")

