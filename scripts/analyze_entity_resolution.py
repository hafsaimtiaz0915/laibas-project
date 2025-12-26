#!/usr/bin/env python3
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
from tqdm import tqdm

# Try to import fuzzy matching library
try:
    from rapidfuzz import fuzz
    FUZZY_AVAILABLE = True
except ImportError:
    from difflib import SequenceMatcher
    FUZZY_AVAILABLE = False
    print("Note: rapidfuzz not installed, using slower difflib")


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
    
    def similarity(self, s1: str, s2: str) -> float:
        """Calculate similarity between two strings."""
        if FUZZY_AVAILABLE:
            return fuzz.ratio(s1, s2) / 100.0
        else:
            return SequenceMatcher(None, s1, s2).ratio()
    
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
                    if abs(len(key1) - len(key2)) / max(len(key1), len(key2), 1) > 0.3:
                        continue
                    
                    ratio = self.similarity(key1, key2)
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
            try:
                for chunk in pd.read_csv(filepath, chunksize=100_000, low_memory=False):
                    analyzer.extract_entities_from_df(chunk, filename, entity_columns)
                    chunk_count += 1
                    if chunk_count >= 10:  # Limit chunks for entity analysis
                        break
                print(f"    Processed {chunk_count} chunks")
            except Exception as e:
                print(f"    Error: {e}")
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

