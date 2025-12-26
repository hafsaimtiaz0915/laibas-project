#!/usr/bin/env python3
"""Validate area_mapping.json coverage and correctness.

Checks:
- Every `master_project_en` in Data/lookups/area_reference.csv exists as a key in area_mapping.json abbreviations and maps to the same DLD area.
- Every alias in `common_aliases` exists and maps to the same DLD area.
- All abbreviation targets exist in `all_areas`.
- Reports collisions and missing keys.

Usage:
  python scripts/validate_area_mapping.py
"""

import json
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).parent.parent
LOOKUPS = ROOT / "Data" / "lookups"

AREA_MAP_PATH = LOOKUPS / "area_mapping.json"
AREA_REF_PATH = LOOKUPS / "area_reference.csv"


def main():
    if not AREA_MAP_PATH.exists():
        raise SystemExit(f"Missing {AREA_MAP_PATH}")
    if not AREA_REF_PATH.exists():
        raise SystemExit(f"Missing {AREA_REF_PATH}")

    area_map = json.loads(AREA_MAP_PATH.read_text(encoding="utf-8"))
    abbrev = area_map.get("abbreviations") or {}
    ambiguous = area_map.get("ambiguous_aliases") or {}
    all_areas = set(area_map.get("all_areas") or [])

    ref = pd.read_csv(AREA_REF_PATH, low_memory=False)

    missing = []
    wrong = []
    invalid_targets = sorted({v for v in abbrev.values() if v not in all_areas})

    for _, row in ref.iterrows():
        mp = str(row.get("master_project_en") or "").strip()
        dld = str(row.get("area_name_en") or "").strip()
        if not mp or not dld:
            continue

        if mp not in abbrev:
            # Accept if intentionally ambiguous
            amb = ambiguous.get(mp)
            if not amb or dld not in (amb.get("candidates") or []):
                missing.append(("master_project_en", mp, dld))
        elif abbrev.get(mp) != dld:
            wrong.append(("master_project_en", mp, dld, abbrev.get(mp)))

        aliases_val = row.get("common_aliases")
        if aliases_val is None or (isinstance(aliases_val, float) and pd.isna(aliases_val)):
            aliases = ""
        else:
            aliases = str(aliases_val).strip()
        if aliases and aliases.lower() != "nan":
            for a in [x.strip() for x in aliases.split(",") if x.strip() and x.strip().lower() != "nan"]:
                if a not in abbrev:
                    amb = ambiguous.get(a)
                    if not amb or dld not in (amb.get("candidates") or []):
                        missing.append(("alias", a, dld))
                elif abbrev.get(a) != dld:
                    # If alias is ambiguous and dld is a candidate, do not treat as wrong
                    amb = ambiguous.get(a)
                    if not amb or dld not in (amb.get("candidates") or []):
                        wrong.append(("alias", a, dld, abbrev.get(a)))

    print("=== Area mapping validation ===")
    print(f"abbreviations keys: {len(abbrev):,}")
    print(f"all_areas: {len(all_areas):,}")
    print(f"invalid targets (not in all_areas): {len(invalid_targets):,}")
    if invalid_targets:
        print("  example invalid targets:", invalid_targets[:10])

    print(f"missing keys: {len(missing):,}")
    if missing:
        print("  first 20 missing:")
        for m in missing[:20]:
            print("   ", m)

    print(f"wrong mappings: {len(wrong):,}")
    if wrong:
        print("  first 20 wrong:")
        for w in wrong[:20]:
            print("   ", w)

    # Non-zero exit if problems
    if invalid_targets or wrong:
        raise SystemExit(1)


if __name__ == "__main__":
    main()




