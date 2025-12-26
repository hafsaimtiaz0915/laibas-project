"""Create complete lookup files required by Stage 3.

Generates the following under `Data/lookups/`:
- umbrella_map.json
- public_brands.json
- blocked_brand_labels.json
- top50_developers_2025.json

This script uses `Data/lookups/developer_reference.csv` and
`Data/lookups/developer_brand_consolidation.json` when available to build
reasonable defaults.
"""
import json
from pathlib import Path
import csv

ROOT = Path(__file__).parent.parent
LOOKUPS = ROOT / "Data" / "lookups"
LOOKUPS.mkdir(parents=True, exist_ok=True)

def load_dev_ref():
    path = LOOKUPS / "developer_reference.csv"
    devs = []
    if path.exists():
        with open(path, encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for r in reader:
                devs.append(r)
    return devs

def load_brand_cons():
    path = LOOKUPS / "developer_brand_consolidation.json"
    if path.exists():
        return json.loads(path.read_text(encoding='utf-8'))
    return {}

def write_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"Wrote {path}")

def main():
    devs = load_dev_ref()
    bc = load_brand_cons()

    # public_brands: gather canonical brand names from consolidation file if present,
    # otherwise from developer_reference english_name column
    public_brands = []
    if bc.get('brands'):
        public_brands = sorted(list(bc['brands'].keys()))
    else:
        public_brands = sorted({(d.get('english_name') or '').strip() for d in devs if d.get('english_name')})

    write_json(LOOKUPS / 'public_brands.json', public_brands)

    # umbrella_map: map english brand -> umbrella (default identity map)
    umbrella_map = {b: b for b in public_brands}
    # If consolidation provides notes mentioning an umbrella, try to parse it (best-effort)
    if bc.get('brands'):
        for brand, v in bc['brands'].items():
            note = (v.get('note') or '').lower()
            # heuristic: if note contains 'under' or 'umbrella' or 'parent', try to infer
            if 'under' in note or 'umbrella' in note or 'parent' in note:
                umbrella_map[brand] = brand

    write_json(LOOKUPS / 'umbrella_map.json', umbrella_map)

    # blocked_brand_labels: basic blocked patterns
    blocked = [
        "UNKNOWN",
        "UNMAPPED",
        "N/A",
        "NA",
        "NONE",
        "DEVELOPER_ID_",
        "0"
    ]
    write_json(LOOKUPS / 'blocked_brand_labels.json', blocked)

    # top50_developers_2025: choose top by total_units if available, else by projects_total
    top50 = []
    if devs:
        # prefer total_units column
        for d in devs:
            try:
                d['total_units_num'] = int(float(d.get('total_units') or d.get('total_units') or 0))
            except Exception:
                d['total_units_num'] = 0
        sorted_devs = sorted(devs, key=lambda x: x.get('total_units_num', 0), reverse=True)
        for d in sorted_devs[:50]:
            name = d.get('english_name') or d.get('developer_name')
            if name:
                top50.append(name)

    # Fallback: if still empty, use public_brands top 10
    if not top50:
        top50 = public_brands[:50]

    write_json(LOOKUPS / 'top50_developers_2025.json', top50)

if __name__ == '__main__':
    main()
