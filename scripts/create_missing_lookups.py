"""Create missing lookup files used by the pipeline:
- Data/lookups/umbrella_map.json
- Data/lookups/public_brands.json
- Data/lookups/blocked_brand_labels.json
- Data/lookups/top50_developers_2025.json

This script derives sensible defaults from existing lookup artifacts when possible.
"""
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).parent.parent
LOOKUPS = ROOT / "Data" / "lookups"
LOOKUPS.mkdir(parents=True, exist_ok=True)

def load_json(p):
    try:
        return json.loads(Path(p).read_text(encoding='utf-8'))
    except Exception:
        return None

def main():
    # 1) public_brands.json: use keys from developer_brand_consolidation.json if available
    dbc = load_json(LOOKUPS / 'developer_brand_consolidation.json')
    public_brands = []
    if dbc and 'brands' in dbc:
        public_brands = sorted(list(dbc['brands'].keys()))
    else:
        # fallback: try developer_stats.csv english column
        ref = LOOKUPS / 'developer_reference.csv'
        if ref.exists():
            df = pd.read_csv(ref, low_memory=False)
            public_brands = sorted(df['english_name'].dropna().unique().tolist())

    (LOOKUPS / 'public_brands.json').write_text(json.dumps(public_brands, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Wrote public_brands.json ({len(public_brands)} entries)')

    # 2) top50_developers_2025.json: pick top developers by total_units from developer_stats.csv
    top50 = []
    ds_path = LOOKUPS / 'developer_stats.csv'
    ref_path = LOOKUPS / 'developer_reference.csv'
    if ds_path.exists() and ref_path.exists():
        ds = pd.read_csv(ds_path, low_memory=False)
        ref = pd.read_csv(ref_path, low_memory=False)
        # developer_stats.csv has developer_name (arabic) and total_units
        if 'developer_name' in ds.columns and 'total_units' in ds.columns:
            merged = ds.merge(ref[['developer_name','english_name']], on='developer_name', how='left')
            merged['total_units'] = pd.to_numeric(merged['total_units'], errors='coerce').fillna(0)
            merged = merged.sort_values('total_units', ascending=False)
            top50 = merged['english_name'].dropna().astype(str).tolist()[:50]
    # fallback: use public_brands first 50
    if not top50:
        top50 = public_brands[:50]

    (LOOKUPS / 'top50_developers_2025.json').write_text(json.dumps(top50, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Wrote top50_developers_2025.json ({len(top50)} entries)')

    # 3) blocked_brand_labels.json: default to empty list
    blocked = []
    (LOOKUPS / 'blocked_brand_labels.json').write_text(json.dumps(blocked, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Wrote blocked_brand_labels.json (empty list)')

    # 4) umbrella_map.json: conservative mapping brand -> umbrella (self) for now
    umbrella = {b: b for b in public_brands}
    (LOOKUPS / 'umbrella_map.json').write_text(json.dumps(umbrella, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Wrote umbrella_map.json ({len(umbrella)} entries)')

if __name__ == '__main__':
    main()
