#!/usr/bin/env python3
"""Generate area mapping coverage report (CSV + MD).

Outputs:
- Data/lookups/area_mapping_coverage.csv
- Docs/frontend/LLM/AREA_MAPPING_COVERAGE.md

Report includes:
- ambiguous_aliases from Data/lookups/area_mapping.json
- ambiguous marketing names / aliases inferred from Data/lookups/area_reference.csv

Usage:
  python scripts/generate_area_mapping_coverage.py
"""

from __future__ import annotations

import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).parent.parent
LOOKUPS = ROOT / "Data" / "lookups"
DOCS = ROOT / "Docs" / "frontend" / "LLM"

AREA_MAP_PATH = LOOKUPS / "area_mapping.json"
AREA_REF_PATH = LOOKUPS / "area_reference.csv"

OUT_CSV = LOOKUPS / "area_mapping_coverage.csv"
OUT_MD = DOCS / "AREA_MAPPING_COVERAGE.md"


def _clean(x) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and pd.isna(x):
        return ""
    s = str(x).strip()
    if not s or s.lower() == "nan":
        return ""
    return s


def main() -> None:
    if not AREA_MAP_PATH.exists():
        raise SystemExit(f"Missing {AREA_MAP_PATH}")
    if not AREA_REF_PATH.exists():
        raise SystemExit(f"Missing {AREA_REF_PATH}")

    area_map = json.loads(AREA_MAP_PATH.read_text(encoding="utf-8"))
    amb = area_map.get("ambiguous_aliases") or {}

    ref = pd.read_csv(AREA_REF_PATH, low_memory=False)

    # Normalize columns
    ref["master_project_en"] = ref["master_project_en"].apply(_clean)
    ref["area_name_en"] = ref["area_name_en"].apply(_clean)
    ref["common_aliases"] = ref.get("common_aliases", "").apply(_clean)
    ref["total_units"] = pd.to_numeric(ref.get("total_units"), errors="coerce").fillna(0)
    ref = ref[(ref["master_project_en"] != "") & (ref["area_name_en"] != "")]

    rows = []

    # 1) From area_mapping.json ambiguous_aliases
    for k, info in amb.items():
        rows.append({
            "source": "area_mapping.json",
            "key": k,
            "key_type": "alias_or_marketing",
            "dominant_dld_area": _clean((info or {}).get("dominant")),
            "candidates": " | ".join((info or {}).get("candidates") or []),
            "reason": "maps_to_multiple_dld_areas",
        })

    # 2) Infer ambiguity from reference table: master_project_en that maps to >1 area
    mp_grp = ref.groupby("master_project_en")
    for mp, g in mp_grp:
        areas = sorted(set(g["area_name_en"].tolist()))
        if len(areas) > 1:
            dominant = (
                g.groupby("area_name_en")["total_units"].sum().sort_values(ascending=False).index[0]
                if len(g) else ""
            )
            rows.append({
                "source": "area_reference.csv",
                "key": mp,
                "key_type": "master_project_en",
                "dominant_dld_area": _clean(dominant),
                "candidates": " | ".join(areas),
                "reason": "master_project_en_maps_to_multiple_dld_areas",
            })

    # 3) Infer ambiguity from reference aliases: alias token maps to >1 area
    alias_rows = []
    for _, r in ref.iterrows():
        dld = r["area_name_en"]
        tu = float(r.get("total_units") or 0)
        aliases = r["common_aliases"]
        if not aliases:
            continue
        for a in [x.strip() for x in aliases.split(",") if x.strip()]:
            a = _clean(a)
            if a:
                alias_rows.append((a, dld, tu))

    if alias_rows:
        adf = pd.DataFrame(alias_rows, columns=["alias", "area_name_en", "total_units"]) 
        for alias, g in adf.groupby("alias"):
            areas = sorted(set(g["area_name_en"].tolist()))
            if len(areas) > 1:
                dominant = g.groupby("area_name_en")["total_units"].sum().sort_values(ascending=False).index[0]
                rows.append({
                    "source": "area_reference.csv",
                    "key": alias,
                    "key_type": "common_alias",
                    "dominant_dld_area": _clean(dominant),
                    "candidates": " | ".join(areas),
                    "reason": "alias_maps_to_multiple_dld_areas",
                })

    out = pd.DataFrame(rows).drop_duplicates(subset=["source", "key", "reason"]).sort_values(["reason", "key"])
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)

    # MD summary
    DOCS.mkdir(parents=True, exist_ok=True)
    total_amb = len(out)
    from_map = int((out["source"] == "area_mapping.json").sum())
    from_ref = total_amb - from_map

    md = []
    md.append("## Area mapping coverage report\n")
    md.append("### Summary\n")
    md.append(f"- Total ambiguous names/aliases: **{total_amb}**\n")
    md.append(f"- From `area_mapping.json:ambiguous_aliases`: **{from_map}**\n")
    md.append(f"- Inferred from `area_reference.csv` (multi-mapped): **{from_ref}**\n")
    md.append("\n### What this means\n")
    md.append("- Any key listed here maps to **multiple DLD areas**. We intentionally resolve it with lower confidence (or require disambiguation) to avoid confidently wrong outputs.\n")
    md.append("\n### Needs manual disambiguation list\n")
    if total_amb == 0:
        md.append("- None\n")
    else:
        for k in out["key"].unique().tolist():
            md.append(f"- {k}\n")

    md.append("\n### Details (top 50)\n")
    if total_amb:
        show = out.head(50)
        md.append("| key | dominant_dld_area | candidates | source | reason |\n")
        md.append("|---|---|---|---|---|\n")
        for _, r in show.iterrows():
            md.append(
                f"| {r['key']} | {r['dominant_dld_area']} | {r['candidates']} | {r['source']} | {r['reason']} |\n"
            )

    OUT_MD.write_text("".join(md), encoding="utf-8")

    print(f"Wrote: {OUT_CSV}")
    print(f"Wrote: {OUT_MD}")


if __name__ == "__main__":
    main()




