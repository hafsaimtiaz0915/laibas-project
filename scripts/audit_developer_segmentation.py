#!/usr/bin/env python3
"""
Audit developer brand segmentation using RAW DLD extracts.

Goal:
- Prove that major brands (Binghatti, Danube, Ellington, Damac) are NOT silently
  lost under master/registered developers in the training data.
- Surface gaps: alias-hit areas not covered by rules, ambiguous registered-under distributions,
  and missing consolidations.

Outputs:
- Docs/Data_audits/DEVELOPER_SEGMENTATION_AUDIT.md
- Docs/Data_audits/DEVELOPER_SEGMENTATION_AUDIT.csv
"""

from __future__ import annotations

from pathlib import Path
import json
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "Data"
DOCS_OUT = ROOT / "Docs" / "Data_audits"
DOCS_OUT.mkdir(parents=True, exist_ok=True)


RAW_TRANSACTIONS = DATA / "Transactions.csv"
RAW_PROJECTS = DATA / "Projects.csv"

BUILDING_DEVS = DATA / "lookups" / "building_developers.json"
CONSOLIDATION = DATA / "lookups" / "developer_brand_consolidation.json"
TRAINING = DATA / "tft" / "tft_training_data.csv"


BRANDS = {
    "Binghatti": ["binghatti", "بن غاطي", "بنغاتي"],
    "Danube": ["danube", "دانوب"],
    "Ellington": ["ellington", "إلينجتون"],
    # Note: 'Damac' appears frequently in building/project names (e.g., hotel/brand names),
    # so alias hits are NOT a reliable proxy for developer identity. We audit DAMAC via
    # registered-entity consolidation instead.
    "Damac": ["damac", "داماك"],
}

CONSOLIDATION_BRANDS = ["Damac", "Emaar"]

def _md_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    """Render a small markdown table without requiring tabulate."""
    if df is None:
        return ""
    if max_rows is not None and len(df) > max_rows:
        df = df.head(max_rows).copy()
    if len(df.columns) == 0:
        return ""
    df = df.fillna("")
    cols = list(df.columns)
    header = "| " + " | ".join(cols) + " |\n"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |\n"
    lines = []
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return header + sep + "\n".join(lines) + ("\n" if lines else "")


def _norm_mp(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.strip()
    return s.where(~s.str.lower().isin(["nan", "none", ""]), other=pd.NA)


def load_master_dev_maps() -> tuple[dict[float, str], dict[str, str]]:
    """Return maps: project_number -> master_developer_name, master_project_en -> master_developer_name."""
    proj = pd.read_csv(
        RAW_PROJECTS,
        usecols=["project_number", "master_project_en", "master_developer_name"],
        low_memory=False,
    )
    proj["project_number"] = pd.to_numeric(proj["project_number"], errors="coerce")
    proj["master_project_en"] = proj["master_project_en"].astype(str).str.strip()
    proj["master_developer_name"] = proj["master_developer_name"].astype(str).str.strip()

    pnum = (
        proj.dropna(subset=["project_number", "master_developer_name"])
        .drop_duplicates("project_number")
        .set_index("project_number")["master_developer_name"]
        .to_dict()
    )
    mp = (
        proj.dropna(subset=["master_project_en", "master_developer_name"])
        .groupby(["master_project_en", "master_developer_name"])
        .size()
        .reset_index(name="n")
        .sort_values(["master_project_en", "n"], ascending=[True, False])
        .drop_duplicates("master_project_en")
        .set_index("master_project_en")["master_developer_name"]
        .to_dict()
    )
    return pnum, mp


def load_rules() -> dict:
    with open(BUILDING_DEVS, "r", encoding="utf-8") as f:
        bd = json.load(f)
    with open(CONSOLIDATION, "r", encoding="utf-8") as f:
        cons = json.load(f)
    return {"building_devs": bd, "consolidation": cons}


def audit_raw_alias_hits(pnum_map: dict, mp_map: dict) -> pd.DataFrame:
    hdr = pd.read_csv(RAW_TRANSACTIONS, nrows=0)
    use = [c for c in ["area_name_en", "project_number", "master_project_en", "project_name_en", "building_name_en"] if c in hdr.columns]
    rows = []

    for chunk in pd.read_csv(RAW_TRANSACTIONS, usecols=use, chunksize=250_000, low_memory=False):
        area = chunk["area_name_en"].astype(str).str.strip()
        pn = pd.to_numeric(chunk["project_number"], errors="coerce")
        mp = _norm_mp(chunk["master_project_en"])

        dev_by_pn = pn.map(pnum_map)
        dev_by_mp = mp.map(mp_map)
        reg = dev_by_pn.fillna(dev_by_mp).fillna("UNRESOLVED")

        txt = (
            chunk["project_name_en"].astype(str).str.lower().fillna("")
            + " "
            + chunk["building_name_en"].astype(str).str.lower().fillna("")
        )

        for brand, toks in BRANDS.items():
            mask = pd.Series(False, index=chunk.index)
            for t in toks:
                mask = mask | txt.str.contains(t, regex=False)
            if mask.any():
                rows.append(
                    pd.DataFrame(
                        {
                            "brand": brand,
                            "area_name_en": area[mask].values,
                            "registered_under": reg[mask].values,
                        }
                    )
                )

    if not rows:
        return pd.DataFrame(columns=["brand", "area_name_en", "registered_under"])
    return pd.concat(rows, ignore_index=True)


def audit_training_presence() -> pd.DataFrame:
    df = pd.read_csv(
        TRAINING,
        usecols=["developer_name", "developer_brand", "developer_registered_name", "group_id"],
        low_memory=False,
    )
    out = []
    for brand in list(BRANDS.keys()) + [b for b in CONSOLIDATION_BRANDS if b not in BRANDS]:
        sub = df[df["developer_name"] == brand]
        out.append(
            {
                "brand": brand,
                "training_rows": len(sub),
                "training_groups": sub["group_id"].nunique(),
            }
        )
    return pd.DataFrame(out)

def audit_damac_consolidation_leakage() -> pd.DataFrame:
    """
    Check if any DAMAC-registered entities still appear as developer_name (they should consolidate to 'Damac').
    """
    df = pd.read_csv(
        TRAINING,
        usecols=["developer_name", "developer_registered_name"],
        low_memory=False,
    )
    damac_reg = df["developer_registered_name"].astype(str).str.contains("داماك", na=False) | df["developer_registered_name"].astype(str).str.contains("damac", case=False, na=False)
    leaked = df[damac_reg & (df["developer_name"] != "Damac")]
    out = {
        "damac_registered_rows": int(damac_reg.sum()),
        "damac_leaked_rows": int(len(leaked)),
        "damac_leaked_developer_names": ", ".join(sorted(set(leaked["developer_name"].astype(str).unique()))[:20]),
    }
    return pd.DataFrame([out])

def audit_emaar_consolidation_leakage() -> pd.DataFrame:
    """
    Check if Emaar-registered entities are consolidated to developer_name='Emaar'.
    Note: a small number of rows may be overridden by building-developer brand logic
    (e.g., Binghatti projects registered under an Emaar entity) - those are reported separately.
    """
    df = pd.read_csv(
        TRAINING,
        usecols=["developer_name", "developer_registered_name"],
        low_memory=False,
    )
    emaar_reg = (
        df["developer_registered_name"].astype(str).str.contains("اعمار", na=False)
        | df["developer_registered_name"].astype(str).str.contains("إعمار", na=False)
    )
    overridden = df[emaar_reg & (df["developer_name"] != "Emaar")]
    out = {
        "emaar_registered_rows": int(emaar_reg.sum()),
        "emaar_non_emaar_developer_name_rows": int(len(overridden)),
        "non_emaar_developer_name_top": ", ".join(overridden["developer_name"].value_counts().head(5).index.astype(str).tolist()),
    }
    return pd.DataFrame([out])


def main() -> None:
    pnum_map, mp_map = load_master_dev_maps()
    rules = load_rules()

    raw = audit_raw_alias_hits(pnum_map, mp_map)
    train = audit_training_presence()
    damac_leak = audit_damac_consolidation_leakage()
    emaar_leak = audit_emaar_consolidation_leakage()

    # Per-brand per-area dominance
    grp = raw.groupby(["brand", "area_name_en", "registered_under"]).size().reset_index(name="n")
    tot = raw.groupby(["brand", "area_name_en"]).size().reset_index(name="tot")
    top = (
        grp.sort_values(["brand", "area_name_en", "n"], ascending=[True, True, False])
        .groupby(["brand", "area_name_en"])
        .head(1)
        .merge(tot, on=["brand", "area_name_en"], how="left")
    )
    top["share"] = top["n"] / top["tot"]

    # Areas where the rules explicitly allow multiple registered-under values (do not flag as ambiguity)
    multi_allowed = set()
    bd = rules["building_devs"].get("building_developers_without_own_data", {}) or {}
    for brand, info in bd.items():
        ra = (info or {}).get("registered_under_by_area") or {}
        for area_name_en, expected in ra.items():
            if isinstance(expected, list):
                multi_allowed.add((brand, area_name_en))

    # Identify rule gaps (areas present in raw alias hits but absent from mapping rules)
    gaps = []
    for brand in ["Binghatti", "Danube", "Ellington"]:
        areas_in_raw = set(raw.loc[raw.brand == brand, "area_name_en"].unique())
        mapped = set((bd.get(brand, {}).get("registered_under_by_area") or {}).keys())
        missing = sorted(a for a in areas_in_raw if a not in mapped)
        for a in missing:
            gaps.append({"brand": brand, "missing_area_rule": a})
    gaps_df = pd.DataFrame(gaps)

    # Damac consolidation gap: Damac token hits in raw but training developer_name != Damac
    # (should be zero if consolidation + naming catches all)
    # We can only check training presence: if training_rows > 0, consolidation exists; else missing.

    # Write CSV summary (top dominance rows)
    csv_path = DOCS_OUT / "DEVELOPER_SEGMENTATION_AUDIT.csv"
    top.sort_values(["brand", "tot"], ascending=[True, False]).to_csv(csv_path, index=False)

    md_path = DOCS_OUT / "DEVELOPER_SEGMENTATION_AUDIT.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Developer Segmentation Audit (RAW → Training)\n\n")
        f.write("This audit compares **RAW Transactions brand signals** (project/building names) to **training data developer series**.\n\n")

        f.write("## Training series presence (are brands first-class?)\n\n")
        f.write(_md_table(train))
        f.write("\n\n")

        f.write("## DAMAC consolidation leakage check (registered entities → brand)\n\n")
        f.write(_md_table(damac_leak))
        f.write("\n\n")

        f.write("## EMAAR consolidation leakage check (registered entities → brand)\n\n")
        f.write(_md_table(emaar_leak))
        f.write("\n\n")

        f.write("## RAW alias-hit volume (how many transactions mention the brand?)\n\n")
        raw_counts = raw.groupby("brand").size().reset_index(name="raw_alias_hit_rows").sort_values("raw_alias_hit_rows", ascending=False)
        f.write(_md_table(raw_counts))
        f.write("\n\n")

        f.write("## Dominant registered-under per area (RAW)\n\n")
        f.write("Columns: `tot`=alias-hit rows in area, `registered_under`=dominant master developer, `share`=dominance share.\n\n")
        f.write(_md_table(top.sort_values(["brand", "tot"], ascending=[True, False]), max_rows=200))
        f.write("\n\n")

        f.write("## Gaps: areas with RAW alias hits but no `registered_under_by_area` rule\n\n")
        if len(gaps_df) == 0:
            f.write("✓ No gaps found for Binghatti/Danube/Ellington.\n\n")
        else:
            f.write(_md_table(gaps_df))
            f.write("\n\n")

        f.write("## Ambiguity flags (needs manual review)\n\n")
        # Only flag ambiguity for building-developer brands where alias mapping is intended to represent developer identity.
        amb = top[
            (top["brand"].isin(["Binghatti", "Danube", "Ellington"]))
            & (top["share"] < 0.85)
            & (top["tot"] >= 200)
            & (~top.apply(lambda r: (r["brand"], r["area_name_en"]) in multi_allowed, axis=1))
        ].sort_values(["brand", "tot"], ascending=[True, False])
        if len(amb) == 0:
            f.write("✓ No high-volume ambiguous areas found under current thresholds.\n")
        else:
            f.write("Areas where alias-hit rows are split across multiple registered-under developers (dominance share < 0.85):\n\n")
            f.write(_md_table(amb))
            f.write("\n")

    print("Wrote:", md_path)
    print("Wrote:", csv_path)


if __name__ == "__main__":
    main()


