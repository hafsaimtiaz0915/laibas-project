"""
Brand-first developer resolution for Dubai real estate data.

Product definition:
- Every row MUST resolve to a public-facing developer brand (Emaar, Meraas, Ellington, Binghatti, Danube, ...).
- Legal developer entities (DLD developer_id, developer_name_ar/en) are secondary evidence + audit fields.
- Never leak legal entity names into the brand output.

Resolution strategy (deterministic):
1) TEXT evidence (primary): match brand aliases against project/building/master project names.
2) LEGAL evidence (secondary): project_number -> Projects.developer_id -> brand_map(developer_id)
3) OVERRIDES (coverage closure): explicit mapping tables:
   - project_number -> brand
   - master_project_en -> brand
If still unresolved: FAIL (caller should write an actionable audit report).
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


def _norm_ws(x: object) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    s = str(x).strip()
    s = re.sub(r"\s+", " ", s)
    return s


def _norm_text(x: object) -> str:
    """
    Text normalization for boundary-safe matching:
    - uppercase
    - replace punctuation with spaces
    - collapse whitespace
    """
    s = _norm_ws(x).upper()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _alias_to_pattern(alias: str) -> re.Pattern:
    """
    Boundary-safe alias regex on normalized UPPERCASE text.

    Example:
      alias = "ELLINGTON" matches " ... ELLINGTON ... "
      but does NOT match "WELLINGTON" or "BELLINGTON".
    """
    a = _norm_text(alias)
    tokens = [re.escape(t) for t in a.split() if t]
    if not tokens:
        return re.compile(r"a^")
    body = r"\s+".join(tokens)
    # Strict boundary: not preceded/followed by A-Z or 0-9 (underscore treated as word char; keep it excluded too)
    pat = rf"(?<![A-Z0-9_]){body}(?![A-Z0-9_])"
    return re.compile(pat)


def _score_alias(alias: str) -> Tuple[int, int, int]:
    """
    Score tuple: (token_count, char_len, -lex) used for tie-breaks.
    Higher is better.
    """
    a = _norm_text(alias)
    tokens = [t for t in a.split() if t]
    return (len(tokens), len(a), 0)


@dataclass(frozen=True)
class BrandMatch:
    brand: str
    source: str  # TEXT_STRONG, TEXT_WEAK, LEGAL_MAP, MASTER_FALLBACK, PROJECT_OVERRIDE
    field: str
    alias: str
    score: int


class BrandResolver:
    def __init__(
        self,
        approved_brands: List[str],
        brand_aliases: Dict[str, List[str]],
        precedence: List[str],
        dev_id_to_brand: Dict[str, str],
        overrides_project_number: Dict[str, str],
        overrides_master_project: Dict[str, str],
    ) -> None:
        self.approved_brands = approved_brands
        self.brand_aliases = brand_aliases
        self.precedence = precedence
        self.dev_id_to_brand = dev_id_to_brand
        self.overrides_project_number = overrides_project_number
        self.overrides_master_project = overrides_master_project

        # Precompile alias patterns per brand, sorted by specificity (tokens, len)
        compiled: Dict[str, List[Tuple[str, re.Pattern, Tuple[int, int, int]]]] = {}
        for b in approved_brands:
            aliases = brand_aliases.get(b) or []
            rows = []
            for a in aliases:
                a2 = _norm_ws(a)
                if not a2:
                    continue
                rows.append((a2, _alias_to_pattern(a2), _score_alias(a2)))
            # sort descending by (token_count, len)
            rows.sort(key=lambda t: (t[2][0], t[2][1]), reverse=True)
            compiled[b] = rows
        self._compiled = compiled

        self._precedence_rank = {b: i for i, b in enumerate(precedence)}

    @staticmethod
    def load_override_csv(path: Path, key_col: str, brand_col: str) -> Dict[str, str]:
        if not path.exists():
            return {}
        out: Dict[str, str] = {}
        with path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                k = _norm_ws(row.get(key_col))
                v = _norm_ws(row.get(brand_col))
                if not k or not v:
                    continue
                out[k] = v
        return out

    @staticmethod
    def build_approved_brands(
        consolidation_json: Path,
        building_devs_json: Path,
    ) -> List[str]:
        brands = []
        if consolidation_json.exists():
            data = json.loads(consolidation_json.read_text(encoding="utf-8"))
            b = data.get("brands")
            if isinstance(b, dict):
                brands.extend(list(b.keys()))
            elif isinstance(data, dict):
                # support legacy/root-level brands dict (hardening)
                brands.extend([k for k in data.keys() if not str(k).startswith("_") and k != "umbrella_structures"])

        if building_devs_json.exists():
            data = json.loads(building_devs_json.read_text(encoding="utf-8"))
            bd = data.get("building_developers_without_own_data") or {}
            if isinstance(bd, dict):
                brands.extend(list(bd.keys()))

        # stable + unique
        out = []
        seen = set()
        for x in brands:
            if x and x not in seen:
                out.append(x)
                seen.add(x)
        return out

    @staticmethod
    def build_brand_aliases(
        consolidation_json: Path,
        building_devs_json: Path,
    ) -> Dict[str, List[str]]:
        """
        Union aliases from both files:
        - consolidation_json: registered_entities (AR) + aliases (EN)
        - building_devs_json: aliases list (EN/AR)
        """
        out: Dict[str, List[str]] = {}

        if consolidation_json.exists():
            data = json.loads(consolidation_json.read_text(encoding="utf-8"))
            brands = data.get("brands")
            if not isinstance(brands, dict):
                # support legacy/root-level
                brands = {k: v for k, v in data.items() if isinstance(v, dict) and not str(k).startswith("_")}
            for brand, info in (brands or {}).items():
                aliases = []
                aliases.extend(info.get("aliases") or [])
                aliases.extend(info.get("registered_entities") or [])
                out.setdefault(brand, [])
                out[brand].extend(aliases)

        if building_devs_json.exists():
            data = json.loads(building_devs_json.read_text(encoding="utf-8"))
            bd = data.get("building_developers_without_own_data") or {}
            for brand, info in (bd or {}).items():
                aliases = info.get("aliases") or []
                out.setdefault(brand, [])
                out[brand].extend(aliases)

        # Deduplicate per brand
        for b, aliases in list(out.items()):
            # Always include the brand label itself as an alias (boundary-safe matching will prevent substrings).
            # This is critical for mapping common cases like:
            #   brand=Seven Tides, developer_name_en="SEVEN TIDES REAL ESTATE DEVELOPMENT L.L.C"
            out[b].append(b)
            seen = set()
            cleaned = []
            for a in aliases:
                a2 = _norm_ws(a)
                if not a2:
                    continue
                key = _norm_text(a2)
                if not key or key in seen:
                    continue
                seen.add(key)
                cleaned.append(a2)
            out[b] = cleaned

        return out

    def _pick_better(
        self,
        cur: Optional[BrandMatch],
        cand: BrandMatch,
    ) -> BrandMatch:
        if cur is None:
            return cand
        # Higher score wins
        if cand.score > cur.score:
            return cand
        if cand.score < cur.score:
            return cur
        # Tie-break by precedence rank (lower rank wins)
        cr = self._precedence_rank.get(cur.brand, 10**9)
        nr = self._precedence_rank.get(cand.brand, 10**9)
        if nr < cr:
            return cand
        return cur

    def resolve_chunk(
        self,
        df: pd.DataFrame,
        projects_by_project_number: pd.DataFrame,
        require_full_coverage: bool = True,
    ) -> Tuple[pd.DataFrame, Dict[str, object]]:
        """
        Resolve developer_brand for a transaction chunk.

        Input df must contain:
          - project_name_en, building_name_en, master_project_en (string evidence; marketing-facing)
          - project_number (optional)

        projects_by_project_number: DataFrame indexed by project_number with columns:
          - developer_id
        """
        out = df.copy()
        n = len(out)
        audit: Dict[str, object] = {
            "rows": int(n),
            "coverage_by_source": {},
            "conflicts": 0,
        }

        # Normalize evidence fields
        for c in ["project_name_en", "building_name_en", "master_project_en"]:
            if c not in out.columns:
                out[c] = ""
            out[c] = out[c].fillna("").astype(str)

        # Create normalized text per field for matching
        text_fields = {
            "project_name_en": out["project_name_en"].map(_norm_text),
            "building_name_en": out["building_name_en"].map(_norm_text),
            "master_project_en": out["master_project_en"].map(_norm_text),
        }
        # Combined text for faster screening
        combined = (
            text_fields["project_name_en"]
            + " "
            + text_fields["building_name_en"]
            + " "
            + text_fields["master_project_en"]
        ).str.strip()

        # Track best match per row
        best: List[Optional[BrandMatch]] = [None] * n
        conflict_count = 0

        # TEXT evidence pass (A)
        for brand in self.approved_brands:
            alias_rows = self._compiled.get(brand) or []
            if not alias_rows:
                continue
            # For efficiency: check combined first
            for alias, pat, score_tuple in alias_rows:
                mask = combined.str.contains(pat, regex=True)
                if not mask.any():
                    continue
                token_cnt, char_len, _ = score_tuple
                score = 1000 * token_cnt + char_len  # big weight on token_count
                # evidence field: deterministic preference order
                field = "combined"
                field_order = ["project_name_en", "building_name_en", "master_project_en"]
                for fname in field_order:
                    series = text_fields.get(fname)
                    if series is None:
                        continue
                    if series[mask].str.contains(pat, regex=True).any():
                        field = fname
                        break
                # strong vs weak heuristic
                source = "TEXT_STRONG" if (token_cnt >= 2 or char_len >= 10) else "TEXT_WEAK"
                cand = BrandMatch(brand=brand, source=source, field=field, alias=alias, score=score)
                idxs = np.where(mask.values)[0]
                for i in idxs:
                    prev = best[i]
                    if prev is not None and prev.brand != cand.brand and prev.score == cand.score:
                        conflict_count += 1
                    best[i] = self._pick_better(prev, cand)

        audit["conflicts"] = int(conflict_count)

        out["developer_brand"] = pd.Series([b.brand if b else "" for b in best], index=out.index, dtype="object")
        out["developer_brand_source"] = pd.Series([b.source if b else "" for b in best], index=out.index, dtype="object")
        out["developer_brand_evidence_field"] = pd.Series([b.field if b else "" for b in best], index=out.index, dtype="object")
        out["developer_brand_evidence_alias"] = pd.Series([b.alias if b else "" for b in best], index=out.index, dtype="object")
        out["developer_brand_evidence_score"] = pd.Series([b.score if b else 0 for b in best], index=out.index, dtype="int64")
        out["developer_brand_evidence"] = pd.Series(
            [
                json.dumps({"field": b.field, "alias": b.alias, "score": b.score}, ensure_ascii=False)
                if b
                else ""
                for b in best
            ],
            index=out.index,
            dtype="object",
        )

        # LEGAL evidence pass (B) only where still blank
        unresolved = out["developer_brand"].astype(str).str.strip() == ""
        if unresolved.any():
            # project_number -> developer_id
            pn = out["project_number"].fillna("").astype(str).str.strip()
            out.loc[unresolved, "_pn"] = pn.loc[unresolved]
            joined = out.loc[unresolved].merge(
                projects_by_project_number[["project_number", "developer_id"]],
                left_on="_pn",
                right_on="project_number",
                how="left",
                suffixes=("", "_p"),
            )
            dev_id = joined["developer_id"].fillna("").astype(str).str.strip()
            # LEGAL_MAP is only allowed when developer_id is explicitly mapped to a curated public brand.
            brand = dev_id.map(self.dev_id_to_brand).fillna("")
            has = brand.astype(str).str.strip() != ""
            # Write back
            out_idx = joined.index
            out.loc[out_idx[has], "developer_brand"] = brand.loc[has].values
            out.loc[out_idx[has], "developer_brand_source"] = "LEGAL_MAP"
            out.loc[out_idx[has], "developer_brand_evidence_field"] = "LEGAL_MAP"
            out.loc[out_idx[has], "developer_brand_evidence_alias"] = ""
            out.loc[out_idx[has], "developer_brand_evidence_score"] = 0
            out.loc[out_idx[has], "developer_brand_evidence"] = [
                json.dumps({"developer_id": d}, ensure_ascii=False) for d in dev_id.loc[has].tolist()
            ]

        # OVERRIDES pass (C) only where still blank
        unresolved = out["developer_brand"].astype(str).str.strip() == ""
        if unresolved.any():
            # project_number override
            pn = out["project_number"].fillna("").astype(str).str.strip()
            brand = pn.map(self.overrides_project_number).fillna("")
            has = unresolved & (brand.astype(str).str.strip() != "")
            if has.any():
                out.loc[has, "developer_brand"] = brand.loc[has]
                out.loc[has, "developer_brand_source"] = "PROJECT_OVERRIDE"
                out.loc[has, "developer_brand_evidence_field"] = "PROJECT_OVERRIDE"
                out.loc[has, "developer_brand_evidence_alias"] = ""
                out.loc[has, "developer_brand_evidence_score"] = 0
                out.loc[has, "developer_brand_evidence"] = [
                    json.dumps({"project_number": p}, ensure_ascii=False) for p in pn.loc[has].tolist()
                ]

        unresolved = out["developer_brand"].astype(str).str.strip() == ""
        if unresolved.any():
            mp = out["master_project_en"].fillna("").astype(str).str.strip()
            brand = mp.map(self.overrides_master_project).fillna("")
            has = unresolved & (brand.astype(str).str.strip() != "")
            if has.any():
                out.loc[has, "developer_brand"] = brand.loc[has]
                out.loc[has, "developer_brand_source"] = "MASTER_FALLBACK"
                out.loc[has, "developer_brand_evidence_field"] = "MASTER_FALLBACK"
                out.loc[has, "developer_brand_evidence_alias"] = ""
                out.loc[has, "developer_brand_evidence_score"] = 0
                out.loc[has, "developer_brand_evidence"] = [
                    json.dumps({"master_project_en": m}, ensure_ascii=False) for m in mp.loc[has].tolist()
                ]

        # Validate coverage / approved brands
        out["developer_brand"] = out["developer_brand"].astype(str).str.strip()
        out["developer_brand_source"] = out["developer_brand_source"].astype(str).str.strip()
        out["developer_brand_evidence"] = out["developer_brand_evidence"].astype(str)
        out["developer_brand_evidence_field"] = out["developer_brand_evidence_field"].astype(str).str.strip()
        out["developer_brand_evidence_alias"] = out["developer_brand_evidence_alias"].astype(str)
        out["developer_brand_evidence_score"] = pd.to_numeric(out["developer_brand_evidence_score"], errors="coerce").fillna(0).astype(int)

        unresolved = out["developer_brand"] == ""
        # Never hard-fail on non-approved brands here. BrandResolver must only emit public brands (approved) or blank.
        # If something slips through (e.g. misconfigured override), blank it and let the caller fill developer_brand_label.
        bad = ~out["developer_brand"].isin(self.approved_brands) & ~unresolved
        if bad.any():
            out.loc[bad, "developer_brand"] = ""
            out.loc[bad, "developer_brand_source"] = ""
            out.loc[bad, "developer_brand_evidence"] = ""
            out.loc[bad, "developer_brand_evidence_field"] = ""
            out.loc[bad, "developer_brand_evidence_alias"] = ""
            out.loc[bad, "developer_brand_evidence_score"] = 0

        # Coverage stats
        vc = out["developer_brand_source"].replace("", "UNRESOLVED").value_counts().to_dict()
        audit["coverage_by_source"] = {k: int(v) for k, v in vc.items()}

        if require_full_coverage and unresolved.any():
            # caller will write audit report; we raise with a small sample
            sample = out.loc[unresolved, ["project_name_en", "building_name_en", "master_project_en", "project_number"]].head(25)
            raise RuntimeError(f"Unresolved developer_brand rows: {int(unresolved.sum())}. Sample:\n{sample.to_string(index=False)}")

        return out, audit


