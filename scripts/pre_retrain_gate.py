#!/usr/bin/env python3
"""
Pre-retrain gate (one-command) for investor-grade training data.

Usage:
  python scripts/pre_retrain_gate.py

What it does:
  1) Runs developer segmentation audit (brands / consolidation).
  2) Runs hard QA gates on Data/tft/tft_training_data.csv.
  3) Prints a PASS/FAIL checklist and writes a Markdown + JSON report.
  4) Exits non-zero if any HARD gate fails.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TRAINING = ROOT / "Data" / "tft" / "tft_training_data.csv"
AUDIT_SCRIPT = ROOT / "scripts" / "audit_developer_segmentation.py"
AUDIT_MD = ROOT / "Docs" / "Data_audits" / "DEVELOPER_SEGMENTATION_AUDIT.md"
OUT_MD = ROOT / "Docs" / "Data_audits" / "PRE_RETRAIN_GATE_REPORT.md"
OUT_JSON = ROOT / "Docs" / "Data_audits" / "PRE_RETRAIN_GATE_REPORT.json"


@dataclass
class GateResult:
    name: str
    passed: bool
    severity: str  # HARD / SOFT / INFO
    details: str = ""


def _run(cmd: List[str]) -> Tuple[int, str]:
    p = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    out = (p.stdout or "") + (("\n" + p.stderr) if p.stderr else "")
    return p.returncode, out.strip()


def _md_table(rows: List[GateResult]) -> str:
    header = "| Gate | Severity | Result | Details |\n"
    sep = "| --- | --- | --- | --- |\n"
    lines = []
    for r in rows:
        res = "PASS" if r.passed else "FAIL"
        det = r.details.replace("\n", " ").strip()
        lines.append(f"| {r.name} | {r.severity} | {res} | {det} |")
    return header + sep + "\n".join(lines) + "\n"


def main() -> int:
    results: List[GateResult] = []
    meta: Dict[str, Any] = {}

    # 1) Developer segmentation audit
    if AUDIT_SCRIPT.exists():
        code, out = _run(["python3", str(AUDIT_SCRIPT)])
        results.append(
            GateResult(
                name="Developer segmentation audit script runs",
                severity="HARD",
                passed=(code == 0),
                details="ok" if code == 0 else out[:500],
            )
        )
        results.append(
            GateResult(
                name="Developer segmentation audit output exists",
                severity="HARD",
                passed=AUDIT_MD.exists(),
                details=str(AUDIT_MD) if AUDIT_MD.exists() else "missing audit MD",
            )
        )
    else:
        results.append(GateResult("Developer segmentation audit script present", False, "HARD", "missing scripts/audit_developer_segmentation.py"))

    # 2) Training data presence
    results.append(
        GateResult(
            name="Training data exists",
            severity="HARD",
            passed=TRAINING.exists() and TRAINING.stat().st_size > 0,
            details=str(TRAINING),
        )
    )
    if not (TRAINING.exists() and TRAINING.stat().st_size > 0):
        _write_reports(results, meta)
        return 2

    df = pd.read_csv(TRAINING, low_memory=False)
    meta["rows"] = int(len(df))
    meta["groups"] = int(df["group_id"].nunique()) if "group_id" in df.columns else None
    meta["areas"] = int(df["area_name"].nunique()) if "area_name" in df.columns else None
    meta["date_range"] = f"{df['year_month'].min()} -> {df['year_month'].max()}" if "year_month" in df.columns else None

    # Required columns (HARD)
    required_cols = [
        "time_idx", "year_month", "group_id",
        "area_name", "property_type", "bedroom",
        "reg_type", "reg_type_dld",
        "developer_name",
        "median_price", "transaction_count",
        "units_completing",
        "months_to_handover_signed", "months_since_handover",
        "dld_offplan_after_handover",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    results.append(GateResult("Required columns present", len(missing) == 0, "HARD", f"missing: {missing}" if missing else "ok"))

    # Developer identity gates
    unknown_rows = int((df["developer_name"].astype(str).str.lower() == "unknown").sum())
    results.append(GateResult("No Unknown developers", unknown_rows == 0, "HARD", f"Unknown rows={unknown_rows}"))
    unmapped_rows = int(df["developer_name"].astype(str).str.startswith("UNMAPPED_DEVELOPER__").sum())
    results.append(GateResult("No UNMAPPED_DEVELOPER__ rows", unmapped_rows == 0, "HARD", f"unmapped rows={unmapped_rows}"))

    all_dev_rows = int((df["developer_name"] == "ALL_DEVELOPERS").sum())
    all_dev_pct = all_dev_rows / max(len(df), 1) * 100.0
    results.append(GateResult("ALL_DEVELOPERS bucket small (<5%)", all_dev_pct < 5.0, "SOFT", f"{all_dev_rows} rows ({all_dev_pct:.2f}%)"))

    # Lifecycle gates
    rt = df["reg_type"].value_counts().to_dict()
    results.append(GateResult("Lifecycle reg_type has both OffPlan and Ready", ("OffPlan" in rt and "Ready" in rt), "HARD", str(rt)))

    # Handover labeling: DLD OffPlan after handover is allowed but must be visible
    flips = int((df["dld_offplan_after_handover"] > 0).sum()) if "dld_offplan_after_handover" in df.columns else 0
    results.append(GateResult("DLD OffPlan after handover is tracked", flips >= 0, "INFO", f"flagged series-rows={flips} ({(flips/max(len(df),1))*100:.2f}%)"))

    # Price/rent sanity (HARD)
    price_low = int((df["median_price"] < 100).sum())
    price_high = int((df["median_price"] > 50_000).sum())
    results.append(GateResult("median_price within [100, 50k] AED/sqft", (price_low == 0 and price_high == 0), "HARD", f"low={price_low}, high={price_high}"))

    if "median_rent" in df.columns:
        rent_neg = int((df["median_rent"] < 0).sum())
        rent_hi = int((df["median_rent"] > 2_000_000).sum())
        results.append(GateResult("median_rent non-negative", rent_neg == 0, "HARD", f"neg={rent_neg}"))
        results.append(GateResult("median_rent not absurd (>2M)", rent_hi == 0, "SOFT", f">2M rows={rent_hi}"))

    # Supply schedule presence
    units_nonzero = int((df["units_completing"] > 0).sum())
    units_pct = units_nonzero / max(len(df), 1) * 100.0
    results.append(GateResult("units_completing has signal (non-zero rows)", units_nonzero > 0, "SOFT", f"nonzero={units_nonzero} ({units_pct:.2f}%)"))

    # Group history (SOFT) - investor confidence improves with history
    if "group_id" in df.columns and "time_idx" in df.columns:
        glen = df.groupby("group_id")["time_idx"].count()
        lt12 = int((glen < 12).sum())
        lt24 = int((glen < 24).sum())
        results.append(GateResult("Group history <12 months (monitor)", True, "INFO", f"{lt12}/{len(glen)} groups ({lt12/max(len(glen),1)*100:.2f}%)"))
        results.append(GateResult("Group history <24 months (monitor)", True, "INFO", f"{lt24}/{len(glen)} groups ({lt24/max(len(glen),1)*100:.2f}%)"))

    # Write reports
    _write_reports(results, meta)

    hard_failed = [r for r in results if r.severity == "HARD" and not r.passed]
    return 1 if hard_failed else 0


def _write_reports(results: List[GateResult], meta: Dict[str, Any]) -> None:
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    md = []
    md.append("# Pre-Retrain Gate Report\n")
    if meta:
        md.append("## Dataset snapshot\n")
        for k, v in meta.items():
            md.append(f"- **{k}**: {v}")
        md.append("")
    md.append("## Gate results\n")
    md.append(_md_table(results))
    md.append("\n## Notes\n- HARD failures must be fixed before retraining.\n- SOFT items are acceptable but should be monitored.\n")
    OUT_MD.write_text("\n".join(md).strip() + "\n", encoding="utf-8")

    OUT_JSON.write_text(
        json.dumps(
            {
                "meta": meta,
                "results": [
                    {"name": r.name, "severity": r.severity, "passed": r.passed, "details": r.details}
                    for r in results
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())






