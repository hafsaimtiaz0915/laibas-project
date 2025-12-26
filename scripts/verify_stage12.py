"""Verify Stage 1 (cleaning) and Stage 2 (lookup generation).

Produces a human-readable markdown report at
`Docs/Data_docs/STAGE1_2_VERIFICATION.md` describing test results
and a final PASS/FAIL conclusion.
"""
import json
from pathlib import Path

ROOT = Path(__file__).parent.parent
cleaned_dir = ROOT / "Data" / "cleaned"
lookups_dir = ROOT / "Data" / "lookups"
report_path = ROOT / "Docs" / "Data_docs" / "STAGE1_2_VERIFICATION.md"

tests = []

def load_json(p):
    try:
        return json.loads(p.read_text(encoding='utf-8'))
    except Exception:
        return None

def run():
    report_lines = ["# Stage 1 & 2 Verification Report\n"]

    cs_path = cleaned_dir / "cleaning_stats.json"
    if not cs_path.exists():
        report_lines.append("- ERROR: cleaning_stats.json not found\n")
        write(report_lines)
        return

    cs = load_json(cs_path)
    report_lines.append("## Cleaning stats summary\n")
    report_lines.append(f"- Path: {cs_path}\n")
    report_lines.append(f"- Keys present: {', '.join(sorted(cs.keys()))}\n")

    # Transactions checks
    tx = cs.get('transactions', {})
    tx_total = tx.get('total_rows')
    tx_valid = tx.get('valid_rows')
    try:
        ratio = float(tx_valid) / float(tx_total)
    except Exception:
        ratio = 0.0

    report_lines.append("\n### Transactions checks\n")
    report_lines.append(f"- Total rows: {tx_total}\n")
    report_lines.append(f"- Valid rows: {tx_valid}\n")
    report_lines.append(f"- Valid/Total ratio: {ratio:.2%}\n")

    tests.append(('transactions_valid_ratio', ratio >= 0.70))

    # Rent contracts checks
    rent = cs.get('rent_contracts', {})
    res_rows = rent.get('residential_rows', 0)
    total_rent = rent.get('total_rows', 0)
    rent_frac = float(res_rows) / float(total_rent) if total_rent else 0.0
    report_lines.append("\n### Rent contracts checks\n")
    report_lines.append(f"- Total rent rows: {total_rent}\n")
    report_lines.append(f"- Residential rows: {res_rows}\n")
    report_lines.append(f"- Residential fraction: {rent_frac:.2%}\n")
    tests.append(('rent_residential_fraction', 0.2 <= rent_frac <= 0.9))

    # Duplicates and missing price/area
    dup = tx.get('duplicates_removed', 0)
    miss = tx.get('missing_price_or_area_removed', 0)
    report_lines.append("\n### Data loss checks\n")
    report_lines.append(f"- Duplicates removed (transactions): {dup}\n")
    report_lines.append(f"- Missing price/area removed: {miss}\n")
    tests.append(('duplicates_low', dup < max(1, int(0.01 * int(tx_total or 1)))),)
    tests.append(('missing_price_area_low', miss < max(1, int(0.05 * int(tx_total or 1)))),)

    # Lookup tables
    report_lines.append("\n## Lookup tables checks\n")
    # developer_mapping.json
    dev_map_path = lookups_dir / 'developer_mapping.json'
    if dev_map_path.exists():
        dm = load_json(dev_map_path)
        unmapped = dm.get('_unmapped', None)
        total_dev = dm.get('_total_developers', None)
        report_lines.append(f"- developer_mapping.json: total={total_dev}, unmapped={unmapped}\n")
        tests.append(('developer_mapping_complete', unmapped == 0))
    else:
        report_lines.append('- developer_mapping.json: MISSING\n')
        tests.append(('developer_mapping_complete', False))

    # area_stats
    area_stats_path = lookups_dir / 'area_stats.csv'
    if area_stats_path.exists():
        # quick row count
        n = sum(1 for _ in area_stats_path.read_text(encoding='utf-8').splitlines()) - 1
        report_lines.append(f"- area_stats.csv rows (excluding header): {n}\n")
        tests.append(('area_stats_populated', n >= 50))
    else:
        report_lines.append('- area_stats.csv: MISSING\n')
        tests.append(('area_stats_populated', False))

    # rent_benchmarks
    rent_bench_path = lookups_dir / 'rent_benchmarks.csv'
    if rent_bench_path.exists():
        n = sum(1 for _ in rent_bench_path.read_text(encoding='utf-8').splitlines()) - 1
        report_lines.append(f"- rent_benchmarks.csv rows (excluding header): {n}\n")
        tests.append(('rent_benchmarks_populated', n >= 50))
    else:
        report_lines.append('- rent_benchmarks.csv: MISSING\n')
        tests.append(('rent_benchmarks_populated', False))

    # ambiguous aliases in area_mapping.json
    area_map_path = lookups_dir / 'area_mapping.json'
    if area_map_path.exists():
        am = load_json(area_map_path)
        ambiguous = am.get('ambiguous_aliases', {})
        report_lines.append(f"- area_mapping.json ambiguous aliases: {len(ambiguous)} (<=10 expected)\n")
        tests.append(('area_ambiguous_aliases_ok', len(ambiguous) <= 10))
    else:
        report_lines.append('- area_mapping.json: MISSING\n')
        tests.append(('area_ambiguous_aliases_ok', False))

    # developer_stats.csv presence
    dev_stats_path = lookups_dir / 'developer_stats.csv'
    if dev_stats_path.exists():
        n = sum(1 for _ in dev_stats_path.read_text(encoding='utf-8').splitlines()) - 1
        report_lines.append(f"- developer_stats.csv rows: {n}\n")
        tests.append(('developer_stats_populated', n >= 20))
    else:
        report_lines.append('- developer_stats.csv: MISSING\n')
        tests.append(('developer_stats_populated', False))

    # Summarize tests
    report_lines.append("\n## Tests summary\n")
    passed = [name for (name, ok) in tests if ok]
    failed = [name for (name, ok) in tests if not ok]
    report_lines.append(f"- Total tests: {len(tests)}\n")
    report_lines.append(f"- Passed: {len(passed)}\n")
    report_lines.append(f"- Failed: {len(failed)}\n")
    if failed:
        report_lines.append(f"- Failed tests: {', '.join(failed)}\n")

    # Final conclusion heuristic: require >80% tests pass
    pass_ratio = len(passed) / max(1, len(tests))
    conclusion = 'PASS' if pass_ratio >= 0.8 else 'FAIL'
    report_lines.append(f"\n## Conclusion: {conclusion} (pass ratio: {pass_ratio:.2%})\n")

    write(report_lines)

def write(lines):
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text('\n'.join(lines), encoding='utf-8')
    print(f"Wrote verification report to: {report_path}")

if __name__ == '__main__':
    run()
