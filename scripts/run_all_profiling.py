#!/usr/bin/env python3
"""
Master script to run all data profiling tasks.

Usage:
    python scripts/run_all_profiling.py [--phase PHASE]
    
Phases:
    1 - Large CSV profiling (Transactions, Rent_Contracts, Units)
    2 - Small CSV profiling (Buildings, Projects, Valuation)
    3 - Entity resolution analysis
    4 - Tourism data profiling
    5 - Generate quality report
    all - Run all phases (default)

Example:
    python scripts/run_all_profiling.py --phase 1
    python scripts/run_all_profiling.py --phase all
"""

import subprocess
import sys
import argparse
from pathlib import Path
from datetime import datetime


def run_script(script_path: str, args: list = None) -> bool:
    """Run a Python script and return success status."""
    cmd = [sys.executable, script_path]
    if args:
        cmd.extend(args)
    
    print(f"\n{'='*60}")
    print(f"Running: {script_path}")
    print(f"{'='*60}\n")
    
    try:
        result = subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n⚠️ Script failed with exit code {e.returncode}")
        return False
    except FileNotFoundError:
        print(f"\n❌ Script not found: {script_path}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Run data profiling pipeline")
    parser.add_argument(
        "--phase", 
        default="all",
        help="Which phase to run (1-5 or 'all')"
    )
    args = parser.parse_args()
    
    SCRIPTS_DIR = Path(__file__).parent
    
    # Define phases
    phases = {
        "1": [
            ("Large CSV Profiling", str(SCRIPTS_DIR / "profile_large_csv.py"), ["--all"]),
        ],
        "2": [
            ("Small CSV Profiling", str(SCRIPTS_DIR / "profile_small_csv.py"), []),
        ],
        "3": [
            ("Entity Resolution", str(SCRIPTS_DIR / "analyze_entity_resolution.py"), []),
        ],
        "4": [
            ("Tourism Data", str(SCRIPTS_DIR / "profile_tourism_data.py"), []),
        ],
        "5": [
            ("Quality Report", str(SCRIPTS_DIR / "generate_quality_report.py"), []),
        ],
    }
    
    print("="*60)
    print("DATA PROFILING PIPELINE")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    # Determine which phases to run
    if args.phase.lower() == "all":
        phases_to_run = ["1", "2", "3", "4", "5"]
    else:
        phases_to_run = [args.phase]
    
    # Run phases
    results = {}
    for phase_num in phases_to_run:
        if phase_num not in phases:
            print(f"\n❌ Unknown phase: {phase_num}")
            continue
        
        for name, script, script_args in phases[phase_num]:
            print(f"\n📊 Phase {phase_num}: {name}")
            success = run_script(script, script_args)
            results[f"Phase {phase_num}: {name}"] = "✅ Success" if success else "❌ Failed"
    
    # Summary
    print("\n" + "="*60)
    print("PIPELINE SUMMARY")
    print("="*60)
    
    for task, status in results.items():
        print(f"  {status} {task}")
    
    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check if quality report exists
    report_path = Path("Docs/DATA_QUALITY_REPORT.md")
    if report_path.exists():
        print(f"\n📋 Quality report available at: {report_path}")


if __name__ == "__main__":
    main()

