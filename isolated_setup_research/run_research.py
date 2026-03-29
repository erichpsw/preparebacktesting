"""
run_research.py — Main entry point for the isolated setup research framework.

Usage
-----
Basic (uses DEFAULT_INPUT_CSV from config.py):
    python run_research.py

Custom input:
    python run_research.py --input /path/to/my_trades.csv

Custom output directory:
    python run_research.py --output output/my_run

Both:
    python run_research.py --input my_trades.csv --output output/run1
"""

from __future__ import annotations

import argparse
from pathlib import Path

from branch_tester import run_branch_comparison
from config import DEFAULT_INPUT_CSV, OUTPUT_DIR
from data_loader import load_data
from reporting import run_reporting
from utils import info


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Isolated Setup Research Framework — hypothesis testing for trading setups."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help=f"Path to input CSV (default: {DEFAULT_INPUT_CSV})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=f"Output directory (default: {OUTPUT_DIR})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input if args.input is not None else DEFAULT_INPUT_CSV
    output_dir = args.output if args.output is not None else OUTPUT_DIR

    info("=" * 60)
    info("Isolated Setup Research Framework")
    info("=" * 60)

    # 1. Load and validate data
    df = load_data(input_path)

    # 2. Run branch comparison (A: CORE, B: CORE+PROFILE, C: PROFILE→CORE)
    info("Running branch comparisons...")
    branch_results = run_branch_comparison(df)

    # 3. Generate all output files
    run_reporting(df, branch_results, output_dir)

    info("Done.")


if __name__ == "__main__":
    main()
