#!/usr/bin/env python3
"""
Filter tickers from a campaign-level CSV by cumulative P/L % and/or expectancy %.

Expected input columns from campaigns_clean.csv:
- symbol
- return_pct   (per-campaign return in percent units, e.g. 2.35 means +2.35%)
Optional:
- pnl_usd
- sid
- entry_dt

Important:
- cumulative P/L % != expectancy %
- cumulative P/L % = sum of return_pct for each ticker
- expectancy %     = mean(return_pct) for each ticker

Examples:
  python filter_tickers_by_pl_or_expectancy.py \
    --input campaigns_clean.csv \
    --output tickers_expectancy_over_1_2.csv \
    --metric expectancy \
    --threshold 1.2
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd


REQUIRED_COLUMNS = {"symbol", "return_pct"}


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    # utf-8-sig safely handles BOM-prefixed CSVs exported from Excel / macOS tools
    df = pd.read_csv(path, encoding="utf-8-sig")

    if df.empty:
        raise ValueError(
            "The input CSV contains headers but no data rows. "
            "Your current uploaded file appears to be empty."
        )

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    return df



def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["symbol"] = out["symbol"].astype(str).str.strip().str.upper()
    out["return_pct"] = pd.to_numeric(out["return_pct"], errors="coerce")

    out = out.dropna(subset=["symbol", "return_pct"])
    out = out[out["symbol"] != ""]

    return out



def summarize_by_symbol(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for symbol, group in df.groupby("symbol", dropna=True):
        trades = len(group)

        expectancy_pct = round(group["return_pct"].mean() * 100.0, 2)
        cumulative_pl_pct = round(group["return_pct"].sum() * 100.0, 2)

        rows.append({
            "symbol": symbol,
            "trades": trades,
            "expectancy_pct": expectancy_pct,
            "cumulative_pl_pct": cumulative_pl_pct,
        })

    summary = pd.DataFrame(rows)

    return summary.sort_values(
        ["expectancy_pct", "cumulative_pl_pct", "trades"],
        ascending=[False, False, False]
    )

    if "pnl_usd" in df.columns:
        df2 = df.copy()
        df2["pnl_usd"] = pd.to_numeric(df2["pnl_usd"], errors="coerce")
        pnl = df2.groupby("symbol", dropna=False)["pnl_usd"].sum(min_count=1).reset_index()
        pnl = pnl.rename(columns={"pnl_usd": "cumulative_pnl_usd"})
        summary = summary.merge(pnl, on="symbol", how="left")

    return summary.sort_values(
        by=["cumulative_pl_pct", "expectancy_pct", "trades"],
        ascending=[False, False, False],
    )



def apply_filter(summary: pd.DataFrame, metric: str, threshold: float) -> pd.DataFrame:
    if metric == "cumulative":
        filtered = summary[summary["cumulative_pl_pct"] > threshold]
    elif metric == "expectancy":
        filtered = summary[summary["expectancy_pct"] > threshold]
    elif metric == "both":
        filtered = summary[
            (summary["cumulative_pl_pct"] > threshold)
            & (summary["expectancy_pct"] > threshold)
        ]
    else:
        raise ValueError(f"Unsupported metric: {metric}")

    return filtered.copy()



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Filter tickers by cumulative P/L %% or expectancy %% from a campaign CSV."
    )
    parser.add_argument("--input", required=True, help="Path to input CSV file")
    parser.add_argument("--output", required=True, help="Path to output CSV file")
    parser.add_argument(
        "--metric",
        choices=["cumulative", "expectancy", "both"],
        default="cumulative",
        help="Which metric to filter on",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=1.2,
        help="Percent threshold. Example: 1.2 means 1.2%%",
    )
    parser.add_argument(
        "--min-trades",
        type=int,
        default=1,
        help="Minimum number of trades required for a ticker to be kept",
    )
    return parser



def main() -> int:
    args = build_parser().parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    try:
        df = load_csv(input_path)
        df = clean_data(df)
        summary = summarize_by_symbol(df)
        summary = summary[summary["trades"] >= args.min_trades].copy()
        filtered = apply_filter(summary, args.metric, args.threshold)

        export_cols = filtered[["symbol", "trades", "expectancy_pct", "cumulative_pl_pct"]].copy()
        export_cols = export_cols.rename(columns={
            "symbol": "Ticker",
            "trades": "Trades",
            "expectancy_pct": "ExpectancyPct",
            "cumulative_pl_pct": "CumPLPct",
        })

        output_path.parent.mkdir(parents=True, exist_ok=True)
        export_cols.to_csv(output_path, index=False)

        print({
            "input_rows": int(len(df)),
            "unique_tickers": int(df["symbol"].nunique()),
            "metric": args.metric,
            "threshold_pct": args.threshold,
            "min_trades": args.min_trades,
            "matched_tickers": int(len(filtered)),
            "output_file": str(output_path),
        })
        return 0

    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())