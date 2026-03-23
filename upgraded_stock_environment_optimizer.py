#!/usr/bin/env python3
"""
upgraded_stock_environment_optimizer.py

Analyze a combined ticker-level backtest + TradingView export and discover
which stock characteristics are associated with the best average campaign
return / expectancy.

What it does
------------
1) Loads the combined report CSV
2) Normalizes common column names
3) Parses numeric fields like:
   - $4.94B
   - 5.32%
   - 95.75M
4) Evaluates bucket-level expectancy for:
   - Price
   - Market Cap
   - ADR
   - Dollar Volume Avg 60 Day
   - Float
   - EPS Growth YoY
   - Revenue Growth YoY
5) Evaluates category-level expectancy for:
   - Sector
   - Industry
6) Optionally evaluates 2-factor combinations across numeric buckets
7) Evaluates trade distribution by bucket/category so you can see
   where 80% / 90% of campaigns actually occurred
8) Writes:
   - a text report
   - bucket summary CSV
   - category summary CSV
   - 2-factor combo CSV
   - trade distribution CSV
   - scored ticker CSV with bucket assignments

Important
---------
This optimizer intentionally EXCLUDES RS from the analysis.
RS should be handled as a separate universe process.

Interpretation
--------------
- AvgExpectancyPct:
    simple average expectancy across ticker rows in the bucket
- WeightedExpectancyPct:
    campaign-weighted expectancy across tickers in the bucket
    = sum(ticker_expectancy * ticker_campaigns) / sum(ticker_campaigns)
- PercentOfTotal:
    share of total campaigns that fell in a bucket/category
- CumulativePercent:
    cumulative campaign share after sorting largest to smallest

This is usually the better statistic for research.

Example
-------
python upgraded_stock_environment_optimizer.py \
    --input combined_tradingview_backtest_report_clean.csv \
    --output-report Environment_Report/stock_environment_optimizer_report.txt \
    --output-buckets Environment_Report/stock_environment_bucket_summary.csv \
    --output-categories Environment_Report/stock_environment_category_summary.csv \
    --output-combos Environment_Report/stock_environment_combo_summary.csv \
    --output-distribution Environment_Report/environment_trade_distribution.csv \
    --output-scored Environment_Report/stock_environment_scored_tickers.csv
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


ALIASES = {
    "ticker": ["Ticker", "ticker", "Symbol", "symbol"],
    "campaigns": ["Campaigns", "campaigns", "Trades", "trades"],
    "expectancy": [
        "AvgCampaignReturnPct",
        "ExpectancyPct",
        "avg_campaign_return_pct",
        "expectancy_pct",
    ],
    "sum_return": [
        "SumCampaignReturnPct",
        "CumPLPct",
        "sum_campaign_return_pct",
        "cumulative_pl_pct",
    ],
    "price": ["Price", "price", "Last", "Close"],
    "market_cap": [
        "Market Cap",
        "MarketCap",
        "Market capitalization",
        "market_cap",
        "market capitalization",
    ],
    "sector": ["Sector", "sector"],
    "industry": ["Industry", "industry"],
    "adr": [
        "ADR",
        "Adr",
        "Average Daily Range %",
        "average daily range %",
        "adr",
    ],
    "dollar_volume": [
        "Dollar Volume Avg 30 Day",
        "DollarVolume60D",
        "Price * Average Volume 30 days",
        "price * average volume 30 days",
        "dollar_volume_30d",
    ],
    "float": [
        "Float",
        "Free float",
        "float",
        "free float",
    ],
    "eps_growth": [
        "EPS Growth YoY",
        "Earnings per share diluted growth %, TTM YoY",
        "eps_growth_yoy",
        "eps growth yoy",
    ],
    "revenue_growth": [
        "Revenue Growth YoY",
        "Revenue growth %, TTM YoY",
        "revenue_growth_yoy",
        "revenue growth yoy",
    ],
}

NUMERIC_BUCKETS = {
    "price": [0, 10, 20, 50, 100, 200, 300, 500, math.inf],
    "market_cap": [0, 500e6, 1e9, 2e9, 5e9, 10e9, 25e9, 50e9, 100e9, 250e9, 500e9, math.inf],
    "adr": [0, 3, 5, 7, 10, math.inf],
    "dollar_volume": [0, 10e6, 20e6, 50e6, 100e6, 200e6, 500e6, math.inf],
    "float": [0, 50e6, 150e6, 300e6, 600e6, 1e9, math.inf],
    "eps_growth": [-math.inf, 0, 20, 50, 100, math.inf],
    "revenue_growth": [-math.inf, 10, 25, 50, math.inf],
}

NUMERIC_LABELS = {
    "price": "Price",
    "market_cap": "Market Cap",
    "adr": "ADR",
    "dollar_volume": "Dollar Volume Avg 30 Day",
    "float": "Float",
    "eps_growth": "EPS Growth YoY",
    "revenue_growth": "Revenue Growth YoY",
}

CATEGORICAL_LABELS = {
    "sector": "Sector",
    "industry": "Industry",
}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Discover the optimal stock environment from a combined backtest + TradingView dataset."
    )
    p.add_argument("--input", required=True, help="Combined CSV input file")
    p.add_argument(
        "--output-report",
        default="stock_environment_optimizer_report.txt",
        help="Text report output path",
    )
    p.add_argument(
        "--output-buckets",
        default="stock_environment_bucket_summary.csv",
        help="Bucket summary CSV output path",
    )
    p.add_argument(
        "--output-categories",
        default="stock_environment_category_summary.csv",
        help="Category summary CSV output path",
    )
    p.add_argument(
        "--output-combos",
        default="stock_environment_combo_summary.csv",
        help="Two-factor combo summary CSV output path",
    )
    p.add_argument(
        "--output-distribution",
        default="environment_trade_distribution.csv",
        help="Trade distribution CSV output path",
    )
    p.add_argument(
        "--output-scored",
        default="stock_environment_scored_tickers.csv",
        help="Ticker-level scored CSV output path",
    )
    p.add_argument(
        "--min-campaigns",
        type=int,
        default=3,
        help="Minimum campaigns required for a ticker row to be included",
    )
    p.add_argument(
        "--min-bucket-campaigns",
        type=int,
        default=20,
        help="Minimum campaigns required to keep a bucket/category in expectancy summaries",
    )
    p.add_argument(
        "--top-n-categories",
        type=int,
        default=15,
        help="How many sector/industry rows to show in report",
    )
    p.add_argument(
        "--top-n-combos",
        type=int,
        default=30,
        help="How many two-factor combos to show in report",
    )
    p.add_argument(
        "--disable-combos",
        action="store_true",
        help="Skip two-factor combination analysis",
    )
    return p


def normalize_colname(name: str) -> str:
    return re.sub(r"\s+", " ", str(name).strip())


def find_col(df: pd.DataFrame, logical_name: str) -> Optional[str]:
    candidates = ALIASES[logical_name]
    norm_map = {normalize_colname(c): c for c in df.columns}
    for cand in candidates:
        if normalize_colname(cand) in norm_map:
            return norm_map[normalize_colname(cand)]
    return None


def parse_human_number(x) -> float:
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)

    s = str(x).strip()
    if not s or s.lower() in {"nan", "none", "null", "n/a", "na", "-"}:
        return np.nan

    s = s.replace(",", "").replace("$", "").replace("€", "").replace("£", "")
    pct = False
    if s.endswith("%"):
        pct = True
        s = s[:-1].strip()

    mult = 1.0
    if s.endswith(("K", "k")):
        mult = 1e3
        s = s[:-1]
    elif s.endswith(("M", "m")):
        mult = 1e6
        s = s[:-1]
    elif s.endswith(("B", "b")):
        mult = 1e9
        s = s[:-1]
    elif s.endswith(("T", "t")):
        mult = 1e12
        s = s[:-1]

    s = s.strip()
    try:
        val = float(s) * mult
    except ValueError:
        return np.nan

    return val


def weighted_expectancy(expectancy: pd.Series, campaigns: pd.Series) -> float:
    mask = (~expectancy.isna()) & (~campaigns.isna())
    if mask.sum() == 0:
        return np.nan
    total_campaigns = campaigns[mask].sum()
    if total_campaigns <= 0:
        return np.nan
    return float((expectancy[mask] * campaigns[mask]).sum() / total_campaigns)


def format_money_compact(x: float) -> str:
    if pd.isna(x):
        return "NA"
    ax = abs(x)
    if ax >= 1e12:
        return f"${x / 1e12:.2f}T"
    if ax >= 1e9:
        return f"${x / 1e9:.2f}B"
    if ax >= 1e6:
        return f"${x / 1e6:.2f}M"
    if ax >= 1e3:
        return f"${x / 1e3:.2f}K"
    return f"${x:.2f}"


def format_num_compact(x: float) -> str:
    if pd.isna(x):
        return "NA"
    ax = abs(x)
    if ax >= 1e12:
        return f"{x / 1e12:.2f}T"
    if ax >= 1e9:
        return f"{x / 1e9:.2f}B"
    if ax >= 1e6:
        return f"{x / 1e6:.2f}M"
    if ax >= 1e3:
        return f"{x / 1e3:.2f}K"
    return f"{x:.2f}"


def bucket_labels(edges: Sequence[float], kind: str) -> List[str]:
    labels: List[str] = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        if kind in {"market_cap", "dollar_volume"}:
            if math.isinf(hi):
                labels.append(f">{format_money_compact(lo)}")
            else:
                labels.append(f"{format_money_compact(lo)}–{format_money_compact(hi)}")
        elif kind == "float":
            if math.isinf(hi):
                labels.append(f">{format_num_compact(lo)}")
            else:
                labels.append(f"{format_num_compact(lo)}–{format_num_compact(hi)}")
        else:
            if math.isinf(hi):
                labels.append(f">{lo:g}")
            else:
                labels.append(f"{lo:g}–{hi:g}")
    return labels


def assign_bucket(s: pd.Series, kind: str, edges: Sequence[float]) -> pd.Series:
    labels = bucket_labels(edges, kind)
    return pd.cut(
        s,
        bins=edges,
        labels=labels,
        include_lowest=True,
        right=False,
    )


def analyze_bucket_table(
    df: pd.DataFrame,
    var_key: str,
    display_name: str,
    min_bucket_campaigns: int,
) -> pd.DataFrame:
    bucket_col = f"{var_key}_bucket"
    rows = []

    for bucket, g in df.groupby(bucket_col, dropna=True):
        campaigns = int(g["Campaigns"].sum())
        if campaigns < min_bucket_campaigns:
            continue

        rows.append(
            {
                "Variable": display_name,
                "Key": var_key,
                "Bucket": str(bucket),
                "Tickers": int(len(g)),
                "Campaigns": campaigns,
                "PercentOfTotal": round(campaigns / float(df["Campaigns"].sum()) * 100.0, 2),
                "AvgExpectancyPct": round(float(g["ExpectancyPct"].mean()), 2),
                "WeightedExpectancyPct": round(
                    weighted_expectancy(g["ExpectancyPct"], g["Campaigns"]), 2
                ),
                "MedianExpectancyPct": round(float(g["ExpectancyPct"].median()), 2),
            }
        )

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(
            ["WeightedExpectancyPct", "Campaigns", "Tickers"],
            ascending=[False, False, False],
        ).reset_index(drop=True)
    return out


def analyze_category_table(
    df: pd.DataFrame,
    col: str,
    display_name: str,
    min_bucket_campaigns: int,
) -> pd.DataFrame:
    rows = []
    temp = df[df[col].notna() & (df[col].astype(str).str.strip() != "")].copy()
    if temp.empty:
        return pd.DataFrame()

    for val, g in temp.groupby(col, dropna=True):
        campaigns = int(g["Campaigns"].sum())
        if campaigns < min_bucket_campaigns:
            continue

        rows.append(
            {
                "Variable": display_name,
                "Key": col,
                "Category": str(val),
                "Tickers": int(len(g)),
                "Campaigns": campaigns,
                "PercentOfTotal": round(campaigns / float(df["Campaigns"].sum()) * 100.0, 2),
                "AvgExpectancyPct": round(float(g["ExpectancyPct"].mean()), 2),
                "WeightedExpectancyPct": round(
                    weighted_expectancy(g["ExpectancyPct"], g["Campaigns"]), 2
                ),
                "MedianExpectancyPct": round(float(g["ExpectancyPct"].median()), 2),
            }
        )

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(
            ["WeightedExpectancyPct", "Campaigns", "Tickers"],
            ascending=[False, False, False],
        ).reset_index(drop=True)
    return out


def analyze_two_factor_combos(
    df: pd.DataFrame,
    numeric_keys: Sequence[str],
    min_bucket_campaigns: int,
) -> pd.DataFrame:
    rows = []

    bucket_cols = [f"{k}_bucket" for k in numeric_keys if f"{k}_bucket" in df.columns]
    for i in range(len(bucket_cols)):
        for j in range(i + 1, len(bucket_cols)):
            c1 = bucket_cols[i]
            c2 = bucket_cols[j]
            k1 = c1.replace("_bucket", "")
            k2 = c2.replace("_bucket", "")
            d1 = NUMERIC_LABELS[k1]
            d2 = NUMERIC_LABELS[k2]

            subset = df[df[c1].notna() & df[c2].notna()].copy()
            if subset.empty:
                continue

            grouped = subset.groupby([c1, c2], dropna=True)
            for (b1, b2), g in grouped:
                campaigns = int(g["Campaigns"].sum())
                if campaigns < min_bucket_campaigns:
                    continue

                rows.append(
                    {
                        "Factor1": d1,
                        "Bucket1": str(b1),
                        "Factor2": d2,
                        "Bucket2": str(b2),
                        "Tickers": int(len(g)),
                        "Campaigns": campaigns,
                        "PercentOfTotal": round(campaigns / float(df["Campaigns"].sum()) * 100.0, 2),
                        "AvgExpectancyPct": round(float(g["ExpectancyPct"].mean()), 2),
                        "WeightedExpectancyPct": round(
                            weighted_expectancy(g["ExpectancyPct"], g["Campaigns"]), 2
                        ),
                        "MedianExpectancyPct": round(float(g["ExpectancyPct"].median()), 2),
                    }
                )

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(
            ["WeightedExpectancyPct", "Campaigns", "Tickers"],
            ascending=[False, False, False],
        ).reset_index(drop=True)
    return out


def analyze_trade_distribution(df: pd.DataFrame, bucket_keys: Sequence[str]) -> pd.DataFrame:
    total_campaigns = float(df["Campaigns"].sum())
    rows: List[Dict[str, object]] = []

    for key in bucket_keys:
        bucket_col = f"{key}_bucket"
        if bucket_col not in df.columns:
            continue

        grouped = (
            df.groupby(bucket_col, dropna=True)["Campaigns"]
            .sum()
            .reset_index()
            .rename(columns={bucket_col: "Bucket"})
        )

        if grouped.empty:
            continue

        grouped["Variable"] = NUMERIC_LABELS[key]
        grouped["PercentOfTotal"] = grouped["Campaigns"] / total_campaigns * 100.0
        grouped = grouped.sort_values("Campaigns", ascending=False).reset_index(drop=True)
        grouped["CumulativePercent"] = grouped["PercentOfTotal"].cumsum()

        for _, r in grouped.iterrows():
            rows.append(
                {
                    "Variable": r["Variable"],
                    "Bucket": str(r["Bucket"]),
                    "Campaigns": int(r["Campaigns"]),
                    "PercentOfTotal": round(float(r["PercentOfTotal"]), 2),
                    "CumulativePercent": round(float(r["CumulativePercent"]), 2),
                }
            )

    for col, display_name in [("Sector", "Sector"), ("Industry", "Industry")]:
        if col not in df.columns:
            continue

        temp = df[df[col].notna() & (df[col].astype(str).str.strip() != "")].copy()
        if temp.empty:
            continue

        grouped = (
            temp.groupby(col, dropna=True)["Campaigns"]
            .sum()
            .reset_index()
            .rename(columns={col: "Bucket"})
        )
        grouped["Variable"] = display_name
        grouped["PercentOfTotal"] = grouped["Campaigns"] / total_campaigns * 100.0
        grouped = grouped.sort_values("Campaigns", ascending=False).reset_index(drop=True)
        grouped["CumulativePercent"] = grouped["PercentOfTotal"].cumsum()

        for _, r in grouped.iterrows():
            rows.append(
                {
                    "Variable": r["Variable"],
                    "Bucket": str(r["Bucket"]),
                    "Campaigns": int(r["Campaigns"]),
                    "PercentOfTotal": round(float(r["PercentOfTotal"]), 2),
                    "CumulativePercent": round(float(r["CumulativePercent"]), 2),
                }
            )

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["Variable", "Campaigns"], ascending=[True, False]).reset_index(drop=True)
    return out


def find_trade_zone(distribution: pd.DataFrame, variable: str, target_pct: float) -> str:
    temp = distribution[distribution["Variable"] == variable].copy()
    if temp.empty:
        return "NA"
    selected = temp[temp["CumulativePercent"] <= target_pct].copy()
    if selected.empty:
        selected = temp.head(1).copy()
    else:
        if selected["CumulativePercent"].max() < target_pct and len(selected) < len(temp):
            next_row = temp.iloc[[len(selected)]]
            selected = pd.concat([selected, next_row], ignore_index=True)
    buckets = selected["Bucket"].astype(str).tolist()
    return ", ".join(buckets)


def render_top_bucket_lines(table: pd.DataFrame, limit: int = 5) -> List[str]:
    if table.empty:
        return ["No valid buckets."]
    lines = []
    for _, r in table.head(limit).iterrows():
        lines.append(
            f"{r['Bucket']:<18} campaigns={int(r['Campaigns']):>4}  "
            f"weighted={r['WeightedExpectancyPct']:.2f}%  "
            f"avg={r['AvgExpectancyPct']:.2f}%"
        )
    return lines


def render_top_category_lines(table: pd.DataFrame, limit: int = 10) -> List[str]:
    if table.empty:
        return ["No valid categories."]
    lines = []
    for _, r in table.head(limit).iterrows():
        lines.append(
            f"{str(r['Category'])[:45]:<45} campaigns={int(r['Campaigns']):>4}  "
            f"weighted={r['WeightedExpectancyPct']:.2f}%  "
            f"avg={r['AvgExpectancyPct']:.2f}%"
        )
    return lines


def render_top_distribution_lines(dist: pd.DataFrame, variable: str, limit: int = 5) -> List[str]:
    temp = dist[dist["Variable"] == variable].copy()
    if temp.empty:
        return ["No distribution rows."]
    lines = []
    for _, r in temp.head(limit).iterrows():
        lines.append(
            f"{str(r['Bucket'])[:45]:<45} campaigns={int(r['Campaigns']):>4}  "
            f"share={r['PercentOfTotal']:.2f}%  cum={r['CumulativePercent']:.2f}%"
        )
    return lines


def build_profile_guess(bucket_tables: Dict[str, pd.DataFrame]) -> List[str]:
    lines = ["Discovered Profile Guess", "----------------------"]
    for key in ["price", "market_cap", "adr", "dollar_volume", "float", "eps_growth", "revenue_growth"]:
        table = bucket_tables.get(key, pd.DataFrame())
        if table.empty:
            continue
        top = table.iloc[0]
        lines.append(f"{NUMERIC_LABELS[key]}: {top['Bucket']}")
    return lines


def load_and_prepare(input_path: Path, min_campaigns: int) -> Tuple[pd.DataFrame, Dict[str, str]]:
    df = pd.read_csv(input_path)
    df.columns = [normalize_colname(c) for c in df.columns]

    colmap: Dict[str, str] = {}
    required = ["ticker", "campaigns", "expectancy"]
    optional = [
        "sum_return", "price", "market_cap", "sector", "industry",
        "adr", "dollar_volume", "float", "eps_growth", "revenue_growth"
    ]

    for key in required + optional:
        found = find_col(df, key)
        if found:
            colmap[key] = found

    missing = [k for k in required if k not in colmap]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. Found columns: {list(df.columns)}"
        )

    work = pd.DataFrame()
    work["Ticker"] = df[colmap["ticker"]].astype(str).str.strip()
    work["Campaigns"] = pd.to_numeric(df[colmap["campaigns"]], errors="coerce")
    work["ExpectancyPct"] = df[colmap["expectancy"]].map(parse_human_number)

    if "sum_return" in colmap:
        work["SumCampaignReturnPct"] = df[colmap["sum_return"]].map(parse_human_number)

    for key in optional:
        if key in colmap and key not in {"sum_return", "sector", "industry"}:
            work[NUMERIC_LABELS.get(key, key)] = df[colmap[key]].map(parse_human_number)

    if "sector" in colmap:
        work["Sector"] = df[colmap["sector"]].astype(str).str.strip()
        work.loc[work["Sector"].isin(["", "nan", "None"]), "Sector"] = np.nan

    if "industry" in colmap:
        work["Industry"] = df[colmap["industry"]].astype(str).str.strip()
        work.loc[work["Industry"].isin(["", "nan", "None"]), "Industry"] = np.nan

    work = work[work["Ticker"].notna() & (work["Ticker"] != "")]
    work = work[work["Campaigns"].notna() & (work["Campaigns"] >= min_campaigns)]
    work = work[work["ExpectancyPct"].notna()].copy()

    rename_back = {
        "Price": "price",
        "Market Cap": "market_cap",
        "ADR": "adr",
        "Dollar Volume Avg 30 Day": "dollar_volume",
        "Float": "float",
        "EPS Growth YoY": "eps_growth",
        "Revenue Growth YoY": "revenue_growth",
    }
    for human, internal in rename_back.items():
        if human in work.columns:
            work[internal] = work[human]

    return work.reset_index(drop=True), colmap


def main() -> int:
    args = build_parser().parse_args()

    input_path = Path(args.input)
    out_report = Path(args.output_report)
    out_buckets = Path(args.output_buckets)
    out_categories = Path(args.output_categories)
    out_combos = Path(args.output_combos)
    out_distribution = Path(args.output_distribution)
    out_scored = Path(args.output_scored)

    try:
        df, colmap = load_and_prepare(input_path, args.min_campaigns)

        baseline_tickers = int(len(df))
        baseline_campaigns = int(df["Campaigns"].sum())
        baseline_avg = round(float(df["ExpectancyPct"].mean()), 2)
        baseline_weighted = round(weighted_expectancy(df["ExpectancyPct"], df["Campaigns"]), 2)

        bucket_tables: Dict[str, pd.DataFrame] = {}
        all_bucket_tables: List[pd.DataFrame] = []

        for key, edges in NUMERIC_BUCKETS.items():
            if key not in df.columns:
                continue
            if df[key].notna().sum() == 0:
                continue

            bucket_col = f"{key}_bucket"
            df[bucket_col] = assign_bucket(df[key], key, edges)
            table = analyze_bucket_table(df, key, NUMERIC_LABELS[key], args.min_bucket_campaigns)
            bucket_tables[key] = table
            if not table.empty:
                all_bucket_tables.append(table)

        category_tables: Dict[str, pd.DataFrame] = {}
        all_category_tables: List[pd.DataFrame] = []

        if "Sector" in df.columns:
            table = analyze_category_table(df, "Sector", "Sector", args.min_bucket_campaigns)
            category_tables["sector"] = table
            if not table.empty:
                all_category_tables.append(table)

        if "Industry" in df.columns:
            table = analyze_category_table(df, "Industry", "Industry", args.min_bucket_campaigns)
            category_tables["industry"] = table
            if not table.empty:
                all_category_tables.append(table)

        combo_table = pd.DataFrame()
        if not args.disable_combos:
            numeric_keys_present = [k for k in NUMERIC_BUCKETS if f"{k}_bucket" in df.columns]
            if len(numeric_keys_present) >= 2:
                combo_table = analyze_two_factor_combos(df, numeric_keys_present, args.min_bucket_campaigns)

        distribution_table = analyze_trade_distribution(df, [k for k in NUMERIC_BUCKETS if f"{k}_bucket" in df.columns])

        out_report.parent.mkdir(parents=True, exist_ok=True)
        out_buckets.parent.mkdir(parents=True, exist_ok=True)
        out_categories.parent.mkdir(parents=True, exist_ok=True)
        out_combos.parent.mkdir(parents=True, exist_ok=True)
        out_distribution.parent.mkdir(parents=True, exist_ok=True)
        out_scored.parent.mkdir(parents=True, exist_ok=True)

        bucket_out = pd.concat(all_bucket_tables, ignore_index=True) if all_bucket_tables else pd.DataFrame()
        cat_out = pd.concat(all_category_tables, ignore_index=True) if all_category_tables else pd.DataFrame()

        bucket_out.to_csv(out_buckets, index=False)
        cat_out.to_csv(out_categories, index=False)
        combo_table.to_csv(out_combos, index=False)
        distribution_table.to_csv(out_distribution, index=False)
        df.to_csv(out_scored, index=False, float_format="%.2f")

        lines: List[str] = []
        lines.append("STOCK ENVIRONMENT OPTIMIZER REPORT")
        lines.append("==================================")
        lines.append("")
        lines.append("Input Summary")
        lines.append("-------------")
        lines.append(f"Input file: {input_path}")
        lines.append(f"Ticker rows included: {baseline_tickers}")
        lines.append(f"Campaigns included: {baseline_campaigns}")
        lines.append(f"Min campaigns per ticker: {args.min_campaigns}")
        lines.append(f"Min campaigns per bucket: {args.min_bucket_campaigns}")
        lines.append(f"Universe avg expectancy: {baseline_avg:.2f}%")
        lines.append(f"Universe weighted expectancy: {baseline_weighted:.2f}%")
        lines.append("")
        lines.append("Variables Analyzed")
        lines.append("------------------")
        lines.append("Price, Market Cap, ADR, Dollar Volume Avg 30 Day, Float, EPS Growth YoY, Revenue Growth YoY, Sector, Industry")
        lines.append("RS intentionally excluded.")
        lines.append("")

        for key in ["price", "market_cap", "adr", "dollar_volume", "float", "eps_growth", "revenue_growth"]:
            table = bucket_tables.get(key, pd.DataFrame())
            if table.empty:
                continue
            lines.append(NUMERIC_LABELS[key])
            lines.append("-" * len(NUMERIC_LABELS[key]))
            lines.extend(render_top_bucket_lines(table, limit=5))
            lines.append("")

        for key in ["sector", "industry"]:
            table = category_tables.get(key, pd.DataFrame())
            if table.empty:
                continue
            title = CATEGORICAL_LABELS[key]
            lines.append(title)
            lines.append("-" * len(title))
            lines.extend(render_top_category_lines(table, limit=args.top_n_categories))
            lines.append("")

        if not combo_table.empty:
            lines.append("Top Two-Factor Combos")
            lines.append("---------------------")
            for _, r in combo_table.head(args.top_n_combos).iterrows():
                lines.append(
                    f"{r['Factor1']}={r['Bucket1']}  AND  {r['Factor2']}={r['Bucket2']}  |  "
                    f"campaigns={int(r['Campaigns'])}  weighted={r['WeightedExpectancyPct']:.2f}%  avg={r['AvgExpectancyPct']:.2f}%"
                )
            lines.append("")

        lines.append("Trade Distribution")
        lines.append("------------------")
        for variable in [
            "Price", "Market Cap", "ADR", "Dollar Volume Avg 30 Day",
            "Float", "EPS Growth YoY", "Revenue Growth YoY", "Sector", "Industry"
        ]:
            temp = distribution_table[distribution_table["Variable"] == variable]
            if temp.empty:
                continue
            lines.append(variable)
            lines.append("~" * len(variable))
            lines.extend(render_top_distribution_lines(distribution_table, variable, limit=5))
            lines.append(f"80% zone: {find_trade_zone(distribution_table, variable, 80.0)}")
            lines.append(f"90% zone: {find_trade_zone(distribution_table, variable, 90.0)}")
            lines.append("")

        lines.extend(build_profile_guess(bucket_tables))
        lines.append("")
        lines.append("Output Files")
        lines.append("------------")
        lines.append(f"Bucket summary CSV:    {out_buckets}")
        lines.append(f"Category summary CSV:  {out_categories}")
        lines.append(f"Combo summary CSV:     {out_combos}")
        lines.append(f"Distribution CSV:      {out_distribution}")
        lines.append(f"Scored ticker CSV:     {out_scored}")

        out_report.write_text("\n".join(lines), encoding="utf-8")

        print({
            "ticker_rows_included": baseline_tickers,
            "campaigns_included": baseline_campaigns,
            "universe_avg_expectancy_pct": baseline_avg,
            "universe_weighted_expectancy_pct": baseline_weighted,
            "bucket_rows": int(len(bucket_out)),
            "category_rows": int(len(cat_out)),
            "combo_rows": int(len(combo_table)),
            "distribution_rows": int(len(distribution_table)),
            "report_file": str(out_report),
            "bucket_file": str(out_buckets),
            "category_file": str(out_categories),
            "combo_file": str(out_combos),
            "distribution_file": str(out_distribution),
            "scored_file": str(out_scored),
        })
        return 0

    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
