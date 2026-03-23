#!/usr/bin/env python3
"""
Analyze a merged backtest + TradingView characteristics CSV and produce an
"Optimal Stock Profile" report.

Default target profile:
- Market Cap: 1B–15B
- ADR: 3–7
- Dollar Volume Avg 30 day: >25M
- RS: >90 (only if an RS-like column exists)
- Float: <300M

Input expected from prior merge:
Ticker, Trades, ExpectancyPct, CumPLPct, Price, Market capitalization,
Sector, Industry, Average Daily Range %, Price * Average Volume 60 days,
Free float, ...

Example:
python analyze_optimal_stock_profile.py \
    --input combined_tradingview_backtest_report_clean.csv \
    --output-report optimal_stock_profile_report.txt \
    --output-matches optimal_stock_profile_matches.csv \
    --output-scored optimal_stock_profile_scored.csv \
    --min-trades 1
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Analyze optimal stock profile from combined report.")
    p.add_argument("--input", required=True, help="Combined CSV input file")
    p.add_argument("--output-report", default="optimal_stock_profile_report.txt", help="Text report path")
    p.add_argument("--output-matches", default="optimal_stock_profile_matches.csv", help="Matching tickers CSV path")
    p.add_argument("--output-scored", default="optimal_stock_profile_scored.csv", help="Scored full CSV path")
    p.add_argument("--min-trades", type=int, default=1, help="Minimum campaign/trade count to include")
    p.add_argument("--market-cap-min", type=float, default=1_000_000_000.0, help="Minimum market cap")
    p.add_argument("--market-cap-max", type=float, default=15_000_000_000.0, help="Maximum market cap")
    p.add_argument("--adr-min", type=float, default=3.0, help="Minimum ADR percent")
    p.add_argument("--adr-max", type=float, default=7.0, help="Maximum ADR percent")
    p.add_argument("--dollar-vol-min", type=float, default=25_000_000.0, help="Minimum 30D average dollar volume")
    p.add_argument("--float-max", type=float, default=300_000_000.0, help="Maximum free float")
    p.add_argument("--rs-min", type=float, default=90.0, help="Minimum RS if an RS column exists")
    return p


def find_column(df: pd.DataFrame, candidates: Sequence[str], required: bool = True) -> Optional[str]:
    lowered = {str(c).strip().lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lowered:
            return lowered[cand.lower()]
    if required:
        raise KeyError(f"Could not find required column. Tried: {candidates}")
    return None


def first_existing_column(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    lowered = {str(c).strip().lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lowered:
            return lowered[cand.lower()]
    return None


def safe_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def fmt_pct(x: float | int | None) -> str:
    if x is None or pd.isna(x):
        return "n/a"
    return f"{x:.2f}%"


def fmt_num(x: float | int | None, digits: int = 2) -> str:
    if x is None or pd.isna(x):
        return "n/a"
    return f"{x:.{digits}f}"


def fmt_money(x: float | int | None) -> str:
    if x is None or pd.isna(x):
        return "n/a"
    ax = abs(float(x))
    if ax >= 1_000_000_000:
        return f"${x/1_000_000_000:.2f}B"
    if ax >= 1_000_000:
        return f"${x/1_000_000:.2f}M"
    if ax >= 1_000:
        return f"${x/1_000:.2f}K"
    return f"${x:.2f}"


def fmt_shares(x: float | int | None) -> str:
    if x is None or pd.isna(x):
        return "n/a"
    ax = abs(float(x))
    if ax >= 1_000_000_000:
        return f"{x/1_000_000_000:.2f}B"
    if ax >= 1_000_000:
        return f"{x/1_000_000:.2f}M"
    if ax >= 1_000:
        return f"{x/1_000:.2f}K"
    return f"{x:.0f}"


def summarize_subset(df: pd.DataFrame, expect_col: str, trades_col: str) -> dict[str, float]:
    if df.empty:
        return {
            "tickers": 0,
            "campaigns": 0,
            "avg_expectancy": float("nan"),
            "median_expectancy": float("nan"),
            "weighted_expectancy": float("nan"),
        }
    campaigns = int(df[trades_col].fillna(0).sum())
    weights = df[trades_col].fillna(0)
    weighted = float((df[expect_col] * weights).sum() / weights.sum()) if weights.sum() > 0 else float("nan")
    return {
        "tickers": int(len(df)),
        "campaigns": campaigns,
        "avg_expectancy": float(df[expect_col].mean()),
        "median_expectancy": float(df[expect_col].median()),
        "weighted_expectancy": weighted,
    }


def main() -> int:
    args = build_parser().parse_args()

    input_path = Path(args.input)
    report_path = Path(args.output_report)
    matches_path = Path(args.output_matches)
    scored_path = Path(args.output_scored)

    try:
        df = pd.read_csv(input_path)
        df.columns = [str(c).strip() for c in df.columns]

        ticker_col = find_column(df, ["Ticker", "symbol"])
        trades_col = find_column(df, ["Trades", "Campaigns", "trades"])
        expectancy_col = find_column(df, ["ExpectancyPct", "AvgCampaignReturnPct", "expectancy_pct"])
        cumpl_col = first_existing_column(df, ["CumPLPct", "SumCampaignReturnPct", "cumulative_pl_pct"])
        market_cap_col = find_column(df, ["Market capitalization", "Market Cap", "MarketCap"])
        adr_col = find_column(df, ["Average Daily Range %", "ADR", "ADR%"])
        dollar_vol_col = find_column(df, ["Price * Average Volume 60 days", "Dollar Volume Avg 30 Day", "DollarVolume30D"])
        float_col = find_column(df, ["Free float", "Float", "Shares Float"])
        sector_col = first_existing_column(df, ["Sector"])
        industry_col = first_existing_column(df, ["Industry"])
        rs_col = first_existing_column(df, [
            "RS", "RS Rating", "RS Rating (12M)", "Relative Strength", "Relative Strength 12M",
            "RelativeStrength12M", "Relative Strength vs SPX"
        ])

        # numeric cleanup
        df[ticker_col] = df[ticker_col].astype(str).str.strip()
        for col in [trades_col, expectancy_col, market_cap_col, adr_col, dollar_vol_col, float_col]:
            df[col] = safe_num(df[col])
        if cumpl_col:
            df[cumpl_col] = safe_num(df[cumpl_col])
        if rs_col:
            df[rs_col] = safe_num(df[rs_col])

        df = df.dropna(subset=[ticker_col, trades_col, expectancy_col]).copy()
        df = df[df[trades_col] >= args.min_trades].copy()

        # rules
        df["MarketCapPass"] = df[market_cap_col].between(args.market_cap_min, args.market_cap_max, inclusive="both")
        df["ADRPass"] = df[adr_col].between(args.adr_min, args.adr_max, inclusive="both")
        df["DollarVolPass"] = df[dollar_vol_col] > args.dollar_vol_min
        df["FloatPass"] = df[float_col] < args.float_max

        active_rules = ["MarketCapPass", "ADRPass", "DollarVolPass", "FloatPass"]

        if rs_col:
            df["RSPass"] = df[rs_col] > args.rs_min
            active_rules.append("RSPass")
        else:
            df["RSPass"] = pd.NA

        df["ProfileScore"] = df[active_rules].fillna(False).sum(axis=1)
        df["ProfilePass"] = df[active_rules].fillna(False).all(axis=1)

        # helpful formatted cols
        df["MarketCapLabel"] = df[market_cap_col].apply(fmt_money)
        df["DollarVolLabel"] = df[dollar_vol_col].apply(fmt_money)
        df["FloatLabel"] = df[float_col].apply(fmt_shares)

        # summaries
        baseline = summarize_subset(df, expectancy_col, trades_col)
        matched = summarize_subset(df[df["ProfilePass"]], expectancy_col, trades_col)

        rule_blocks: list[str] = []
        for rule in active_rules:
            pass_df = df[df[rule] == True]
            fail_df = df[df[rule] == False]
            pass_stats = summarize_subset(pass_df, expectancy_col, trades_col)
            fail_stats = summarize_subset(fail_df, expectancy_col, trades_col)
            delta = pass_stats["weighted_expectancy"] - fail_stats["weighted_expectancy"] if (
                not pd.isna(pass_stats["weighted_expectancy"]) and not pd.isna(fail_stats["weighted_expectancy"])
            ) else float("nan")
            rule_blocks.append(
                "\n".join([
                    f"{rule}",
                    f"  Pass: tickers={pass_stats['tickers']}, campaigns={pass_stats['campaigns']}, "
                    f"avg={fmt_pct(pass_stats['avg_expectancy'])}, weighted={fmt_pct(pass_stats['weighted_expectancy'])}",
                    f"  Fail: tickers={fail_stats['tickers']}, campaigns={fail_stats['campaigns']}, "
                    f"avg={fmt_pct(fail_stats['avg_expectancy'])}, weighted={fmt_pct(fail_stats['weighted_expectancy'])}",
                    f"  Edge (pass - fail, weighted): {fmt_pct(delta)}",
                ])
            )

        match_cols = [
            ticker_col, trades_col, expectancy_col
        ]
        if cumpl_col:
            match_cols.append(cumpl_col)
        for col in [sector_col, industry_col, market_cap_col, adr_col, dollar_vol_col, float_col, rs_col]:
            if col and col not in match_cols:
                match_cols.append(col)
        match_cols += active_rules + ["ProfileScore", "ProfilePass"]

        matches = df[df["ProfilePass"]].copy()
        matches = matches.sort_values([expectancy_col, trades_col], ascending=[False, False])

        # export cleaned/scored files
        scored_cols = []
        preferred_cols = [ticker_col, trades_col, expectancy_col]
        if cumpl_col:
            preferred_cols.append(cumpl_col)
        for col in [sector_col, industry_col, market_cap_col, adr_col, dollar_vol_col, float_col, rs_col]:
            if col and col not in preferred_cols:
                preferred_cols.append(col)
        preferred_cols += active_rules + ["ProfileScore", "ProfilePass"]
        for col in preferred_cols:
            if col and col in df.columns and col not in scored_cols:
                scored_cols.append(col)

        scored_export = df[scored_cols].copy()
        matches_export = matches[[c for c in match_cols if c in matches.columns]].copy()

        # write files
        report_path.parent.mkdir(parents=True, exist_ok=True)
        matches_path.parent.mkdir(parents=True, exist_ok=True)
        scored_path.parent.mkdir(parents=True, exist_ok=True)

        matches_export.to_csv(matches_path, index=False, float_format="%.2f")
        scored_export.to_csv(scored_path, index=False, float_format="%.2f")

        top_matches = matches.head(15)
        lines: list[str] = []
        lines.append("OPTIMAL STOCK PROFILE REPORT")
        lines.append("============================")
        lines.append("")
        lines.append("Target Profile")
        lines.append("--------------")
        lines.append(f"Market Cap: {fmt_money(args.market_cap_min)}–{fmt_money(args.market_cap_max)}")
        lines.append(f"ADR: {args.adr_min:.1f}–{args.adr_max:.1f}%")
        lines.append(f"Dollar Volume Avg 30 Day: > {fmt_money(args.dollar_vol_min)}")
        lines.append(f"Float: < {fmt_shares(args.float_max)}")
        if rs_col:
            lines.append(f"RS: > {args.rs_min:.1f}")
        else:
            lines.append("RS: not evaluated (no RS column found in input)")
        lines.append("")
        lines.append("Baseline vs Profile Match")
        lines.append("-------------------------")
        lines.append(
            f"Universe baseline: tickers={baseline['tickers']}, campaigns={baseline['campaigns']}, "
            f"avg expectancy={fmt_pct(baseline['avg_expectancy'])}, weighted expectancy={fmt_pct(baseline['weighted_expectancy'])}"
        )
        lines.append(
            f"Profile matches:  tickers={matched['tickers']}, campaigns={matched['campaigns']}, "
            f"avg expectancy={fmt_pct(matched['avg_expectancy'])}, weighted expectancy={fmt_pct(matched['weighted_expectancy'])}"
        )
        if baseline["tickers"] > 0:
            lines.append(f"Match rate: {matched['tickers'] / baseline['tickers'] * 100:.2f}% of tickers")
        if baseline["campaigns"] > 0:
            lines.append(f"Campaign capture: {matched['campaigns'] / baseline['campaigns'] * 100:.2f}% of campaigns")
        lines.append("")
        lines.append("Rule-Level Breakdown")
        lines.append("--------------------")
        lines.extend(rule_blocks)
        lines.append("")
        lines.append("Top Matching Tickers")
        lines.append("--------------------")
        if top_matches.empty:
            lines.append("No tickers matched the full profile.")
        else:
            header = ["Ticker", "Trades", "AvgCampaignReturnPct"]
            if cumpl_col:
                header.append("SumCampaignReturnPct")
            header += ["Sector", "Industry", "MarketCap", "ADR", "DollarVol30D", "Float"]
            if rs_col:
                header.append("RS")
            lines.append(" | ".join(header))
            for _, row in top_matches.iterrows():
                parts = [
                    str(row[ticker_col]),
                    str(int(row[trades_col])) if not pd.isna(row[trades_col]) else "n/a",
                    fmt_pct(row[expectancy_col]),
                ]
                if cumpl_col:
                    parts.append(fmt_pct(row[cumpl_col]))
                parts += [
                    str(row[sector_col]) if sector_col else "n/a",
                    str(row[industry_col]) if industry_col else "n/a",
                    fmt_money(row[market_cap_col]),
                    fmt_pct(row[adr_col]),
                    fmt_money(row[dollar_vol_col]),
                    fmt_shares(row[float_col]),
                ]
                if rs_col:
                    parts.append(fmt_num(row[rs_col], 2))
                lines.append(" | ".join(parts))
        lines.append("")
        lines.append("Output Files")
        lines.append("------------")
        lines.append(f"Matches CSV: {matches_path}")
        lines.append(f"Scored CSV:  {scored_path}")

        report_path.write_text("\n".join(lines), encoding="utf-8")

        print({
            "input_rows": int(len(df)),
            "min_trades": args.min_trades,
            "rs_column_found": rs_col is not None,
            "matched_tickers": int(len(matches)),
            "matched_campaigns": int(matches[trades_col].sum()) if not matches.empty else 0,
            "report_file": str(report_path),
            "matches_file": str(matches_path),
            "scored_file": str(scored_path),
        })
        return 0

    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
