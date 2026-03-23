#!/usr/bin/env python3
"""
top-pl-winners.py

Purpose
-------
Analyze campaigns_clean.csv and produce a report of the
top cumulative P/L winners by ticker.

Output
------
Console output + exported report:

Top-Stock-Report.txt

Examples
--------
python top-pl-winners.py
python top-pl-winners.py --top 10
python top-pl-winners.py --top 50
python top-pl-winners.py --top 25 --output My-Top-Stocks.txt

Report contains:
    • Top cumulative $ P/L winners
    • Trade count
    • Average P/L per trade
"""

import argparse
from datetime import datetime

import pandas as pd


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Top cumulative P/L winners by ticker")
    p.add_argument(
        "--input",
        default="campaigns_clean.csv",
        help="Input CSV file (default: campaigns_clean.csv)",
    )
    p.add_argument(
        "--output",
        default="Top-Stock-Report.txt",
        help="Output text report (default: Top-Stock-Report.txt)",
    )
    p.add_argument(
        "--top",
        type=int,
        default=25,
        help="Number of top tickers to report (default: 25)",
    )
    return p


def detect_ticker_column(df: pd.DataFrame) -> str:
    if "Ticker" in df.columns:
        return "Ticker"
    if "symbol" in df.columns:
        return "symbol"
    print("Available columns:")
    print(df.columns.tolist())
    raise ValueError("No ticker column detected.")


def detect_pl_column(df: pd.DataFrame) -> str:
    possible_cols = [
        "pnl_usd",
        "PL",
        "PnL",
        "Profit",
        "ProfitUSD",
        "CampaignPL",
        "TradePL",
        "Return",
        "ReturnPct",
        "SumCampaignReturnPct",
    ]

    for c in df.columns:
        for p in possible_cols:
            if p.lower() == c.lower() or p.lower() in c.lower():
                return c

    print("Available columns:")
    print(df.columns.tolist())
    raise ValueError("No P/L column detected.")


def main() -> int:
    args = build_parser().parse_args()

    if args.top <= 0:
        raise ValueError("--top must be greater than 0")

    df = pd.read_csv(args.input)

    ticker_col = detect_ticker_column(df)
    pl_col = detect_pl_column(df)

    print(f"Using ticker column: {ticker_col}")
    print(f"Using P/L column: {pl_col}")

    summary = (
        df.groupby(ticker_col)
        .agg(
            Trades=(ticker_col, "count"),
            Total_PL=(pl_col, "sum"),
            Avg_PL=(pl_col, "mean"),
        )
        .sort_values("Total_PL", ascending=False)
    )

    top_n = summary.head(args.top)

    lines = []
    lines.append("TOP STOCK REPORT")
    lines.append("================")
    lines.append("")
    lines.append(f"Generated: {datetime.now()}")
    lines.append("")
    lines.append(f"Top {args.top} Cumulative P/L Winners")
    lines.append("-" * (len(f"Top {args.top} Cumulative P/L Winners")))
    lines.append("Ticker | Trades | Total P/L | Avg/Trade")
    lines.append("")

    for ticker, row in top_n.iterrows():
        lines.append(
            f"{str(ticker):6} | {int(row.Trades):6} | "
            f"${row.Total_PL:12.2f} | ${row.Avg_PL:10.2f}"
        )

    lines.append("")
    lines.append("Remarks")
    lines.append("-------")
    lines.append(
        "These tickers produced the highest cumulative dollar profit across all campaigns."
    )
    lines.append(
        "Momentum systems usually generate the majority of profits from a small number of stocks."
    )
    lines.append(
        f"Reviewing the top {args.top} helps reveal recurring structural patterns."
    )
    lines.append(
        "Use this report to compare float, ADR, market cap, and sector profile of the true profit leaders."
    )

    with open(args.output, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("\n".join(lines))
    print("")
    print(f"Report exported to: {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())