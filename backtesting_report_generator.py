#!/usr/bin/env python3
"""
Backtesting TXT Report Generator
================================

Purpose
-------
Reads a campaign-level backtesting CSV and writes a clean plain-text report with:
- Stated expectancy goals
- Overall backtest summary
- P/L statement
- Expectancy by SetupID table
- Per-SetupID breakdown

Default output example
----------------------
Back Testing March 15, 2026 11-34 AM.txt

Expected input
--------------
A CSV like campaigns_clean.csv with campaign-level rows.

Recommended columns
-------------------
sid or SetupID
return_pct
pnl_usd
symbol
entry_dt
exit_dt
hold_days
exit_type

Usage
-----
python backtesting_report_generator.py --input campaigns_clean.csv
python backtesting_report_generator.py --input campaigns_clean.csv --output "Back Testing August 15, 2026 08-30 AM.txt"
python backtesting_report_generator.py --input campaigns_clean.csv --goal-low 1.2 --goal-high 2.0

Notes
-----
- Expectancy is calculated as the arithmetic mean of return_pct.
- return_pct is treated as percent units. Example: 1.25 means +1.25%.
- If pnl_usd is present, the report also includes a P/L statement.
- This script is standalone and uses only pandas + numpy.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


SETUP_NAMES: Dict[int, str] = {
    1: "Pullback Reclaim",
    2: "Handle Compression",
    3: "Upper Handle",
    4: "Pivot Breakout",
    5: "Continuation High",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a TXT backtesting report from a campaign-level CSV.")
    parser.add_argument("--input", required=True, help="Path to input CSV, e.g. campaigns_clean.csv")
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output TXT filename. If omitted, a timestamped name is created.",
    )
    parser.add_argument(
        "--goal-low",
        type=float,
        default=1.20,
        help="Lower bound of expectancy goal in percent. Default: 1.20",
    )
    parser.add_argument(
        "--goal-high",
        type=float,
        default=2.00,
        help="Upper bound of expectancy goal in percent. Default: 2.00",
    )
    return parser.parse_args()



def auto_output_name() -> str:
    now = datetime.now()
    return now.strftime("Back Testing %B %d, %Y %I-%M %p.txt")



def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df



def resolve_col(df: pd.DataFrame, candidates: List[str]) -> str | None:
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None



def prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    sid_col = resolve_col(df, ["sid", "setupid", "setup_id"])
    if sid_col is None:
        raise ValueError("Missing required SetupID column. Expected one of: sid, SetupID, setup_id")

    ret_col = resolve_col(df, ["return_pct", "return%", "return pct", "return"])
    if ret_col is None:
        raise ValueError("Missing required return column. Expected one of: return_pct, return%, return pct, return")

    pnl_col = resolve_col(df, ["pnl_usd", "pnl", "net_pnl", "profit_loss_usd"])

    out = df.copy()
    out["sid"] = pd.to_numeric(out[sid_col], errors="coerce")
    out["return_pct"] = pd.to_numeric(out[ret_col], errors="coerce")
    out["pnl_usd"] = pd.to_numeric(out[pnl_col], errors="coerce") if pnl_col else np.nan

    # Optional normalized columns for richer reporting
    for src, dst in [
        (resolve_col(df, ["symbol", "ticker"]), "symbol"),
        (resolve_col(df, ["entry_dt", "entry_date", "entry time"]), "entry_dt"),
        (resolve_col(df, ["exit_dt", "exit_date", "exit time"]), "exit_dt"),
        (resolve_col(df, ["hold_days", "days_held"]), "hold_days"),
        (resolve_col(df, ["exit_type", "reason", "exit reason"]), "exit_type"),
    ]:
        if src is not None:
            out[dst] = out[src]

    out = out.dropna(subset=["sid", "return_pct"]).copy()
    out["sid"] = out["sid"].astype(int)
    return out



def pct_fmt(x: float | int | None, signed: bool = False) -> str:
    if x is None or pd.isna(x):
        return "n/a"
    return f"{x:+.2f}%" if signed else f"{x:.1f}%"



def usd_fmt(x: float | int | None) -> str:
    if x is None or pd.isna(x):
        return "n/a"
    return f"${x:,.2f}"



def classify_expectancy(exp_pct: float, goal_low: float, goal_high: float) -> str:
    if pd.isna(exp_pct):
        return "n/a"
    if exp_pct >= goal_low and exp_pct <= goal_high:
        return "✅ In Goal Range"
    if exp_pct > goal_high:
        return "✅ Above Goal"
    if exp_pct >= goal_low - 0.20:
        return "⚠ Slightly Below"
    return "❌ Weak"



def compute_overall_stats(df: pd.DataFrame) -> Dict[str, float]:
    wins = df["return_pct"] > 0
    losses = df["return_pct"] < 0

    gross_profit = df.loc[df["pnl_usd"] > 0, "pnl_usd"].sum(min_count=1)
    gross_loss = df.loc[df["pnl_usd"] < 0, "pnl_usd"].sum(min_count=1)
    net_profit = df["pnl_usd"].sum(min_count=1)

    avg_win_pct = df.loc[wins, "return_pct"].mean()
    avg_loss_pct = df.loc[losses, "return_pct"].mean()

    profit_factor = np.nan
    gross_profit_abs = df.loc[df["pnl_usd"] > 0, "pnl_usd"].sum()
    gross_loss_abs = abs(df.loc[df["pnl_usd"] < 0, "pnl_usd"].sum())
    if gross_loss_abs > 0:
        profit_factor = gross_profit_abs / gross_loss_abs

    return {
        "trades": len(df),
        "wins": int(wins.sum()),
        "losses": int(losses.sum()),
        "flat": int((df["return_pct"] == 0).sum()),
        "win_rate": wins.mean() * 100.0 if len(df) else np.nan,
        "expectancy_pct": df["return_pct"].mean(),
        "median_return_pct": df["return_pct"].median(),
        "avg_win_pct": avg_win_pct,
        "avg_loss_pct": avg_loss_pct,
        "expectancy_usd": df["pnl_usd"].mean() if df["pnl_usd"].notna().any() else np.nan,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "net_profit": net_profit,
        "profit_factor": profit_factor,
        "avg_hold_days": pd.to_numeric(df.get("hold_days", pd.Series(dtype=float)), errors="coerce").mean()
        if "hold_days" in df.columns
        else np.nan,
    }



def compute_sid_table(df: pd.DataFrame, goal_low: float, goal_high: float) -> pd.DataFrame:
    rows = []
    for sid in sorted(df["sid"].dropna().unique()):
        sdf = df[df["sid"] == sid].copy()
        wins = sdf["return_pct"] > 0
        rows.append(
            {
                "sid": sid,
                "setup_name": SETUP_NAMES.get(int(sid), f"Setup {sid}"),
                "trades": len(sdf),
                "win_rate": wins.mean() * 100.0 if len(sdf) else np.nan,
                "expectancy_pct": sdf["return_pct"].mean(),
                "median_return_pct": sdf["return_pct"].median(),
                "net_pnl_usd": sdf["pnl_usd"].sum(min_count=1) if sdf["pnl_usd"].notna().any() else np.nan,
                "status": classify_expectancy(sdf["return_pct"].mean(), goal_low, goal_high),
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["sid"], ascending=True).reset_index(drop=True)
    return out



def make_ascii_table(headers: List[str], rows: List[List[str]]) -> str:
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))

    def fmt_row(row_vals: List[str]) -> str:
        return "  ".join(str(v).ljust(widths[i]) for i, v in enumerate(row_vals))

    sep = "  ".join("-" * w for w in widths)
    output = [fmt_row(headers), sep]
    output.extend(fmt_row(r) for r in rows)
    return "\n".join(output)



def render_report(df: pd.DataFrame, goal_low: float, goal_high: float, src_name: str) -> str:
    now = datetime.now().strftime("%Y-%m-%d %I:%M:%S %p")
    overall = compute_overall_stats(df)
    sid_table = compute_sid_table(df, goal_low, goal_high)

    lines: List[str] = []
    lines.append("BACKTESTING REPORT")
    lines.append("==================")
    lines.append("")
    lines.append(f"Generated: {now}")
    lines.append(f"Source File: {src_name}")
    lines.append("")

    lines.append("STATED EXPECTANCY GOALS")
    lines.append("-----------------------")
    lines.append(f"Target Win Rate: 45.0% to 55.0%")
    lines.append(f"Target Expectancy: +{goal_low:.2f}% to +{goal_high:.2f}%")
    lines.append("")

    lines.append("OVERALL BACKTEST SUMMARY")
    lines.append("------------------------")
    lines.append(f"Total Trades: {overall['trades']}")
    lines.append(f"Wins / Losses / Flat: {overall['wins']} / {overall['losses']} / {overall['flat']}")
    lines.append(f"Win Rate: {pct_fmt(overall['win_rate'])}")
    lines.append(f"Expectancy: {pct_fmt(overall['expectancy_pct'], signed=True)}")
    lines.append(f"Median Return: {pct_fmt(overall['median_return_pct'], signed=True)}")
    lines.append(f"Average Win: {pct_fmt(overall['avg_win_pct'], signed=True)}")
    lines.append(f"Average Loss: {pct_fmt(overall['avg_loss_pct'], signed=True)}")
    if not pd.isna(overall["avg_hold_days"]):
        lines.append(f"Average Hold Days: {overall['avg_hold_days']:.2f}")
    lines.append("")

    lines.append("P/L STATEMENT")
    lines.append("-------------")
    if df["pnl_usd"].notna().any():
        lines.append(f"Gross Profit: {usd_fmt(overall['gross_profit'])}")
        lines.append(f"Gross Loss: {usd_fmt(overall['gross_loss'])}")
        lines.append(f"Net Profit: {usd_fmt(overall['net_profit'])}")
        lines.append(f"Average P/L per Trade: {usd_fmt(overall['expectancy_usd'])}")
        lines.append(
            f"Profit Factor: {overall['profit_factor']:.2f}" if not pd.isna(overall['profit_factor']) else "Profit Factor: n/a"
        )
    else:
        lines.append("No pnl_usd column was found, so dollar P/L metrics could not be calculated.")
    lines.append("")

    lines.append("EXPECTANCY BY SETUPID")
    lines.append("---------------------")
    if sid_table.empty:
        lines.append("No SetupID data found.")
    else:
        headers = ["SetupID", "Trades", "Win Rate", "Expectancy", "Status vs Goal"]
        table_rows = []
        for _, r in sid_table.iterrows():
            table_rows.append(
                [
                    f"SID{int(r['sid'])} - {r['setup_name']}",
                    str(int(r["trades"])),
                    pct_fmt(r["win_rate"]),
                    pct_fmt(r["expectancy_pct"], signed=True),
                    str(r["status"]),
                ]
            )
        lines.append(make_ascii_table(headers, table_rows))
    lines.append("")

    lines.append("SETUPID BREAKDOWN")
    lines.append("-----------------")
    if sid_table.empty:
        lines.append("No SetupID data found.")
    else:
        for _, r in sid_table.iterrows():
            sid = int(r["sid"])
            sdf = df[df["sid"] == sid].copy()
            wins = sdf["return_pct"] > 0
            losses = sdf["return_pct"] < 0
            lines.append(f"SID{sid} - {r['setup_name']}")
            lines.append(f"Trades: {len(sdf)}")
            lines.append(f"Win Rate: {pct_fmt(r['win_rate'])}")
            lines.append(f"Expectancy: {pct_fmt(r['expectancy_pct'], signed=True)}")
            lines.append(f"Median Return: {pct_fmt(r['median_return_pct'], signed=True)}")
            lines.append(
                f"Average Win / Loss: {pct_fmt(sdf.loc[wins, 'return_pct'].mean(), signed=True)} / {pct_fmt(sdf.loc[losses, 'return_pct'].mean(), signed=True)}"
            )
            if sdf["pnl_usd"].notna().any():
                lines.append(f"Net P/L: {usd_fmt(sdf['pnl_usd'].sum())}")
            lines.append(f"Status vs Goal: {r['status']}")
            lines.append("")

    if "exit_type" in df.columns:
        exit_summary = df["exit_type"].astype(str).fillna("Unknown").value_counts(dropna=False)
        if not exit_summary.empty:
            lines.append("EXIT TYPE SUMMARY")
            lines.append("-----------------")
            for k, v in exit_summary.items():
                lines.append(f"{k}: {v}")
            lines.append("")

    lines.append("BOTTOM LINE")
    lines.append("-----------")
    if sid_table.empty:
        lines.append("No usable SetupID rows were found in the dataset.")
    else:
        best = sid_table.sort_values(["expectancy_pct", "trades"], ascending=[False, False]).iloc[0]
        weakest = sid_table.sort_values(["expectancy_pct", "trades"], ascending=[True, False]).iloc[0]
        lines.append(
            f"Best SetupID by expectancy: SID{int(best['sid'])} - {best['setup_name']} at {pct_fmt(best['expectancy_pct'], signed=True)} across {int(best['trades'])} trades."
        )
        lines.append(
            f"Weakest SetupID by expectancy: SID{int(weakest['sid'])} - {weakest['setup_name']} at {pct_fmt(weakest['expectancy_pct'], signed=True)} across {int(weakest['trades'])} trades."
        )
        lines.append(
            f"Overall system expectancy is {pct_fmt(overall['expectancy_pct'], signed=True)} with a {pct_fmt(overall['win_rate'])} win rate."
        )

    lines.append("")
    return "\n".join(lines)



def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_name = args.output or auto_output_name()
    output_path = Path(output_name)

    df_raw = load_csv(str(input_path))
    df = prepare_df(df_raw)
    report_text = render_report(df, args.goal_low, args.goal_high, input_path.name)
    output_path.write_text(report_text, encoding="utf-8")

    print(f"Report written to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
