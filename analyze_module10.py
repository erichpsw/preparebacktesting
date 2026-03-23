#!/usr/bin/env python3
"""
SID5-focused Module 10 / backtest analyzer.

Purpose
-------
Profile SetupID 5 ("post-breakout / GapGo") in the data-gathering phase.
This version keeps the original report style, but adds the missing buckets
and cross-sections needed to study:
    - RMV
    - DCR
    - CMP
    - ORC
    - ADR
    - hold days
    - setup score
    - state
    - runner vs non-runner behavior

Expected input
--------------
CSV with campaign/backtest rows containing, at minimum:
    sid, return_pct

Preferred fields (case-insensitive aliases supported):
    sid, PROK, RUNOK, exit_type, return_pct, is_win,
    CMP, DCR, ORC, RS, RMV, ADR, hold_days, SS, STATE

Examples
--------
python analyze_module10.py --input campaigns_clean.csv
python analyze_module10.py --input module10_results.csv --sid-detail 5 4 3
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


SECTION = "=" * 70


ALIASES = {
    "sid": ["sid", "setupid", "setup_id", "setupId"],
    "prok": ["PROK", "prok", "profileok", "profile_ok", "profileOk"],
    "runok": ["RUNOK", "runok", "runnerok", "runner_ok", "runnerOk"],
    "exit_type": ["exit_type", "exit", "exitType", "exit_reason", "result"],
    "return_pct": ["return_pct", "return", "pnl_pct", "pl_pct", "ret_pct"],
    "is_win": ["is_win", "win", "winner", "isWinner"],
    "cmp": ["CMP", "cmp", "compression", "compressionScore", "compression_score"],
    "dcr": ["DCR", "dcr"],
    "orc": ["ORC", "orc", "oracleDist", "oracle_dist", "oracle_distance"],
    "rs": ["RS", "rs", "rsScore", "rs_score", "dm_rsScore"],
    "rmv": ["RMV", "rmv", "dm_rmv"],
    "adr": ["ADR", "adr", "adrPct", "adr_pct", "dm_adrPct"],
    "hold_days": ["hold_days", "holdDays", "days_held", "daysHeld"],
    "ss": ["SS", "ss", "setupscore", "setup_score", "totalScore", "total_score"],
    "state": ["STATE", "state", "State"],
}


def find_col(df: pd.DataFrame, key: str) -> str | None:
    wanted = ALIASES[key]
    lower_map = {c.lower(): c for c in df.columns}
    for cand in wanted:
        if cand in df.columns:
            return cand
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def coerce_bool01(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(int)

    if pd.api.types.is_numeric_dtype(series):
        vals = pd.to_numeric(series, errors="coerce")
        return vals.fillna(0).astype(int)

    s = series.astype(str).str.strip().str.lower()
    mapping = {
        "true": 1,
        "false": 0,
        "yes": 1,
        "no": 0,
        "1": 1,
        "0": 0,
        "y": 1,
        "n": 0,
    }
    return s.map(mapping).fillna(0).astype(int)


def to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def print_block(title: str) -> None:
    print(SECTION)
    print(title)
    print(SECTION)


def print_df(title: str, obj) -> None:
    print(title)
    if isinstance(obj, pd.Series):
        print(obj.to_string())
    else:
        print(obj.to_string())
    print()


def expectancy_table(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    grouped = df.groupby(by, dropna=False)["return_pct"]
    out = grouped.agg(["count", "mean", "median"]).rename(
        columns={"count": "trades", "mean": "expectancy_pct", "median": "median_pct"}
    )
    return out


def winrate_table(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    grouped = df.groupby(by, dropna=False)["is_win"]
    out = grouped.mean().to_frame("win_rate")
    return out


def safe_bucket(
    series: pd.Series,
    bins: Iterable[float],
    labels: list[str],
    include_lowest: bool = True,
    right: bool = True,
) -> pd.Series:
    return pd.cut(
        series,
        bins=bins,
        labels=labels,
        include_lowest=include_lowest,
        right=right,
    )


def add_buckets(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["CMP_bucket"] = safe_bucket(
        df["CMP"],
        bins=[-np.inf, 40, 55, 70, np.inf],
        labels=["<40", "40-55", "55-70", "70+"],
    )

    df["ORC_bucket"] = safe_bucket(
        df["ORC"],
        bins=[-np.inf, 0.0, 0.25, 0.50, 1.00, np.inf],
        labels=["<0", "0-0.25", "0.25-0.50", "0.50-1.00", ">1.00"],
    )

    df["DCR_bucket"] = safe_bucket(
        df["DCR"],
        bins=[-np.inf, 50, 65, 80, np.inf],
        labels=["<50", "50-65", "65-80", "80+"],
    )

    df["RS_bucket"] = safe_bucket(
        df["RS"],
        bins=[-np.inf, 80, 90, 95, np.inf],
        labels=["<80", "80-89", "90-94", "95+"],
    )

    df["RMV_bucket"] = safe_bucket(
        df["RMV"],
        bins=[-np.inf, 10, 20, 30, 40, np.inf],
        labels=["0-10", "10-20", "20-30", "30-40", "40+"],
    )

    df["ADR_bucket"] = safe_bucket(
        df["ADR"],
        bins=[-np.inf, 2.5, 4.0, 6.0, 8.0, np.inf],
        labels=["<2.5", "2.5-4", "4-6", "6-8", "8+"],
    )

    df["hold_days_bucket"] = safe_bucket(
        df["hold_days"],
        bins=[-np.inf, 5, 10, 20, np.inf],
        labels=["1-5", "6-10", "11-20", "20+"],
    )

    df["SS_bucket"] = safe_bucket(
        df["SS"],
        bins=[-np.inf, 60, 70, 80, 90, np.inf],
        labels=["<60", "60-69", "70-79", "80-89", "90+"],
    )

    return df


def add_sample_flags(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    out = expectancy_table(df, by)
    out["sample_flag"] = np.where(
        out["trades"] < 10,
        "TINY",
        np.where(out["trades"] < 20, "SMALL", ""),
    )
    return out


def one_dim_profile(df: pd.DataFrame, bucket_col: str, label: str) -> None:
    out = add_sample_flags(df, [bucket_col])
    print_df(f"{label} BUCKETS", out)


def two_way_profile(df: pd.DataFrame, bucket_col: str, label: str) -> None:
    out = add_sample_flags(df, [bucket_col, "RUNOK"])
    print_df(f"{label} x RUNOK", out)


def state_profile(df: pd.DataFrame) -> None:
    if df["STATE"].notna().sum() == 0:
        print("STATE not available.\n")
        return

    exp = add_sample_flags(df, ["STATE"])
    wr = winrate_table(df, ["STATE"])
    merged = exp.join(wr)
    print_df("STATE PROFILE", merged)


def setup_detail(df: pd.DataFrame, sid: int) -> None:
    print_block(f"SETUP {sid} DETAIL")

    sdf = df[df["sid"] == sid].copy()
    print(f"rows: {len(sdf)}\n")
    if sdf.empty:
        print("No rows for this setup.\n")
        return

    print("Non-null counts:")
    for col in ["CMP", "DCR", "ORC", "RS", "RMV", "ADR", "hold_days", "SS", "return_pct"]:
        print(f"{col}: {int(sdf[col].notna().sum())}")
    print()

    for base in ["CMP", "DCR", "ORC", "RS", "RMV", "ADR", "hold_days", "SS"]:
        bcol = f"{base}_bucket" if base not in {"hold_days", "SS"} else (
            "hold_days_bucket" if base == "hold_days" else "SS_bucket"
        )
        out = add_sample_flags(sdf, [bcol])
        print_df(f"{base} BUCKETS", out)

    if sdf["STATE"].notna().sum() > 0:
        out = add_sample_flags(sdf, ["STATE"])
        print_df("STATE", out)


def sid5_deep_dive(df: pd.DataFrame) -> None:
    sdf = df[df["sid"] == 5].copy()

    print_block("SID5 DEEP DIVE")
    print(f"rows: {len(sdf)}\n")
    if sdf.empty:
        print("No SID5 rows found.\n")
        return

    print_df("SID5 OVERALL", add_sample_flags(sdf, ["sid"]))
    print_df("SID5 WIN RATE", winrate_table(sdf, ["sid"]))

    print_df("SID5 RUNNER VS NON-RUNNER", add_sample_flags(sdf, ["RUNOK"]))
    print_df("SID5 RUNNER WIN RATE", winrate_table(sdf, ["RUNOK"]))

    print_df("SID5 PROFILE FILTER IMPACT", add_sample_flags(sdf, ["PROK"]))
    print_df("SID5 PROFILE + RUNNER COMBO", add_sample_flags(sdf, ["PROK", "RUNOK"]))

    state_profile(sdf)

    one_dim_profile(sdf, "DCR_bucket", "SID5 DCR")
    one_dim_profile(sdf, "RMV_bucket", "SID5 RMV")
    one_dim_profile(sdf, "CMP_bucket", "SID5 CMP")
    one_dim_profile(sdf, "ORC_bucket", "SID5 ORC")
    one_dim_profile(sdf, "ADR_bucket", "SID5 ADR")
    one_dim_profile(sdf, "hold_days_bucket", "SID5 HOLD DAYS")
    one_dim_profile(sdf, "SS_bucket", "SID5 SETUP SCORE")

    two_way_profile(sdf, "DCR_bucket", "SID5 DCR")
    two_way_profile(sdf, "RMV_bucket", "SID5 RMV")
    two_way_profile(sdf, "CMP_bucket", "SID5 CMP")
    two_way_profile(sdf, "ORC_bucket", "SID5 ORC")
    two_way_profile(sdf, "hold_days_bucket", "SID5 HOLD DAYS")

    winners = sdf[sdf["is_win"] == 1].copy()
    losers = sdf[sdf["is_win"] == 0].copy()

    print_block("SID5 WINNERS VS LOSERS — MEANS")
    compare = pd.DataFrame(
        {
            "all_mean": sdf[["return_pct", "CMP", "DCR", "ORC", "RS", "RMV", "ADR", "hold_days", "SS"]].mean(),
            "winner_mean": winners[["return_pct", "CMP", "DCR", "ORC", "RS", "RMV", "ADR", "hold_days", "SS"]].mean(),
            "loser_mean": losers[["return_pct", "CMP", "DCR", "ORC", "RS", "RMV", "ADR", "hold_days", "SS"]].mean(),
        }
    )
    print(compare.to_string())
    print()

    print_block("SID5 EXIT DISTRIBUTION")
    exit_counts = sdf.groupby(["RUNOK", "exit_type"], dropna=False).size()
    print(exit_counts.to_string())
    print()
    print("NORMALIZED:")
    exit_norm = exit_counts.groupby(level=0).transform(lambda x: x / x.sum())
    print(exit_norm.to_string())
    print()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="CSV file to analyze")
    ap.add_argument("--sid-detail", nargs="*", type=int, default=[5, 4, 3, 2])
    args = ap.parse_args()

    path = Path(args.input)
    if not path.exists():
        raise SystemExit(f"Input file not found: {path}")

    df = pd.read_csv(path)

    cols = {k: find_col(df, k) for k in ALIASES}
    missing = [k for k, v in cols.items() if k in {"sid", "return_pct"} and v is None]
    if missing:
        raise SystemExit(f"Missing required columns/aliases: {', '.join(missing)}")

    out = pd.DataFrame(index=df.index)
    out["sid"] = to_numeric(df[cols["sid"]]).astype("Int64")
    out["return_pct"] = to_numeric(df[cols["return_pct"]])

    out["PROK"] = coerce_bool01(df[cols["prok"]]) if cols["prok"] else 0
    out["RUNOK"] = coerce_bool01(df[cols["runok"]]) if cols["runok"] else 0

    if cols["exit_type"]:
        out["exit_type"] = df[cols["exit_type"]].astype(str).str.strip()
    else:
        out["exit_type"] = "Unknown"

    if cols["is_win"]:
        out["is_win"] = coerce_bool01(df[cols["is_win"]]).astype(int)
    else:
        out["is_win"] = (out["return_pct"] > 0).astype(int)

    for key, name in [
        ("cmp", "CMP"),
        ("dcr", "DCR"),
        ("orc", "ORC"),
        ("rs", "RS"),
        ("rmv", "RMV"),
        ("adr", "ADR"),
        ("hold_days", "hold_days"),
        ("ss", "SS"),
    ]:
        out[name] = to_numeric(df[cols[key]]) if cols[key] else np.nan

    if cols["state"]:
        out["STATE"] = df[cols["state"]].astype(str).str.strip()
    else:
        out["STATE"] = np.nan

    out = out.dropna(subset=["sid", "return_pct"]).copy()
    out["sid"] = out["sid"].astype(int)

    out = add_buckets(out)

    print(f"TOTAL ROWS: {len(out)}")
    print(f"WIN RATE: {out['is_win'].mean():.4f}")
    print()

    print_block("PROFILE FILTER IMPACT (PROK)")
    print(add_sample_flags(out, ["PROK"]).to_string())
    print()

    print_block("RUNNER FILTER IMPACT (RUNOK)")
    print(add_sample_flags(out, ["RUNOK"]).to_string())
    print()

    print_block("PROFILE + RUNNER COMBO")
    print(add_sample_flags(out, ["PROK", "RUNOK"]).to_string())
    print()

    print_block("SETUP ID PERFORMANCE")
    print(add_sample_flags(out, ["sid"]).to_string())
    print()

    print("WIN RATE BY SETUP")
    print(out.groupby("sid")["is_win"].mean().to_string())
    print()

    print_block("GLOBAL RMV BUCKET EXPECTANCY")
    print(add_sample_flags(out, ["RMV_bucket"]).to_string())
    print()

    non_runners = out[out["RUNOK"] == 0].copy()
    print_block("SETUP PERFORMANCE (NON-RUNNERS)")
    print(add_sample_flags(non_runners, ["sid"]).to_string())
    print()

    sid5_deep_dive(out)

    for sid in args.sid_detail:
        setup_detail(out, sid)


if __name__ == "__main__":
    main()