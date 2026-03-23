#!/usr/bin/env python3
"""
setup_quant_analysis.py

Perform multi-factor interaction analysis and expectancy‑bucket analysis
against campaigns_clean.csv.  Outputs summary tables and prints top
combinations.

This script is a companion to the rule‑miner tools and is intended for
ad‑hoc exploration.  Run as:

    python setup_quant_analysis.py --input campaigns_clean.csv

"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def in_bucket(val, lo, hi):
    if pd.isna(val):
        return False
    if lo is None and hi is None:
        return True
    if lo is None:
        return val < hi
    if hi is None:
        return val >= lo
    return (val >= lo) & (val < hi)


def metrics(sub: pd.DataFrame) -> dict | None:
    n = len(sub)
    if n == 0:
        return None
    r = sub["return_pct"].astype(float)
    trades = n
    win = (r > 0).mean()
    exp = r.mean()
    avg = r.mean()
    median = r.median()
    return {
        "trades": trades,
        "win_rate": win,
        "expectancy": exp,
        "avg_return": avg,
        "median_return": median,
    }


def eval_conditions(df: pd.DataFrame, conds: list[tuple[str, tuple]]) -> dict | None:
    mask = pd.Series(True, index=df.index)
    for factor, (lo, hi) in conds:
        mask &= df[factor].apply(lambda v: in_bucket(v, lo, hi))
    sub = df[mask]
    return metrics(sub)


def run_interaction_analysis(df: pd.DataFrame, out_dir: Path) -> None:
    buckets = {
        "RS": [(None, 70), (70, 80), (80, 90), (90, None)],
        "CMP": [(None, 50), (50, 70), (70, 85), (85, None)],
        "TR": [(None, 50), (50, 65), (65, 80), (80, None)],
        "RMV": [(None, 0), (0, 20), (20, 35), (35, 50), (50, None)],
        "DCR": [(None, 50), (50, 65), (65, 80), (80, None)],
        "ORC": [(None, 0), (0, 0.25), (0.25, 0.5), (0.5, 1.0), (1.0, None)],
        "ADR": [(None, 4), (4, 6), (6, 8), (8, None)],
    }
    factors = list(buckets.keys())
    two, three, four = [], [], []

    for f1, f2 in itertools.combinations(factors, 2):
        for b1 in buckets[f1]:
            for b2 in buckets[f2]:
                m = eval_conditions(df, [(f1, b1), (f2, b2)])
                if m and m["trades"] >= 20:
                    two.append(((f1, b1), (f2, b2), m))
    two_sorted = sorted(two, key=lambda x: x[2]["expectancy"], reverse=True)

    for f1, f2, f3 in itertools.combinations(factors, 3):
        for b1 in buckets[f1]:
            for b2 in buckets[f2]:
                for b3 in buckets[f3]:
                    m = eval_conditions(df, [(f1, b1), (f2, b2), (f3, b3)])
                    if m and m["trades"] >= 15:
                        three.append(((f1, b1), (f2, b2), (f3, b3), m))
    three_sorted = sorted(three, key=lambda x: x[3]["expectancy"], reverse=True)

    for f1, f2, f3, f4 in itertools.combinations(factors, 4):
        for b1 in buckets[f1]:
            for b2 in buckets[f2]:
                for b3 in buckets[f3]:
                    for b4 in buckets[f4]:
                        m = eval_conditions(df, [(f1, b1), (f2, b2), (f3, b3), (f4, b4)])
                        if m and m["trades"] >= 10:
                            four.append(((f1, b1), (f2, b2), (f3, b3), (f4, b4), m))
    four_sorted = sorted(four, key=lambda x: x[4]["expectancy"], reverse=True)

    pd.DataFrame([
        {"condition": str(r[0:2]), **r[2]}
        for r in two_sorted[:20]
    ]).to_csv(out_dir / "best_2factor.csv", index=False)

    pd.DataFrame([
        {"condition": str(r[0:3]), **r[3]}
        for r in three_sorted[:20]
    ]).to_csv(out_dir / "best_3factor.csv", index=False)

    pd.DataFrame([
        {"condition": str(r[0:4]), **r[4]}
        for r in four_sorted[:15]
    ]).to_csv(out_dir / "best_4factor.csv", index=False)

    print("Interaction analysis complete; results in:", out_dir)


def run_expectancy_bucket(df: pd.DataFrame, out_dir: Path) -> None:
    bins = [-np.inf, 0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, np.inf]
    labels = [
        "<0", "0-0.25", "0.25-0.5", "0.5-0.75",
        "0.75-1.0", "1.0-1.25", "1.25-1.5", "1.5+",
    ]
    df["exp_bucket"] = pd.cut(df["return_pct"], bins=bins, labels=labels, right=False)
    summary = (
        df.groupby("exp_bucket")["return_pct"]
        .agg(trades="count", avg_return="mean", median_return="median")
        .assign(win_rate=lambda x: df.groupby("exp_bucket")["return_pct"].apply(lambda r: (r > 0).mean()))
    )
    summary["total_profit"] = df.groupby("exp_bucket")["return_pct"].sum()
    summary.to_csv(out_dir / "expectancy_buckets.csv")

    for sid in sorted(df["sid"].dropna().unique()):
        sub = df[df["sid"] == sid].copy()
        sub["exp_bucket"] = pd.cut(sub["return_pct"], bins=bins, labels=labels, right=False)
        s = sub.groupby("exp_bucket")["return_pct"].agg(count="count", mean="mean", median="median")
        s["win_rate"] = sub.groupby("exp_bucket")["return_pct"].apply(lambda r: (r > 0).mean())
        s.to_csv(out_dir / f"expectancy_buckets_sid{sid}.csv")

    print("Expectancy bucket summaries written to", out_dir)

    # runner vs non-runner summary
    if "RUNNER" in df.columns:
        runners = df[df["RUNNER"] == True]
        nonr = df[df["RUNNER"] != True]
        r_stats = runners["return_pct"].describe().to_frame().T
        r_stats["win_rate"] = (runners["return_pct"] > 0).mean()
        n_stats = nonr["return_pct"].describe().to_frame().T
        n_stats["win_rate"] = (nonr["return_pct"] > 0).mean()
        summary_runner = pd.concat([r_stats.assign(group="runner"), n_stats.assign(group="non_runner")])
        summary_runner.to_csv(out_dir / "runner_vs_nonrunner_summary.csv", index=False)
        print("Runner vs non-runner summary written to", out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="campaigns_clean.csv path")
    parser.add_argument("--output-dir", default="quant_analysis", help="directory for results")
    args = parser.parse_args()
    df = load_data(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_interaction_analysis(df, out_dir)
    run_expectancy_bucket(df, out_dir)
