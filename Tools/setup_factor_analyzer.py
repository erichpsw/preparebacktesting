#!/usr/bin/env python3

"""
setup_factor_analyzer.py

Purpose
-------
Analyze factor behavior in a stable, quant-style way before brute-force rule optimization.

This script answers:
- How does expectancy change across each factor?
- Which factor ranges are strongest?
- Which two-factor interactions matter most?
- How do these relationships differ by SetupID?

Core Factors
------------
RS
CMP
RMV
DCR
ORC

Optional Factors
----------------
TR
VQS

Input
-----
campaigns_clean.csv

Required Columns
----------------
sid, return_pct, RS, CMP, RMV, DCR, ORC

Optional Columns
----------------
TR, VQS, RUNNER

Outputs
-------
results/factor_analysis/
    factor_summary_overall.csv
    factor_summary_sid1.csv
    factor_summary_sid2.csv
    factor_summary_sid3.csv
    factor_summary_sid4.csv

    interaction_summary_overall.csv
    interaction_summary_sid1.csv
    interaction_summary_sid2.csv
    interaction_summary_sid3.csv
    interaction_summary_sid4.csv

results/factor_charts/
    univariate_<factor>_overall.png
    univariate_<factor>_sid1.png
    ...
    heatmap_<factor1>_x_<factor2>_overall.png
    heatmap_<factor1>_x_<factor2>_sid1.png
    ...
"""

import argparse
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use("default")

# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
RESULT_DIR = BASE_DIR / "results" / "factor_analysis"
CHART_DIR = BASE_DIR / "results" / "factor_charts"

RESULT_DIR.mkdir(parents=True, exist_ok=True)
CHART_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------
# Config
# ---------------------------------------------------------

MIN_BUCKET_TRADES = 15
MIN_HEATMAP_CELL_TRADES = 10
MAX_BINS = 6

CORE_FACTORS = ["RS", "CMP", "RMV", "DCR", "ORC"]
OPTIONAL_FACTORS = ["TR", "VQS"]

# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def has_col(df: pd.DataFrame, col: str) -> bool:
    return col in df.columns

def available_factors(df: pd.DataFrame) -> list[str]:
    factors = [f for f in CORE_FACTORS if has_col(df, f)]
    factors += [f for f in OPTIONAL_FACTORS if has_col(df, f)]
    return factors

def safe_qcut(series: pd.Series, q: int = MAX_BINS):
    """Quantile binning with fallback if too few unique values."""
    s = series.dropna()
    n_unique = s.nunique()
    if n_unique < 2:
        return pd.Series([None] * len(series), index=series.index)
    q = min(q, n_unique)
    try:
        return pd.qcut(series, q=q, duplicates="drop")
    except Exception:
        try:
            return pd.cut(series, bins=min(q, n_unique))
        except Exception:
            return pd.Series([None] * len(series), index=series.index)

def expectancy(series: pd.Series) -> float:
    return float(series.mean()) if len(series) else np.nan

def win_rate(series: pd.Series) -> float:
    return float((series > 0).mean()) if len(series) else np.nan

def runner_rate(df: pd.DataFrame) -> float:
    if "RUNNER" not in df.columns or len(df) == 0:
        return np.nan
    return float(df["RUNNER"].mean())

def label_for_scope(scope):
    return "overall" if scope is None else f"sid{scope}"

# ---------------------------------------------------------
# Univariate analysis
# ---------------------------------------------------------

def analyze_factor(df: pd.DataFrame, factor: str) -> pd.DataFrame:
    tmp = df[[factor, "return_pct"] + (["RUNNER"] if "RUNNER" in df.columns else [])].copy()
    tmp["bucket"] = safe_qcut(tmp[factor], q=MAX_BINS)

    grouped = tmp.dropna(subset=["bucket"]).groupby("bucket", observed=True)

    rows = []
    for bucket, g in grouped:
        if len(g) < MIN_BUCKET_TRADES:
            continue

        row = {
            "factor": factor,
            "bucket": str(bucket),
            "trades": len(g),
            "factor_min": float(g[factor].min()),
            "factor_max": float(g[factor].max()),
            "factor_mean": float(g[factor].mean()),
            "expectancy": expectancy(g["return_pct"]),
            "win_rate": win_rate(g["return_pct"]),
            "runner_rate": runner_rate(g),
        }
        rows.append(row)

    return pd.DataFrame(rows)

def plot_univariate(summary_df: pd.DataFrame, factor: str, scope_label: str):
    df = summary_df[summary_df["factor"] == factor].copy()
    if df.empty:
        return

    df = df.sort_values("factor_mean")

    plt.figure()
    plt.plot(df["factor_mean"], df["expectancy"], marker="o")
    plt.xlabel(factor)
    plt.ylabel("Expectancy")
    plt.title(f"{factor} vs Expectancy — {scope_label}")
    plt.savefig(CHART_DIR / f"univariate_{factor}_{scope_label}.png")
    plt.close()

# ---------------------------------------------------------
# Interaction analysis
# ---------------------------------------------------------

def analyze_interaction(df: pd.DataFrame, f1: str, f2: str) -> pd.DataFrame:
    cols = [f1, f2, "return_pct"] + (["RUNNER"] if "RUNNER" in df.columns else [])
    tmp = df[cols].copy()
    tmp["bucket_1"] = safe_qcut(tmp[f1], q=MAX_BINS)
    tmp["bucket_2"] = safe_qcut(tmp[f2], q=MAX_BINS)

    grouped = tmp.dropna(subset=["bucket_1", "bucket_2"]).groupby(
        ["bucket_1", "bucket_2"], observed=True
    )

    rows = []
    for (b1, b2), g in grouped:
        if len(g) < MIN_HEATMAP_CELL_TRADES:
            continue

        rows.append({
            "factor_1": f1,
            "factor_2": f2,
            "bucket_1": str(b1),
            "bucket_2": str(b2),
            "trades": len(g),
            "f1_mean": float(g[f1].mean()),
            "f2_mean": float(g[f2].mean()),
            "expectancy": expectancy(g["return_pct"]),
            "win_rate": win_rate(g["return_pct"]),
            "runner_rate": runner_rate(g),
        })

    return pd.DataFrame(rows)

def plot_heatmap(inter_df: pd.DataFrame, f1: str, f2: str, scope_label: str):
    df = inter_df[(inter_df["factor_1"] == f1) & (inter_df["factor_2"] == f2)].copy()
    if df.empty:
        return

    pivot = df.pivot_table(
        index="bucket_1",
        columns="bucket_2",
        values="expectancy",
        aggfunc="mean"
    )

    if pivot.empty:
        return

    plt.figure(figsize=(8, 6))
    plt.imshow(pivot.values, aspect="auto")
    plt.colorbar(label="Expectancy")
    plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=45, ha="right")
    plt.yticks(range(len(pivot.index)), pivot.index)
    plt.xlabel(f2)
    plt.ylabel(f1)
    plt.title(f"{f1} x {f2} Expectancy — {scope_label}")
    plt.tight_layout()
    plt.savefig(CHART_DIR / f"heatmap_{f1}_x_{f2}_{scope_label}.png")
    plt.close()

# ---------------------------------------------------------
# Scope analysis
# ---------------------------------------------------------

def run_scope_analysis(df: pd.DataFrame, setup_id=None):
    if setup_id is not None:
        work_df = df[df["sid"] == setup_id].copy()
    else:
        work_df = df.copy()

    scope_label = label_for_scope(setup_id)
    factors = available_factors(work_df)

    # Univariate
    uni_frames = []
    for factor in factors:
        factor_df = analyze_factor(work_df, factor)
        if not factor_df.empty:
            factor_df.insert(0, "scope", scope_label)
            uni_frames.append(factor_df)
            plot_univariate(factor_df, factor, scope_label)

    uni_out = pd.concat(uni_frames, ignore_index=True) if uni_frames else pd.DataFrame()
    uni_out.to_csv(RESULT_DIR / f"factor_summary_{scope_label}.csv", index=False)

    # Interactions
    inter_frames = []
    for f1, f2 in combinations(factors, 2):
        inter_df = analyze_interaction(work_df, f1, f2)
        if not inter_df.empty:
            inter_df.insert(0, "scope", scope_label)
            inter_frames.append(inter_df)
            plot_heatmap(inter_df, f1, f2, scope_label)

    inter_out = pd.concat(inter_frames, ignore_index=True) if inter_frames else pd.DataFrame()
    inter_out.to_csv(RESULT_DIR / f"interaction_summary_{scope_label}.csv", index=False)

    return uni_out, inter_out

# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to campaigns_clean.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.input)

    print("\nRunning overall factor analysis...\n")
    run_scope_analysis(df, setup_id=None)

    print("\nRunning SetupID-specific factor analysis...\n")
    for sid in [1, 2, 3, 4]:
        sid_df = df[df["sid"] == sid]
        if sid_df.empty:
            print(f"No trades found for SetupID {sid}")
            continue
        print(f"Analyzing SetupID {sid} ...")
        run_scope_analysis(df, setup_id=sid)

    print("\nDone.\n")
    print(f"Factor summaries saved to: {RESULT_DIR}")
    print(f"Charts saved to: {CHART_DIR}")

if __name__ == "__main__":
    main()