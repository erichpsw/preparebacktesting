#!/usr/bin/env python3
"""
setup_weight_optimizer.py

Standalone research tool for optimizing Setup Grader factor weights by SetupID.

Status
------
Legacy / secondary research utility.
This is not the preferred Stage 2 setup profiling workflow.
Use setupid_baseline_profile_engine.py (winner clustering mode) for primary Stage 2 profiling.

Purpose
-------
This tool is intentionally separate from the stage pipeline.
It reads only campaigns_clean.csv and produces research outputs that help you
judge whether Trend / RS / Compression / Volume weights are predictive.


What it does
------------
For each SetupID present in the dataset and meeting minimum sample thresholds:
1) enumerates weight combinations for TR / RS / CMP / VQS
2) recalculates a normalized 0-100 score for every trade
3) buckets scores into score bands
4) evaluates expectancy, win rate, runner rate, and trade count by bucket
5) measures whether higher score buckets actually correspond to better outcomes
6) ranks weight sets by slope / monotonicity / trade retention balance

Recommended use
---------------
Use this as a research report generator.
Review the outputs manually, then decide whether to update Module 8 in Pine.

Example
-------
python setup_weight_optimizer.py \
  --input campaigns_clean.csv \
  --output results_weight_optimizer \
  --min-trades-per-sid 80 \
  --min-trades-per-bucket 12 \
  --weight-step 0.05 \
  --top-k 50 \
  --score-bucket-size 5 \
  --min-score-threshold 60
"""

from __future__ import annotations

import argparse
import itertools
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------

FACTOR_COLS = ["TR", "RS", "CMP", "VQS"]
RET_COL = "return_pct"
RUNNER_COL = "RUNNER"
SID_COL = "sid"
DATE_COL = "entry_dt"

DEFAULT_WEIGHT_STEP = 0.05
DEFAULT_BUCKET_SIZE = 5
DEFAULT_TOP_K = 50
DEFAULT_MIN_SCORE_THRESHOLD = 60

# Score ranking weights for optimizer output
OPTIMIZER_WEIGHTS = {
    "high_score_expectancy": 0.30,
    "score_slope": 0.25,
    "monotonicity": 0.20,
    "trade_retention": 0.10,
    "runner_lift": 0.05,
    "separation": 0.10,
}


# -----------------------------------------------------------------------------
# HELPERS
# -----------------------------------------------------------------------------

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_float(x: Any) -> Optional[float]:
    try:
        if pd.isna(x):
            return None
        return float(x)
    except Exception:
        return None


def safe_int(x: Any) -> Optional[int]:
    try:
        if pd.isna(x):
            return None
        return int(x)
    except Exception:
        return None


def clamp_100(x: float) -> float:
    return max(0.0, min(100.0, float(x)))


def calc_metrics(df: pd.DataFrame) -> Dict[str, float]:
    if df.empty:
        return {
            "trades": 0,
            "expectancy": np.nan,
            "win_rate": np.nan,
            "median_return": np.nan,
            "runner_rate": np.nan,
        }

    rets = pd.to_numeric(df[RET_COL], errors="coerce").dropna()
    if rets.empty:
        return {
            "trades": 0,
            "expectancy": np.nan,
            "win_rate": np.nan,
            "median_return": np.nan,
            "runner_rate": np.nan,
        }

    runner_rate = np.nan
    if RUNNER_COL in df.columns:
        runner_vals = pd.to_numeric(df[RUNNER_COL], errors="coerce")
        if runner_vals.notna().any():
            runner_rate = float((runner_vals == 1).mean())

    return {
        "trades": int(len(rets)),
        "expectancy": float(rets.mean()),
        "win_rate": float((rets > 0).mean()),
        "median_return": float(rets.median()),
        "runner_rate": runner_rate,
    }


def linear_slope(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or len(y) < 2:
        return np.nan
    try:
        slope = np.polyfit(x, y, 1)[0]
        return float(slope)
    except Exception:
        return np.nan


def monotonic_fraction(values: List[float]) -> float:
    vals = [v for v in values if not pd.isna(v)]
    if len(vals) < 2:
        return np.nan
    good = 0
    total = 0
    for i in range(len(vals) - 1):
        total += 1
        if vals[i + 1] >= vals[i]:
            good += 1
    return float(good / total) if total else np.nan


def score_bucket_label(lo: int, hi: int) -> str:
    return f"{lo}-{hi}"


def bucket_score_series(series: pd.Series, bucket_size: int) -> pd.Series:
    vals = pd.to_numeric(series, errors="coerce")
    bucket_lo = (np.floor(vals / bucket_size) * bucket_size).astype("Int64")
    bucket_hi = bucket_lo + bucket_size
    out = bucket_lo.astype(str) + "-" + bucket_hi.astype(str)
    out[vals.isna()] = pd.NA
    return out


def generate_weight_combos(step: float) -> List[Tuple[float, float, float, float]]:
    points = np.arange(0.0, 1.0 + 1e-9, step)
    combos: List[Tuple[float, float, float, float]] = []

    # integer-composition style enumeration via product, then exact sum ~= 1.0
    for w in itertools.product(points, repeat=4):
        if abs(sum(w) - 1.0) <= 1e-9:
            combos.append(tuple(float(round(v, 6)) for v in w))

    return combos


def format_weight_id(w_tr: float, w_rs: float, w_cmp: float, w_vqs: float) -> str:
    return f"TR{w_tr:.2f}_RS{w_rs:.2f}_CMP{w_cmp:.2f}_VQS{w_vqs:.2f}"


# -----------------------------------------------------------------------------
# DATA LOADING
# -----------------------------------------------------------------------------

def load_data(csv_path: str, min_trades_per_sid: int) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Input file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    required = {SID_COL, DATE_COL, RET_COL, *FACTOR_COLS}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df = df.dropna(subset=[DATE_COL, RET_COL, SID_COL]).copy()

    for col in FACTOR_COLS + [RET_COL, SID_COL]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=FACTOR_COLS + [RET_COL, SID_COL]).copy()
    df[SID_COL] = df[SID_COL].astype(int)

    sid_counts = df[SID_COL].value_counts()
    valid_sids = sorted(sid_counts[sid_counts >= min_trades_per_sid].index.tolist())
    if not valid_sids:
        raise ValueError(f"No SetupIDs meet minimum trade count of {min_trades_per_sid}")

    df = df[df[SID_COL].isin(valid_sids)].copy()
    return df


# -----------------------------------------------------------------------------
# CORE OPTIMIZATION LOGIC
# -----------------------------------------------------------------------------

def compute_weighted_score(
    df: pd.DataFrame,
    w_tr: float,
    w_rs: float,
    w_cmp: float,
    w_vqs: float,
) -> pd.Series:
    tr = pd.to_numeric(df["TR"], errors="coerce").fillna(0.0).clip(0, 100)
    rs = pd.to_numeric(df["RS"], errors="coerce").fillna(0.0).clip(0, 100)
    cmp_ = pd.to_numeric(df["CMP"], errors="coerce").fillna(0.0).clip(0, 100)
    vqs = pd.to_numeric(df["VQS"], errors="coerce").fillna(0.0).clip(0, 100)

    total = tr * w_tr + rs * w_rs + cmp_ * w_cmp + vqs * w_vqs
    return total.clip(0, 100)


def evaluate_weight_combo(
    df_sid: pd.DataFrame,
    w_tr: float,
    w_rs: float,
    w_cmp: float,
    w_vqs: float,
    bucket_size: int,
    min_trades_per_bucket: int,
    min_score_threshold: int,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    work = df_sid.copy()
    work["score"] = compute_weighted_score(work, w_tr, w_rs, w_cmp, w_vqs)
    work["score_bucket"] = bucket_score_series(work["score"], bucket_size)

    bucket_rows: List[Dict[str, Any]] = []
    for bucket, block in work.groupby("score_bucket", dropna=True):
        if pd.isna(bucket):
            continue
        metrics = calc_metrics(block)
        if metrics["trades"] < min_trades_per_bucket:
            continue

        lo_str, hi_str = str(bucket).split("-")
        lo = int(lo_str)
        hi = int(hi_str)
        mid = (lo + hi) / 2.0

        bucket_rows.append(
            {
                "Score_Bucket": bucket,
                "Bucket_Low": lo,
                "Bucket_High": hi,
                "Bucket_Mid": mid,
                "Trades": metrics["trades"],
                "Expectancy": metrics["expectancy"],
                "Win_Rate": metrics["win_rate"],
                "Median_Return": metrics["median_return"],
                "Runner_Rate": metrics["runner_rate"],
            }
        )

    bucket_df = pd.DataFrame(bucket_rows)
    if bucket_df.empty:
        return {
            "Trades": len(work),
            "Valid_Buckets": 0,
            "HighScore_Trades": 0,
            "HighScore_Expectancy": np.nan,
            "Score_Slope": np.nan,
            "Monotonicity": np.nan,
            "Trade_Retention": np.nan,
            "Runner_Lift": np.nan,
            "Expectancy_Separation": np.nan,
            "Composite_Rank_Score": -np.inf,
        }, bucket_df

    bucket_df = bucket_df.sort_values("Bucket_Low").reset_index(drop=True)

    x = bucket_df["Bucket_Mid"].to_numpy(dtype=float)
    y = bucket_df["Expectancy"].to_numpy(dtype=float)
    slope = linear_slope(x, y)
    mono = monotonic_fraction(bucket_df["Expectancy"].tolist())

    high_score_df = work[work["score"] >= min_score_threshold].copy()
    high_metrics = calc_metrics(high_score_df)
    base_metrics = calc_metrics(work)

    low_bucket_exp = float(bucket_df["Expectancy"].min()) if len(bucket_df) else np.nan
    high_bucket_exp = float(bucket_df["Expectancy"].max()) if len(bucket_df) else np.nan
    separation = high_bucket_exp - low_bucket_exp if not (pd.isna(high_bucket_exp) or pd.isna(low_bucket_exp)) else np.nan

    runner_lift = np.nan
    if not pd.isna(high_metrics["runner_rate"]) and not pd.isna(base_metrics["runner_rate"]):
        runner_lift = high_metrics["runner_rate"] - base_metrics["runner_rate"]

    retention = high_metrics["trades"] / max(len(work), 1)

    # bounded ranking transforms
    high_exp_score = 0.0 if pd.isna(high_metrics["expectancy"]) else min(max(high_metrics["expectancy"] / 0.02, 0.0), 3.0)
    slope_score = 0.0 if pd.isna(slope) else min(max(slope / 0.001, 0.0), 3.0)
    mono_score = 0.0 if pd.isna(mono) else mono
    retention_score = min(retention / 0.50, 1.0)
    runner_score = 0.0 if pd.isna(runner_lift) else min(max(runner_lift / 0.10, 0.0), 1.0)
    sep_score = 0.0 if pd.isna(separation) else min(max(separation / 0.03, 0.0), 2.0)

    composite = (
        OPTIMIZER_WEIGHTS["high_score_expectancy"] * high_exp_score +
        OPTIMIZER_WEIGHTS["score_slope"] * slope_score +
        OPTIMIZER_WEIGHTS["monotonicity"] * mono_score +
        OPTIMIZER_WEIGHTS["trade_retention"] * retention_score +
        OPTIMIZER_WEIGHTS["runner_lift"] * runner_score +
        OPTIMIZER_WEIGHTS["separation"] * sep_score
    )

    summary = {
        "Trades": len(work),
        "Valid_Buckets": len(bucket_df),
        "Base_Expectancy": base_metrics["expectancy"],
        "Base_Win_Rate": base_metrics["win_rate"],
        "Base_Runner_Rate": base_metrics["runner_rate"],
        "HighScore_Trades": high_metrics["trades"],
        "HighScore_Expectancy": high_metrics["expectancy"],
        "HighScore_Win_Rate": high_metrics["win_rate"],
        "HighScore_Runner_Rate": high_metrics["runner_rate"],
        "Score_Slope": slope,
        "Monotonicity": mono,
        "Trade_Retention": retention,
        "Runner_Lift": runner_lift,
        "Expectancy_Separation": separation,
        "Composite_Rank_Score": composite,
    }
    return summary, bucket_df


# -----------------------------------------------------------------------------
# OUTPUTS
# -----------------------------------------------------------------------------

def save_heatmap_like_table(df_results: pd.DataFrame, sid: int, output_dir: Path) -> None:
    if df_results.empty:
        return

    pivot = df_results.pivot_table(
        index="wRS",
        columns="wVQS",
        values="Composite_Rank_Score",
        aggfunc="mean",
    )
    pivot.to_csv(output_dir / f"sid_{sid}_weights_heatmap_rs_vqs.csv")


def save_top_chart(df_top: pd.DataFrame, sid: int, output_dir: Path) -> None:
    if df_top.empty:
        return

    chart_df = df_top.head(15).copy().iloc[::-1]
    plt.figure(figsize=(10, 6))
    plt.barh(chart_df["Weight_ID"], chart_df["Composite_Rank_Score"])
    plt.xlabel("Composite Rank Score")
    plt.ylabel("Weight Set")
    plt.title(f"SID {sid} — Top Weight Sets")
    plt.tight_layout()
    plt.savefig(output_dir / f"sid_{sid}_top_weight_sets.png", dpi=120, facecolor="white")
    plt.close()


def save_bucket_chart(bucket_df: pd.DataFrame, sid: int, weight_id: str, output_dir: Path) -> None:
    if bucket_df.empty:
        return

    plt.figure(figsize=(9, 5))
    plt.plot(bucket_df["Bucket_Mid"], bucket_df["Expectancy"] * 100, marker="o")
    plt.xlabel("Score Bucket Mid")
    plt.ylabel("Expectancy (%)")
    plt.title(f"SID {sid} — Score Bucket Expectancy — {weight_id}")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / f"sid_{sid}_{weight_id}_bucket_expectancy.png", dpi=120, facecolor="white")
    plt.close()


def write_summary_txt(
    output_dir: Path,
    sid_summaries: Dict[int, pd.DataFrame],
    params: argparse.Namespace,
) -> None:
    path = output_dir / "weight_optimizer_summary.txt"
    with open(path, "w", encoding="utf-8") as f:
        f.write("SETUP WEIGHT OPTIMIZER SUMMARY\n")
        f.write("==============================\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("Parameters\n")
        f.write("----------\n")
        f.write(f"Input: {params.input}\n")
        f.write(f"Weight step: {params.weight_step}\n")
        f.write(f"Min trades per SID: {params.min_trades_per_sid}\n")
        f.write(f"Min trades per bucket: {params.min_trades_per_bucket}\n")
        f.write(f"Score bucket size: {params.score_bucket_size}\n")
        f.write(f"Min score threshold: {params.min_score_threshold}\n")
        f.write(f"Top K retained: {params.top_k}\n\n")

        for sid, df_top in sid_summaries.items():
            f.write(f"SID {sid}\n")
            f.write("------\n")
            if df_top.empty:
                f.write("No qualifying weight sets.\n\n")
                continue

            best = df_top.iloc[0]
            f.write(f"Best weight set: {best['Weight_ID']}\n")
            f.write(
                f"Weights: TR={best['wTR']:.2f}, RS={best['wRS']:.2f}, "
                f"CMP={best['wCMP']:.2f}, VQS={best['wVQS']:.2f}\n"
            )
            f.write(f"High-score expectancy: {best['HighScore_Expectancy'] * 100:.2f}%\n")
            f.write(f"Score slope: {best['Score_Slope']:.6f}\n")
            f.write(f"Monotonicity: {best['Monotonicity']:.2f}\n")
            f.write(f"Trade retention: {best['Trade_Retention']:.2f}\n")
            f.write(f"Composite score: {best['Composite_Rank_Score']:.4f}\n\n")

            f.write("Top 10\n")
            for _, row in df_top.head(10).iterrows():
                f.write(
                    f"- {row['Weight_ID']} | high_exp={row['HighScore_Expectancy'] * 100:.2f}% | "
                    f"slope={row['Score_Slope']:.6f} | mono={row['Monotonicity']:.2f} | "
                    f"retain={row['Trade_Retention']:.2f} | score={row['Composite_Rank_Score']:.4f}\n"
                )
            f.write("\n")


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone SetupID weight optimizer")
    parser.add_argument("--input", required=True, help="Path to campaigns_clean.csv")
    parser.add_argument("--output", default="results_weight_optimizer", help="Output directory")
    parser.add_argument("--min-trades-per-sid", type=int, default=80)
    parser.add_argument("--min-trades-per-bucket", type=int, default=12)
    parser.add_argument("--weight-step", type=float, default=DEFAULT_WEIGHT_STEP)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--score-bucket-size", type=int, default=DEFAULT_BUCKET_SIZE)
    parser.add_argument("--min-score-threshold", type=int, default=DEFAULT_MIN_SCORE_THRESHOLD)
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    output_dir = Path(args.output)
    ensure_dir(output_dir)

    print("\n" + "=" * 72)
    print("SETUP WEIGHT OPTIMIZER")
    print("=" * 72)

    df = load_data(args.input, args.min_trades_per_sid)
    sids = sorted(df[SID_COL].dropna().astype(int).unique().tolist())
    weight_combos = generate_weight_combos(args.weight_step)

    print(f"Loaded {len(df):,} rows")
    print(f"Eligible SetupIDs: {sids}")
    print(f"Generated {len(weight_combos):,} weight combinations")

    sid_summaries: Dict[int, pd.DataFrame] = {}

    for sid in sids:
        print("\n" + "-" * 72)
        print(f"Analyzing SID {sid}")
        print("-" * 72)

        df_sid = df[df[SID_COL] == sid].copy()
        result_rows: List[Dict[str, Any]] = []
        best_bucket_tables: Dict[str, pd.DataFrame] = {}

        for w_tr, w_rs, w_cmp, w_vqs in weight_combos:
            summary, bucket_df = evaluate_weight_combo(
                df_sid=df_sid,
                w_tr=w_tr,
                w_rs=w_rs,
                w_cmp=w_cmp,
                w_vqs=w_vqs,
                bucket_size=args.score_bucket_size,
                min_trades_per_bucket=args.min_trades_per_bucket,
                min_score_threshold=args.min_score_threshold,
            )

            weight_id = format_weight_id(w_tr, w_rs, w_cmp, w_vqs)
            row = {
                "SID": sid,
                "Weight_ID": weight_id,
                "wTR": w_tr,
                "wRS": w_rs,
                "wCMP": w_cmp,
                "wVQS": w_vqs,
            }
            row.update(summary)
            result_rows.append(row)
            best_bucket_tables[weight_id] = bucket_df

        results_df = pd.DataFrame(result_rows)
        results_df = results_df.sort_values(
            ["Composite_Rank_Score", "HighScore_Expectancy", "Score_Slope", "Monotonicity"],
            ascending=[False, False, False, False],
        ).reset_index(drop=True)

        top_df = results_df.head(args.top_k).copy()
        sid_summaries[sid] = top_df

        sid_dir = output_dir / f"sid_{sid}"
        ensure_dir(sid_dir)

        results_df.to_csv(sid_dir / f"sid_{sid}_all_weight_results.csv", index=False)
        top_df.to_csv(sid_dir / f"sid_{sid}_top_weight_results.csv", index=False)

        # Save bucket tables for top 5
        for _, row in top_df.head(5).iterrows():
            weight_id = row["Weight_ID"]
            bucket_df = best_bucket_tables.get(weight_id, pd.DataFrame())
            if not bucket_df.empty:
                bucket_df.to_csv(sid_dir / f"sid_{sid}_{weight_id}_bucket_table.csv", index=False)
                save_bucket_chart(bucket_df, sid, weight_id, sid_dir)

        save_heatmap_like_table(results_df, sid, sid_dir)
        save_top_chart(top_df, sid, sid_dir)

        if not top_df.empty:
            best = top_df.iloc[0]
            print(
                f"Best: {best['Weight_ID']} | high_exp={best['HighScore_Expectancy'] * 100:.2f}% | "
                f"slope={best['Score_Slope']:.6f} | mono={best['Monotonicity']:.2f} | "
                f"retain={best['Trade_Retention']:.2f}"
            )
        else:
            print("No valid weight sets ranked")

    write_summary_txt(output_dir, sid_summaries, args)

    print("\n" + "=" * 72)
    print(f"Done. Outputs written to: {output_dir}")
    print("=" * 72)


if __name__ == "__main__":
    main()
