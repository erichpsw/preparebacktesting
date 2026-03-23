#!/usr/bin/env python3
"""
setup_factor_analyzer_v2.py

Validation-first factor analyzer and 3-variable rule miner for campaigns_clean.csv

What it does:
1) Single-factor summaries
2) Pairwise interaction summaries
3) 3-variable rule mining
4) Chronological train/validation split
5) Overall + by SetupID outputs
6) Ranks rules by validation expectancy + stability, not just raw backtest sugar
"""

from __future__ import annotations

import argparse
import itertools
import math
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


FACTOR_COLS = ["RS", "CMP", "TR", "VQS", "RMV", "DCR", "ORC"]
SETUP_IDS = [1, 2, 3, 4]

DEFAULT_THRESHOLDS = {
    "RS":  [70, 75, 80, 85, 90, 95],
    "CMP": [40, 50, 60, 70, 80],
    "TR":  [50, 60, 70, 80, 90],
    "VQS": [40, 50, 60, 70, 80],
    "RMV": [15, 25, 35, 50, 70],
    "DCR": [40, 50, 60, 70, 80],
    "ORC": [0.00, 0.25, 0.50, 0.75, 1.00],
}

HIGHER_IS_BETTER = {"RS", "CMP", "TR", "VQS", "DCR"}
LOWER_IS_BETTER = {"RMV"}
RANGE_STYLE = {"ORC"}


def safe_profit_factor(returns: pd.Series) -> float:
    wins = returns[returns > 0].sum()
    losses = -returns[returns < 0].sum()
    if losses <= 0:
        return np.nan if wins <= 0 else np.inf
    return wins / losses


def calc_metrics(df: pd.DataFrame, ret_col: str) -> Dict[str, float]:
    n = len(df)
    if n == 0:
        return {
            "trades": 0,
            "expectancy": np.nan,
            "win_rate": np.nan,
            "avg_gain": np.nan,
            "avg_loss": np.nan,
            "profit_factor": np.nan,
            "median_return": np.nan,
        }

    r = df[ret_col].astype(float)
    wins = r[r > 0]
    losses = r[r < 0]

    return {
        "trades": int(n),
        "expectancy": float(r.mean()),
        "win_rate": float((r > 0).mean()),
        "avg_gain": float(wins.mean()) if len(wins) else np.nan,
        "avg_loss": float(losses.mean()) if len(losses) else np.nan,
        "profit_factor": float(safe_profit_factor(r)),
        "median_return": float(r.median()),
    }


def chronological_split(df: pd.DataFrame, dt_col: str, train_frac: float = 0.70) -> Tuple[pd.DataFrame, pd.DataFrame]:
    dfx = df.sort_values(dt_col).reset_index(drop=True)
    cut = max(1, int(len(dfx) * train_frac))
    return dfx.iloc[:cut].copy(), dfx.iloc[cut:].copy()


def add_quantile_buckets(df: pd.DataFrame, col: str, q: int = 5) -> pd.Series:
    s = pd.to_numeric(df[col], errors="coerce")
    try:
        return pd.qcut(s, q=q, duplicates="drop")
    except ValueError:
        return pd.cut(s, bins=min(q, max(1, s.nunique())))


def factor_summary(df: pd.DataFrame, ret_col: str, out_path: Path) -> None:
    rows = []
    for col in FACTOR_COLS:
        if col not in df.columns:
            continue
        buckets = add_quantile_buckets(df, col, q=5)
        tmp = df.copy()
        tmp["_bucket"] = buckets.astype(str)
        grouped = tmp.groupby("_bucket", dropna=False)

        for bucket, g in grouped:
            m = calc_metrics(g, ret_col)
            rows.append({
                "factor": col,
                "bucket": bucket,
                **m,
                "mean_factor_value": pd.to_numeric(g[col], errors="coerce").mean(),
            })

    out = pd.DataFrame(rows).sort_values(["factor", "expectancy"], ascending=[True, False])
    out.to_csv(out_path, index=False)


def interaction_summary(df: pd.DataFrame, ret_col: str, out_path: Path) -> None:
    rows = []
    for c1, c2 in itertools.combinations(FACTOR_COLS, 2):
        if c1 not in df.columns or c2 not in df.columns:
            continue

        tmp = df.copy()
        tmp["_b1"] = add_quantile_buckets(tmp, c1, q=4).astype(str)
        tmp["_b2"] = add_quantile_buckets(tmp, c2, q=4).astype(str)

        grouped = tmp.groupby(["_b1", "_b2"], dropna=False)
        for (b1, b2), g in grouped:
            m = calc_metrics(g, ret_col)
            rows.append({
                "factor_1": c1,
                "bucket_1": b1,
                "factor_2": c2,
                "bucket_2": b2,
                **m,
            })

    out = pd.DataFrame(rows).sort_values(["factor_1", "factor_2", "expectancy"], ascending=[True, True, False])
    out.to_csv(out_path, index=False)


def make_conditions_for_factor(col: str) -> List[Tuple[str, callable]]:
    rules = []

    if col in HIGHER_IS_BETTER:
        for t in DEFAULT_THRESHOLDS[col]:
            label = f"{col} >= {t}"
            rules.append((label, lambda d, c=col, thr=t: pd.to_numeric(d[c], errors='coerce') >= thr))

    elif col in LOWER_IS_BETTER:
        for t in DEFAULT_THRESHOLDS[col]:
            label = f"{col} <= {t}"
            rules.append((label, lambda d, c=col, thr=t: pd.to_numeric(d[c], errors='coerce') <= thr))

    elif col in RANGE_STYLE:
        vals = DEFAULT_THRESHOLDS[col]
        for lo, hi in zip(vals[:-1], vals[1:]):
            label = f"{col} in [{lo}, {hi})"
            rules.append((label, lambda d, c=col, a=lo, b=hi: (pd.to_numeric(d[c], errors='coerce') >= a) & (pd.to_numeric(d[c], errors='coerce') < b)))

    return rules


def stability_bonus(train_exp: float, val_exp: float) -> float:
    if np.isnan(train_exp) or np.isnan(val_exp):
        return -5.0
    if train_exp > 0 and val_exp > 0:
        return 2.0
    if train_exp > 0 and val_exp <= 0:
        return -4.0
    return -1.0


def overfit_penalty(train_exp: float, val_exp: float) -> float:
    if np.isnan(train_exp) or np.isnan(val_exp):
        return 5.0
    gap = train_exp - val_exp
    return max(0.0, gap) * 100.0


def rule_score(train_metrics: Dict[str, float], val_metrics: Dict[str, float]) -> float:
    val_exp = val_metrics["expectancy"]
    val_trades = val_metrics["trades"]
    tr_exp = train_metrics["expectancy"]

    if np.isnan(val_exp):
        return -9999.0

    return (
        val_exp * 100.0
        + min(val_trades, 50) * 0.03
        + stability_bonus(tr_exp, val_exp)
        - overfit_penalty(tr_exp, val_exp)
    )


def mine_rules(
    df: pd.DataFrame,
    ret_col: str,
    dt_col: str,
    out_path: Path,
    min_train_trades: int = 25,
    min_val_trades: int = 10,
    top_n: int = 200,
) -> None:
    train_df, val_df = chronological_split(df, dt_col=dt_col, train_frac=0.70)

    factor_conditions = {col: make_conditions_for_factor(col) for col in FACTOR_COLS if col in df.columns}

    rows = []

    for factors in itertools.combinations(factor_conditions.keys(), 3):
        cond_sets = [factor_conditions[f] for f in factors]

        for combo in itertools.product(*cond_sets):
            labels = [x[0] for x in combo]
            funcs = [x[1] for x in combo]

            train_mask = np.ones(len(train_df), dtype=bool)
            val_mask = np.ones(len(val_df), dtype=bool)

            for fn in funcs:
                train_mask &= fn(train_df).fillna(False).to_numpy()
                val_mask &= fn(val_df).fillna(False).to_numpy()

            train_sub = train_df.loc[train_mask]
            val_sub = val_df.loc[val_mask]

            train_metrics = calc_metrics(train_sub, ret_col)
            val_metrics = calc_metrics(val_sub, ret_col)

            if train_metrics["trades"] < min_train_trades:
                continue
            if val_metrics["trades"] < min_val_trades:
                continue
            if np.isnan(val_metrics["expectancy"]) or val_metrics["expectancy"] <= 0:
                continue

            score = rule_score(train_metrics, val_metrics)

            rows.append({
                "rule": " AND ".join(labels),
                "factor_1": factors[0],
                "factor_2": factors[1],
                "factor_3": factors[2],
                "train_trades": train_metrics["trades"],
                "train_expectancy": train_metrics["expectancy"],
                "train_win_rate": train_metrics["win_rate"],
                "train_profit_factor": train_metrics["profit_factor"],
                "validation_trades": val_metrics["trades"],
                "validation_expectancy": val_metrics["expectancy"],
                "validation_win_rate": val_metrics["win_rate"],
                "validation_profit_factor": val_metrics["profit_factor"],
                "score": score,
            })

    out = pd.DataFrame(rows)
    if out.empty:
        out.to_csv(out_path, index=False)
        return

    out = out.sort_values(
        ["score", "validation_expectancy", "validation_trades"],
        ascending=[False, False, False]
    ).head(top_n)

    out.to_csv(out_path, index=False)


def run_for_subset(df: pd.DataFrame, label: str, out_dir: Path, ret_col: str, dt_col: str) -> None:
    factor_summary(df, ret_col, out_dir / f"factor_summary_{label}.csv")
    interaction_summary(df, ret_col, out_dir / f"interaction_summary_{label}.csv")
    mine_rules(df, ret_col, dt_col, out_dir / f"rule_miner_{label}.csv")


def build_best_rules(out_dir: Path) -> None:
    paths = sorted(out_dir.glob("rule_miner_*.csv"))
    dfs = []
    for p in paths:
        if p.name == "rule_miner_best_rules.csv":
            continue
        try:
            x = pd.read_csv(p)
            if len(x):
                x.insert(0, "source", p.stem.replace("rule_miner_", ""))
                dfs.append(x)
        except Exception:
            pass

    if not dfs:
        pd.DataFrame().to_csv(out_dir / "rule_miner_best_rules.csv", index=False)
        return

    best = pd.concat(dfs, ignore_index=True).sort_values(
        ["score", "validation_expectancy", "validation_trades"],
        ascending=[False, False, False]
    )
    best.to_csv(out_dir / "rule_miner_best_rules.csv", index=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="campaigns_clean.csv")
    ap.add_argument("--output-dir", required=True, help="folder for analyzer outputs")
    ap.add_argument("--return-col", default="return_pct", help="column used for expectancy")
    ap.add_argument("--date-col", default="entry_dt", help="chronological split column")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)

    if args.date_col not in df.columns:
        raise ValueError(f"Missing date column: {args.date_col}")
    if args.return_col not in df.columns:
        raise ValueError(f"Missing return column: {args.return_col}")

    df[args.date_col] = pd.to_datetime(df[args.date_col], errors="coerce")
    df = df.dropna(subset=[args.date_col, args.return_col]).copy()

    for col in FACTOR_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    run_for_subset(df, "overall", out_dir, args.return_col, args.date_col)

    if "sid" in df.columns:
        for sid in SETUP_IDS:
            sub = df[df["sid"] == sid].copy()
            if len(sub) >= 40:
                run_for_subset(sub, f"sid{sid}", out_dir, args.return_col, args.date_col)

    build_best_rules(out_dir)
    print(f"Done. Outputs written to: {out_dir}")


if __name__ == "__main__":
    main()