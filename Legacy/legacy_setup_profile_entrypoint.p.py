#!/usr/bin/env python3
"""
SetupID Baseline Profile Engine (Stage 2)
=========================================

Purpose
-------
- Learn robust baseline profiles for each SetupID present in the dataset
- Identify truly relevant factors
- Determine preferred ranges and avoid zones
- Mine and validate small deployable rules
- Generate Pine-ready recommendations

Core Principle
--------------
Validation-first, stability-focused, anti-overfit.
This is a profile-learning engine, not a forecasting engine.

Example
-------
python stage_2_setup_profile_entrypoint.py \
  --input campaigns_clean.csv \
  --output results_profiles \
  --train-fraction 0.70 \
  --min-trades-per-sid 80 \
  --min-trades-per-bucket 15 \
  --min-trades-per-rule 10 \
  --max-promoted-rules 10 \
  --save-charts
"""

from __future__ import annotations

import argparse
import math
import os
import re
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime
from itertools import combinations, product
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# =============================================================================
# CONFIGURATION
# =============================================================================

PRIMARY_FACTORS = ["RS", "CMP", "TR", "VQS", "RMV", "DCR", "ORC", "SQZ"]
TARGET_SIDS = None

NUMERIC_FACTORS = {"RS", "CMP", "TR", "VQS", "RMV", "DCR", "ORC"}
CATEGORICAL_FACTORS = {"SQZ"}

RULE_SCORING_WEIGHTS = {
    "validation_expectancy": 0.40,
    "trade_count": 0.15,
    "train_val_stability": 0.20,
    "median_improvement": 0.15,
    "simplicity_bonus": 0.10,
}

MINIMUM_EFFECT_SIZE = 0.003          # 0.3%
MINIMUM_DRIFT_TOLERANCE = 0.03       # 3%
PREFERRED_RANGE_UPLIFT = 0.005       # 0.5%
AVOID_ZONE_DOWNLIFT = 0.005          # 0.5%
MIN_RULE_UPLIFT = 0.005              # 0.5%
NUMERIC_BUCKETS = 5
MAX_SEED_CONDITIONS_PER_FACTOR = 3
RANGE_DECIMALS = 2


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class RuleCondition:
    factor: str
    cond_type: str            # numeric / categorical
    operator: str             # >= <= in
    value: Any

    def to_text(self) -> str:
        if self.cond_type == "numeric":
            return f"{self.factor} {self.operator} {float(self.value):.{RANGE_DECIMALS}f}"
        vals = self.value if isinstance(self.value, list) else [self.value]
        return f"{self.factor} in {'|'.join(map(str, vals))}"


# =============================================================================
# HELPERS
# =============================================================================

def ensure_output_directory(output_path: str) -> Path:
    out = Path(output_path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def merge_intervals(intervals: List[Tuple[float, float]], tolerance: float = 1e-9) -> List[Tuple[float, float]]:
    if not intervals:
        return []
    intervals = sorted(intervals, key=lambda x: (x[0], x[1]))
    merged = [intervals[0]]
    for start, end in intervals[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end + tolerance:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def parse_bucket_bounds(bucket_str: str) -> Tuple[Optional[float], Optional[float]]:
    nums = re.findall(r"-?\d+(?:\.\d+)?", str(bucket_str))
    if len(nums) >= 2:
        return float(nums[0]), float(nums[1])
    return None, None


def format_interval(min_v: Optional[float], max_v: Optional[float]) -> str:
    if min_v is not None and max_v is not None:
        return f"{min_v:.{RANGE_DECIMALS}f}–{max_v:.{RANGE_DECIMALS}f}"
    if min_v is not None:
        return f">= {min_v:.{RANGE_DECIMALS}f}"
    if max_v is not None:
        return f"<= {max_v:.{RANGE_DECIMALS}f}"
    return "NA"


def split_train_validation(df_sid: pd.DataFrame, train_fraction: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    sorted_df = df_sid.sort_values("entry_dt").reset_index(drop=True)
    split_idx = max(1, min(len(sorted_df) - 1, int(len(sorted_df) * train_fraction)))
    return sorted_df.iloc[:split_idx].copy(), sorted_df.iloc[split_idx:].copy()


def safe_float(x: Any) -> Optional[float]:
    try:
        if pd.isna(x):
            return None
        return float(x)
    except Exception:
        return None


def format_threshold_string(factor: str, op: str, value: Any) -> str:
    if isinstance(value, list):
        return f"{factor} in {'|'.join(map(str, value))}"
    return f"{factor} {op} {float(value):.{RANGE_DECIMALS}f}"


def conditions_to_rule_string(conditions: List[RuleCondition]) -> str:
    return " AND ".join(c.to_text() for c in conditions)


# =============================================================================
# ARGUMENTS
# =============================================================================

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SetupID Baseline Profile Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", "-i", required=True, help="Path to clean campaign CSV")
    parser.add_argument("--output", "-o", default="results_profiles", help="Output folder")
    parser.add_argument("--train-fraction", "-tf", type=float, default=0.70)
    parser.add_argument("--min-trades-per-sid", type=int, default=100)
    parser.add_argument("--min-trades-per-bucket", type=int, default=15)
    parser.add_argument("--min-trades-per-rule", type=int, default=10)
    parser.add_argument("--max-promoted-rules", type=int, default=10)
    parser.add_argument("--save-charts", action="store_true")
    return parser.parse_args()


# =============================================================================
# STEP 1: DATA LOADING & VALIDATION
# =============================================================================

def load_and_validate_data(csv_path: str, min_trades_per_sid: int) -> Optional[pd.DataFrame]:
    print("\n" + "=" * 70)
    print("LOADING AND VALIDATING DATA")
    print("=" * 70)

    if not os.path.exists(csv_path):
        print(f"ERROR: File not found: {csv_path}")
        return None

    try:
        df = pd.read_csv(csv_path)
        print(f"✓ Loaded {len(df):,} rows from {csv_path}")
    except Exception as e:
        print(f"ERROR: Could not load CSV: {e}")
        return None

    required_cols = {"sid", "entry_dt", "return_pct", "hold_days"}
    missing = required_cols - set(df.columns)
    if missing:
        print(f"ERROR: Missing required columns: {sorted(missing)}")
        return None

    available_factors = [f for f in PRIMARY_FACTORS if f in df.columns]
    if not available_factors:
        print(f"ERROR: No research factors found. Expected at least one of: {PRIMARY_FACTORS}")
        return None
    print(f"✓ Available research factors: {', '.join(available_factors)}")

    try:
        df["entry_dt"] = pd.to_datetime(df["entry_dt"])
    except Exception as e:
        print(f"ERROR: Could not parse entry_dt: {e}")
        return None

    valid_sid_values = sorted(df["sid"].dropna().unique().tolist())
    print(f"✓ Found SetupIDs in dataset: {valid_sid_values}")

    sid_counts = df["sid"].value_counts()
    insufficient = sid_counts[sid_counts < min_trades_per_sid].index.tolist()
    if insufficient:
        print(f"⚠ Warning: SetupIDs {insufficient} have < {min_trades_per_sid} trades")

    valid_sids = sid_counts[sid_counts >= min_trades_per_sid].index.tolist()
    if not valid_sids:
        print(f"ERROR: No SetupIDs meet minimum trade count of {min_trades_per_sid}")
        return None

    df = df[df["sid"].isin(valid_sids)].copy()
    print(f"✓ Proceeding with SetupIDs {sorted(valid_sids)}")

    critical_cols = ["return_pct"] + available_factors
    missing_counts = df[critical_cols].isnull().sum()
    if missing_counts.sum() > 0:
        print("\nMissing values found:")
        for col, val in missing_counts.items():
            if val > 0:
                print(f"  {col}: {val:,} ({100 * val / len(df):.2f}%)")

    df = df.dropna(subset=critical_cols).copy()
    print(f"✓ Data validated and cleaned: {len(df):,} usable rows")

    print("\nSetupID distribution:")
    for sid in sorted(valid_sids):
        count = len(df[df["sid"] == sid])
        print(f"  SID {sid}: {count:,} trades")

    return df


# =============================================================================
# STEP 1A: BASELINE METRICS
# =============================================================================

def compute_baseline_stats(df_sid: pd.DataFrame) -> Dict[str, Any]:
    returns = df_sid["return_pct"].to_numpy()
    wins = int((returns > 0).sum())
    total_trades = len(returns)
    win_rate = wins / total_trades if total_trades else 0.0
    expectancy = float(np.mean(returns)) if total_trades else np.nan
    median_return = float(np.median(returns)) if total_trades else np.nan
    avg_hold_days = float(df_sid["hold_days"].mean()) if "hold_days" in df_sid.columns else np.nan

    wins_returns = returns[returns > 0]
    losses_returns = returns[returns < 0]
    avg_win = float(np.mean(wins_returns)) if len(wins_returns) else 0.0
    avg_loss_mag = abs(float(np.mean(losses_returns))) if len(losses_returns) else 0.0
    payoff_ratio = avg_win / avg_loss_mag if avg_loss_mag > 0 else np.nan

    runner_rate = None
    if "RUNNER" in df_sid.columns:
        runner_rate = float((df_sid["RUNNER"] == 1).mean())

    return {
        "trades": total_trades,
        "wins": wins,
        "win_rate": win_rate,
        "expectancy": expectancy,
        "median_return": median_return,
        "avg_hold_days": avg_hold_days,
        "payoff_ratio": payoff_ratio,
        "runner_rate": runner_rate,
    }


# =============================================================================
# STEP 2: FACTOR ANALYSIS
# =============================================================================

def analyze_numeric_factor(df_sid: pd.DataFrame, factor_name: str, min_trades_per_bucket: int = 20):
    if factor_name not in df_sid.columns or df_sid[factor_name].isnull().all():
        return None, {"status": "missing"}

    df_factor = df_sid.dropna(subset=[factor_name]).copy()
    if len(df_factor) < min_trades_per_bucket:
        return None, {"status": "insufficient_data"}

    try:
        df_factor["bucket"] = pd.qcut(df_factor[factor_name], q=NUMERIC_BUCKETS, duplicates="drop")
    except Exception:
        df_factor["bucket"] = pd.cut(df_factor[factor_name], bins=NUMERIC_BUCKETS, duplicates="drop")

    rows = []
    for bucket in df_factor["bucket"].cat.categories:
        bucket_data = df_factor[df_factor["bucket"] == bucket]
        if len(bucket_data) < min_trades_per_bucket:
            continue

        returns = bucket_data["return_pct"].to_numpy()
        rows.append(
            {
                "bucket": str(bucket),
                "trades": len(bucket_data),
                "wins": int((returns > 0).sum()),
                "win_rate": float((returns > 0).mean()),
                "expectancy": float(np.mean(returns)),
                "median_return": float(np.median(returns)),
                "mean_factor_value": float(bucket_data[factor_name].mean()),
            }
        )

    if not rows:
        return None, {"status": "insufficient_bucketed_trades"}

    df_buckets = pd.DataFrame(rows)
    expectations = df_buckets["expectancy"].to_numpy()

    monotonic_up = all(expectations[i] <= expectations[i + 1] for i in range(len(expectations) - 1))
    monotonic_down = all(expectations[i] >= expectations[i + 1] for i in range(len(expectations) - 1))

    pattern = "noisy"
    if monotonic_up:
        pattern = "monotonic_increasing"
    elif monotonic_down:
        pattern = "monotonic_decreasing"
    else:
        diffs = np.diff(expectations)
        if len(expectations) >= 3 and expectations[0] >= expectations[1] and expectations[-1] >= expectations[1]:
            pattern = "u_shaped"
        elif len(diffs) and np.max(np.abs(diffs)) > 0.02:
            pattern = "threshold"

    interpretation = {
        "status": "analyzed",
        "pattern": pattern,
        "num_buckets": len(df_buckets),
        "variation": float(np.std(expectations)),
    }
    return df_buckets, interpretation


def analyze_categorical_factor(df_sid: pd.DataFrame, factor_name: str, min_trades_per_bucket: int = 20):
    if factor_name not in df_sid.columns or df_sid[factor_name].isnull().all():
        return None, {"status": "missing"}

    df_factor = df_sid.dropna(subset=[factor_name]).copy()
    if len(df_factor) < min_trades_per_bucket:
        return None, {"status": "insufficient_data"}

    rows = []
    for category in df_factor[factor_name].astype(str).unique():
        cat_data = df_factor[df_factor[factor_name].astype(str) == category]
        if len(cat_data) < min_trades_per_bucket:
            continue
        returns = cat_data["return_pct"].to_numpy()
        rows.append(
            {
                "category": category,
                "trades": len(cat_data),
                "wins": int((returns > 0).sum()),
                "win_rate": float((returns > 0).mean()),
                "expectancy": float(np.mean(returns)),
                "median_return": float(np.median(returns)),
            }
        )

    if not rows:
        return None, {"status": "insufficient_bucketed_trades"}

    df_categories = pd.DataFrame(rows).sort_values("expectancy", ascending=False).reset_index(drop=True)
    interpretation = {
        "status": "analyzed",
        "num_categories": len(df_categories),
        "best_category": df_categories.iloc[0]["category"],
        "worst_category": df_categories.iloc[-1]["category"],
        "expectancy_spread": float(df_categories["expectancy"].max() - df_categories["expectancy"].min()),
    }
    return df_categories, interpretation


# =============================================================================
# STEP 3: RELEVANCE
# =============================================================================

def assess_factor_relevance(
    df_sid: pd.DataFrame,
    factor_name: str,
    baseline_expectancy: float,
    train_fraction: float,
    min_trades_per_bucket: int = 20,
) -> Dict[str, Any]:
    if factor_name in NUMERIC_FACTORS:
        factor_df, interp = analyze_numeric_factor(df_sid, factor_name, min_trades_per_bucket)
    elif factor_name in CATEGORICAL_FACTORS:
        factor_df, interp = analyze_categorical_factor(df_sid, factor_name, min_trades_per_bucket)
    else:
        return {"relevant": False, "reason": "unknown_factor_type"}

    if interp["status"] != "analyzed":
        return {"relevant": False, "reason": f"analysis_failed:{interp['status']}"}

    max_exp = factor_df["expectancy"].max()
    min_exp = factor_df["expectancy"].min()
    effect_size = abs(max_exp - min_exp)

    if effect_size < MINIMUM_EFFECT_SIZE:
        return {
            "relevant": False,
            "reason": f"effect_size_small_{effect_size:.4f}",
            "effect_size": effect_size,
        }

    if factor_name in NUMERIC_FACTORS:
        expectations = factor_df["expectancy"].to_numpy()
        signs = np.sign(np.diff(expectations))
        sign_changes = 0
        for i in range(len(signs) - 1):
            if signs[i] != 0 and signs[i + 1] != 0 and signs[i] != signs[i + 1]:
                sign_changes += 1
        stability_score = max(0.0, 1.0 - sign_changes / max(len(expectations) - 2, 1))
    else:
        stability_score = 1.0

    min_bucket_trades = int(factor_df["trades"].min())
    trade_adequacy = min(min_bucket_trades / min_trades_per_bucket, 1.0)

    # mild validation consistency check by comparing train/val top-half behavior
    df_train, df_val = split_train_validation(df_sid, train_fraction)
    if factor_name in NUMERIC_FACTORS:
        train_df, _ = analyze_numeric_factor(df_train, factor_name, max(5, min_trades_per_bucket // 2))
        val_df, _ = analyze_numeric_factor(df_val, factor_name, max(5, min_trades_per_bucket // 2))
        validation_consistency = 0.7
        if train_df is not None and val_df is not None and not train_df.empty and not val_df.empty:
            train_best = train_df["expectancy"].idxmax()
            val_best = val_df["expectancy"].idxmax()
            validation_consistency = 1.0 if abs(train_best - val_best) <= 1 else 0.6
    else:
        train_df, _ = analyze_categorical_factor(df_train, factor_name, max(5, min_trades_per_bucket // 2))
        val_df, _ = analyze_categorical_factor(df_val, factor_name, max(5, min_trades_per_bucket // 2))
        validation_consistency = 0.7
        if train_df is not None and val_df is not None and not train_df.empty and not val_df.empty:
            validation_consistency = 1.0 if str(train_df.iloc[0]["category"]) == str(val_df.iloc[0]["category"]) else 0.6

    relevance_score = (
        (effect_size / MINIMUM_EFFECT_SIZE)
        * stability_score
        * trade_adequacy
        * validation_consistency
    )

    label = "relevant"
    if relevance_score < 1.0:
        label = "weak"
    if stability_score < 0.35:
        label = "inconclusive"

    return {
        "relevant": label == "relevant",
        "label": label,
        "effect_size": effect_size,
        "stability_score": stability_score,
        "trade_adequacy": trade_adequacy,
        "validation_consistency": validation_consistency,
        "relevance_score": relevance_score,
        "pattern": interp.get("pattern", "unknown"),
        "num_buckets": interp.get("num_buckets", len(factor_df)),
        "reason": "",
    }


# =============================================================================
# STEP 4/5: PREFERRED RANGES & AVOID ZONES
# =============================================================================

def determine_preferred_and_avoid_zones(
    df_sid: pd.DataFrame,
    factor_name: str,
    baseline_expectancy: float,
    min_trades_per_bucket: int = 20,
) -> Dict[str, Any]:
    if factor_name in NUMERIC_FACTORS:
        factor_df, _ = analyze_numeric_factor(df_sid, factor_name, min_trades_per_bucket)
    elif factor_name in CATEGORICAL_FACTORS:
        factor_df, _ = analyze_categorical_factor(df_sid, factor_name, min_trades_per_bucket)
    else:
        return {"preferred": [], "avoid": []}

    if factor_df is None or factor_df.empty:
        return {"preferred": [], "avoid": []}

    preferred_ranges: List[Dict[str, Any]] = []
    avoid_zones: List[Dict[str, Any]] = []

    if factor_name in NUMERIC_FACTORS:
        good_rows = factor_df[factor_df["expectancy"] >= baseline_expectancy + PREFERRED_RANGE_UPLIFT].copy()
        bad_rows = factor_df[factor_df["expectancy"] <= baseline_expectancy - AVOID_ZONE_DOWNLIFT].copy()

        good_intervals = []
        for _, row in good_rows.iterrows():
            lo, hi = parse_bucket_bounds(row["bucket"])
            if lo is not None and hi is not None:
                good_intervals.append((lo, hi))

        bad_intervals = []
        for _, row in bad_rows.iterrows():
            lo, hi = parse_bucket_bounds(row["bucket"])
            if lo is not None and hi is not None:
                bad_intervals.append((lo, hi))

        for lo, hi in merge_intervals(good_intervals):
            block = good_rows[
                good_rows["bucket"].astype(str).apply(
                    lambda s: (
                        lambda a, b: a is not None and b is not None and a >= lo - 1e-9 and b <= hi + 1e-9
                    )(*parse_bucket_bounds(s))
                )
            ]
            preferred_ranges.append(
                {
                    "range": format_interval(lo, hi),
                    "range_min": lo,
                    "range_max": hi,
                    "expectancy": float(block["expectancy"].max()) if not block.empty else np.nan,
                    "uplift": float(block["expectancy"].max() - baseline_expectancy) if not block.empty else np.nan,
                    "trades": int(block["trades"].sum()) if not block.empty else 0,
                }
            )

        for lo, hi in merge_intervals(bad_intervals):
            block = bad_rows[
                bad_rows["bucket"].astype(str).apply(
                    lambda s: (
                        lambda a, b: a is not None and b is not None and a >= lo - 1e-9 and b <= hi + 1e-9
                    )(*parse_bucket_bounds(s))
                )
            ]
            avoid_zones.append(
                {
                    "zone": format_interval(lo, hi),
                    "zone_min": lo,
                    "zone_max": hi,
                    "expectancy": float(block["expectancy"].min()) if not block.empty else np.nan,
                    "downside": float(baseline_expectancy - block["expectancy"].min()) if not block.empty else np.nan,
                    "trades": int(block["trades"].sum()) if not block.empty else 0,
                }
            )

    else:
        good_rows = factor_df[factor_df["expectancy"] >= baseline_expectancy + PREFERRED_RANGE_UPLIFT].copy()
        bad_rows = factor_df[factor_df["expectancy"] <= baseline_expectancy - AVOID_ZONE_DOWNLIFT].copy()

        if not good_rows.empty:
            cats = good_rows["category"].astype(str).tolist()
            preferred_ranges.append(
                {
                    "category": "|".join(cats),
                    "expectancy": float(good_rows["expectancy"].max()),
                    "uplift": float(good_rows["expectancy"].max() - baseline_expectancy),
                    "trades": int(good_rows["trades"].sum()),
                }
            )

        if not bad_rows.empty:
            cats = bad_rows["category"].astype(str).tolist()
            avoid_zones.append(
                {
                    "category": "|".join(cats),
                    "expectancy": float(bad_rows["expectancy"].min()),
                    "downside": float(baseline_expectancy - bad_rows["expectancy"].min()),
                    "trades": int(bad_rows["trades"].sum()),
                }
            )

    return {"preferred": preferred_ranges, "avoid": avoid_zones}


# =============================================================================
# STEP 6/7/8: RULES
# =============================================================================

def evaluate_conditions(conditions: List[RuleCondition], df_data: pd.DataFrame) -> Optional[Dict[str, Any]]:
    mask = pd.Series(True, index=df_data.index)

    for cond in conditions:
        if cond.factor not in df_data.columns:
            return None

        if cond.cond_type == "numeric":
            if cond.operator == ">=":
                mask &= df_data[cond.factor] >= cond.value
            elif cond.operator == "<=":
                mask &= df_data[cond.factor] <= cond.value
            elif cond.operator == ">":
                mask &= df_data[cond.factor] > cond.value
            elif cond.operator == "<":
                mask &= df_data[cond.factor] < cond.value
            else:
                return None
        else:
            vals = cond.value if isinstance(cond.value, list) else [cond.value]
            mask &= df_data[cond.factor].astype(str).isin([str(v) for v in vals])

    sub = df_data[mask].copy()
    if sub.empty:
        return None

    returns = sub["return_pct"].to_numpy()
    return {
        "trades": len(sub),
        "wins": int((returns > 0).sum()),
        "win_rate": float((returns > 0).mean()),
        "expectancy": float(np.mean(returns)),
        "median_return": float(np.median(returns)),
    }


def build_seed_conditions(
    relevant_factors: List[str],
    preferred_avoid: Dict[str, Dict[str, Any]],
    df_sid: pd.DataFrame,
    min_trades_per_bucket: int,
) -> Dict[str, List[RuleCondition]]:
    seeds: Dict[str, List[RuleCondition]] = {}

    for factor in relevant_factors:
        factor_seeds: List[RuleCondition] = []
        zones = preferred_avoid.get(factor, {"preferred": [], "avoid": []})

        if factor in NUMERIC_FACTORS:
            for pref in zones.get("preferred", []):
                min_v = pref.get("range_min")
                max_v = pref.get("range_max")
                if min_v is not None:
                    factor_seeds.append(RuleCondition(factor, "numeric", ">=", float(min_v)))
                if max_v is not None:
                    factor_seeds.append(RuleCondition(factor, "numeric", "<=", float(max_v)))

            # Convert avoid zones into opposite-style thresholds.
            for avoid in zones.get("avoid", []):
                min_v = avoid.get("zone_min")
                max_v = avoid.get("zone_max")
                if min_v is not None and max_v is not None:
                    factor_seeds.append(RuleCondition(factor, "numeric", "<=", float(min_v)))
                    factor_seeds.append(RuleCondition(factor, "numeric", ">=", float(max_v)))

            # If preferred/avoid still yields nothing, backfill from top expectancy buckets.
            if not factor_seeds:
                buckets_df, interp = analyze_numeric_factor(df_sid, factor, min_trades_per_bucket)
                if interp.get("status") == "analyzed" and buckets_df is not None and not buckets_df.empty:
                    top_buckets = buckets_df.sort_values("expectancy", ascending=False).head(2)
                    for _, row in top_buckets.iterrows():
                        lo, hi = parse_bucket_bounds(row["bucket"])
                        if lo is not None:
                            factor_seeds.append(RuleCondition(factor, "numeric", ">=", float(lo)))
                        if hi is not None:
                            factor_seeds.append(RuleCondition(factor, "numeric", "<=", float(hi)))

        elif factor in CATEGORICAL_FACTORS:
            for pref in zones.get("preferred", []):
                cats = str(pref.get("category", "")).split("|")
                cats = [c.strip() for c in cats if c.strip()]
                if cats:
                    factor_seeds.append(RuleCondition(factor, "categorical", "in", cats))

            # Backfill categorical seeds from top expectancy categories if preferred is empty.
            if not factor_seeds:
                cat_df, interp = analyze_categorical_factor(df_sid, factor, min_trades_per_bucket)
                if interp.get("status") == "analyzed" and cat_df is not None and not cat_df.empty:
                    top_cats = cat_df.sort_values("expectancy", ascending=False).head(2)
                    for _, row in top_cats.iterrows():
                        factor_seeds.append(RuleCondition(factor, "categorical", "in", [str(row["category"])]))

        if factor_seeds:
            dedup = {}
            for s in factor_seeds:
                dedup[(s.factor, s.cond_type, s.operator, str(s.value))] = s
            seeds[factor] = list(dedup.values())[:MAX_SEED_CONDITIONS_PER_FACTOR]

    return seeds


def mine_candidate_rules(
    relevant_factors: List[str],
    preferred_avoid: Dict[str, Dict[str, Any]],
    df_sid: pd.DataFrame,
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    baseline_stats: Dict[str, Any],
    min_trades_per_bucket: int,
    min_trades_per_rule: int = 15,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, List[RuleCondition]]]:
    seeds = build_seed_conditions(
        relevant_factors=relevant_factors,
        preferred_avoid=preferred_avoid,
        df_sid=df_sid,
        min_trades_per_bucket=min_trades_per_bucket,
    )
    available = [f for f in relevant_factors if f in seeds and len(seeds[f]) > 0]
    candidates: List[Dict[str, Any]] = []

    combos_attempted = {1: 0, 2: 0, 3: 0}
    filtered_train = 0
    filtered_val = 0

    for r in (1, 2, 3):
        if len(available) < r:
            continue
        for factor_combo in combinations(available, r):
            cond_options = [seeds[f] for f in factor_combo]
            for conds in product(*cond_options):
                combos_attempted[r] += 1
                cond_list = list(conds)

                # keep at most one condition per factor/operator family
                factors_used = [c.factor for c in cond_list]
                if len(set(factors_used)) != len(factors_used):
                    continue

                train_eval = evaluate_conditions(cond_list, df_train)
                val_eval = evaluate_conditions(cond_list, df_val)

                if train_eval is None or val_eval is None:
                    continue
                if train_eval["trades"] < min_trades_per_rule:
                    filtered_train += 1
                    continue
                if val_eval["trades"] < min_trades_per_rule:
                    filtered_val += 1
                    continue

                candidates.append(
                    {
                        "num_factors": len(cond_list),
                        "conditions": cond_list,
                        "train_eval": train_eval,
                        "val_eval": val_eval,
                        "baseline_expectancy": baseline_stats["expectancy"],
                        "baseline_median": baseline_stats["median_return"],
                    }
                )

    debug_info = {
        "relevant_factor_count": len(relevant_factors),
        "available_factor_count": len(available),
        "seed_counts": {f: len(seeds.get(f, [])) for f in relevant_factors},
        "rule_combos_attempted_1f": combos_attempted[1],
        "rule_combos_attempted_2f": combos_attempted[2],
        "rule_combos_attempted_3f": combos_attempted[3],
        "rule_combos_attempted_total": combos_attempted[1] + combos_attempted[2] + combos_attempted[3],
        "filtered_insufficient_train": filtered_train,
        "filtered_insufficient_validation": filtered_val,
        "candidates_survived": len(candidates),
    }

    return candidates, debug_info, seeds


def score_and_rank_rules(
    candidates: List[Dict[str, Any]],
    min_trades_per_rule: int,
) -> List[Dict[str, Any]]:
    ranked = []
    for candidate in candidates:
        val_eval = candidate["val_eval"]
        train_eval = candidate["train_eval"]
        baseline_exp = candidate["baseline_expectancy"]
        baseline_med = candidate["baseline_median"]

        val_uplift = val_eval["expectancy"] - baseline_exp
        exp_score = max(0.0, val_uplift / MIN_RULE_UPLIFT)
        exp_score = min(exp_score, 3.0)

        trade_score = min(val_eval["trades"] / 100.0, 1.0)

        drift = abs(train_eval["expectancy"] - val_eval["expectancy"])
        stability = max(0.0, 1.0 - drift / MINIMUM_DRIFT_TOLERANCE)

        median_uplift = val_eval["median_return"] - baseline_med
        median_score = max(0.0, median_uplift / 0.005)
        median_score = min(median_score, 1.0)

        simplicity_bonus = 1.05 if candidate["num_factors"] == 2 else 1.00

        composite_score = (
            RULE_SCORING_WEIGHTS["validation_expectancy"] * exp_score
            + RULE_SCORING_WEIGHTS["trade_count"] * trade_score
            + RULE_SCORING_WEIGHTS["train_val_stability"] * stability
            + RULE_SCORING_WEIGHTS["median_improvement"] * median_score
            + RULE_SCORING_WEIGHTS["simplicity_bonus"] * simplicity_bonus
        )

        candidate["composite_score"] = composite_score
        candidate["exp_uplift_pct"] = val_uplift
        candidate["median_uplift_pct"] = median_uplift
        candidate["drift_pct"] = drift
        candidate["promoted"] = (
            val_uplift >= MIN_RULE_UPLIFT
            and val_eval["trades"] >= min_trades_per_rule
            and drift <= MINIMUM_DRIFT_TOLERANCE
            and val_eval["median_return"] >= baseline_med
        )
        ranked.append(candidate)

    return sorted(ranked, key=lambda x: x["composite_score"], reverse=True)


def _rule_dedup_key(rule: Dict[str, Any]) -> Tuple[str, ...]:
    conditions = sorted(
        rule["conditions"],
        key=lambda c: (c.factor, c.cond_type, c.operator, str(c.value)),
    )
    return tuple(condition.factor for condition in conditions)


def _rules_are_near_duplicates(
    left_rule: Dict[str, Any],
    right_rule: Dict[str, Any],
    numeric_tolerance: float = 0.05,
) -> bool:
    left_conditions = sorted(
        left_rule["conditions"],
        key=lambda c: (c.factor, c.cond_type, c.operator, str(c.value)),
    )
    right_conditions = sorted(
        right_rule["conditions"],
        key=lambda c: (c.factor, c.cond_type, c.operator, str(c.value)),
    )

    if len(left_conditions) != len(right_conditions):
        return False

    for left_cond, right_cond in zip(left_conditions, right_conditions):
        if left_cond.factor != right_cond.factor:
            return False
        if left_cond.cond_type != right_cond.cond_type:
            return False
        if left_cond.operator != right_cond.operator:
            return False

        if left_cond.cond_type == "categorical":
            left_vals = left_cond.value if isinstance(left_cond.value, list) else [left_cond.value]
            right_vals = right_cond.value if isinstance(right_cond.value, list) else [right_cond.value]
            if sorted(map(str, left_vals)) != sorted(map(str, right_vals)):
                return False
            continue

        left_value = safe_float(left_cond.value)
        right_value = safe_float(right_cond.value)
        if left_value is None or right_value is None:
            return False
        if abs(left_value - right_value) > numeric_tolerance:
            return False

    return True


def select_deployable_rules(
    ranked_candidates: List[Dict[str, Any]],
    max_rules_per_sid: int,
    sid: int,
) -> List[Dict[str, Any]]:
    filtered_rules = [
        rule for rule in ranked_candidates
        if rule["val_eval"]["trades"] >= 15
        and rule["drift_pct"] <= 0.02
        and rule["median_uplift_pct"] >= 0
    ]

    filtered_rules = sorted(
        filtered_rules,
        key=lambda rule: (
            rule["num_factors"],
            -rule["val_eval"]["trades"],
            rule["drift_pct"],
            -rule["exp_uplift_pct"],
        ),
    )

    deduped_rules: List[Dict[str, Any]] = []
    for rule in filtered_rules:
        is_duplicate = False
        for kept_rule in deduped_rules:
            if _rule_dedup_key(rule) != _rule_dedup_key(kept_rule):
                continue
            if _rules_are_near_duplicates(rule, kept_rule):
                is_duplicate = True
                break
        if not is_duplicate:
            deduped_rules.append(rule)

    sid_cap = 2 if sid == 5 else 3
    final_cap = min(max_rules_per_sid, sid_cap)
    return deduped_rules[:final_cap]


# =============================================================================
# OUTPUTS
# =============================================================================

def generate_per_sid_report(
    sid: int,
    baseline_stats: Dict[str, Any],
    relevance_assessment: Dict[str, Dict[str, Any]],
    preferred_avoid: Dict[str, Dict[str, Any]],
    candidate_rules: List[Dict[str, Any]],
    deployable_rules: List[Dict[str, Any]],
    output_dir: Path,
):
    report_path = output_dir / f"report_sid_{sid}.txt"
    with open(report_path, "w") as f:
        f.write("\n" + "=" * 70 + "\n")
        f.write(f"SETUPID {sid} - BASELINE PROFILE REPORT\n")
        f.write("=" * 70 + "\n\n")

        f.write("BASELINE METRICS\n")
        f.write("-" * 70 + "\n")
        f.write(f"Total Trades:        {baseline_stats['trades']:,}\n")
        f.write(f"Wins:                {baseline_stats['wins']:,}\n")
        f.write(f"Win Rate:            {baseline_stats['win_rate']:.1%}\n")
        f.write(f"Expectancy:          {baseline_stats['expectancy']*100:.2f}%\n")
        f.write(f"Median Return:       {baseline_stats['median_return']*100:.2f}%\n")
        f.write(f"Average Hold Days:   {baseline_stats['avg_hold_days']:.1f}\n")
        f.write(f"Payoff Ratio:        {baseline_stats['payoff_ratio']:.2f}\n")
        if baseline_stats["runner_rate"] is not None:
            f.write(f"Runner Rate:         {baseline_stats['runner_rate']:.1%}\n")

        f.write("\nFACTOR RELEVANCE\n")
        f.write("-" * 70 + "\n")
        for factor, rel in relevance_assessment.items():
            status = rel.get("label", "not_relevant").upper()
            f.write(f"{factor}: {status}")
            if rel.get("relevant"):
                f.write(
                    f" | effect={rel['effect_size']:.4f} | score={rel['relevance_score']:.2f} "
                    f"| pattern={rel['pattern']}\n"
                )
            else:
                f.write(f" | reason={rel.get('reason', '')}\n")

        f.write("\nPREFERRED RANGES / AVOID ZONES\n")
        f.write("-" * 70 + "\n")
        for factor, zones in preferred_avoid.items():
            if not zones["preferred"] and not zones["avoid"]:
                continue
            f.write(f"{factor}:\n")
            if zones["preferred"]:
                f.write("  Preferred:\n")
                for z in zones["preferred"]:
                    if factor in NUMERIC_FACTORS:
                        f.write(
                            f"    {z['range']} | uplift=+{z['uplift']*100:.2f}% | trades={z['trades']}\n"
                        )
                    else:
                        f.write(
                            f"    {z['category']} | uplift=+{z['uplift']*100:.2f}% | trades={z['trades']}\n"
                        )
            if zones["avoid"]:
                f.write("  Avoid:\n")
                for z in zones["avoid"]:
                    if factor in NUMERIC_FACTORS:
                        f.write(
                            f"    {z['zone']} | downside={z['downside']*100:.2f}% | trades={z['trades']}\n"
                        )
                    else:
                        f.write(
                            f"    {z['category']} | downside={z['downside']*100:.2f}% | trades={z['trades']}\n"
                        )

        f.write("\nCANDIDATE RULES\n")
        f.write("-" * 70 + "\n")
        if candidate_rules:
            for i, rule in enumerate(candidate_rules[:20], 1):
                f.write(
                    f"{i}. {conditions_to_rule_string(rule['conditions'])} | "
                    f"val_exp={rule['val_eval']['expectancy']*100:.2f}% | "
                    f"uplift=+{rule['exp_uplift_pct']*100:.2f}% | "
                    f"trades={rule['val_eval']['trades']} | "
                    f"drift={rule['drift_pct']*100:.2f}% | "
                    f"score={rule['composite_score']:.2f}\n"
                )
        else:
            f.write("No candidate rules.\n")

        f.write("\nDEPLOYABLE RULES\n")
        f.write("-" * 70 + "\n")
        if deployable_rules:
            for i, rule in enumerate(deployable_rules, 1):
                f.write(
                    f"{i}. {conditions_to_rule_string(rule['conditions'])} | "
                    f"val_exp={rule['val_eval']['expectancy']*100:.2f}% | "
                    f"uplift=+{rule['exp_uplift_pct']*100:.2f}% | "
                    f"median_uplift=+{rule['median_uplift_pct']*100:.2f}% | "
                    f"trades={rule['val_eval']['trades']} | "
                    f"drift={rule['drift_pct']*100:.2f}% | "
                    f"score={rule['composite_score']:.2f}\n"
                )
        else:
            f.write("No deployable rules.\n")

    print(f"✓ Report saved: {report_path}")


def generate_overall_summary(summary_dict: Dict[int, Dict[str, Any]], output_dir: Path):
    summary_path = output_dir / "SUMMARY_OVERALL.txt"
    with open(summary_path, "w") as f:
        f.write("\n" + "=" * 70 + "\n")
        f.write("SETUPID BASELINE PROFILE ENGINE - OVERALL SUMMARY\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("BASELINE COMPARISON\n")
        f.write("-" * 70 + "\n\n")
        for sid in sorted(summary_dict.keys()):
            b = summary_dict[sid]["baseline"]
            f.write(
                f"SID {sid}: trades={b['trades']:,}, expectancy={b['expectancy']*100:.2f}%, "
                f"win_rate={b['win_rate']:.1%}, payoff={b['payoff_ratio']:.2f}\n"
            )

        f.write("\nRELEVANT FACTORS BY SETUPID\n")
        f.write("-" * 70 + "\n\n")
        for sid in sorted(summary_dict.keys()):
            rel = summary_dict[sid]["relevance"]
            relevant = [f for f, v in rel.items() if v.get("relevant")]
            f.write(f"SID {sid}: {', '.join(relevant) if relevant else 'None'}\n")

        f.write("\nDEPLOYABLE RULES SUMMARY\n")
        f.write("-" * 70 + "\n\n")
        for sid in sorted(summary_dict.keys()):
            rules = summary_dict[sid]["deployable_rules"]
            f.write(f"SID {sid}: {len(rules)} deployable rule(s)\n")
            for i, rule in enumerate(rules, 1):
                f.write(
                    f"  {i}. {conditions_to_rule_string(rule['conditions'])} | "
                    f"val_exp={rule['val_eval']['expectancy']*100:.2f}% | "
                    f"trades={rule['val_eval']['trades']} | score={rule['composite_score']:.2f}\n"
                )

    print(f"✓ Overall summary saved: {summary_path}")


def generate_pine_ready_recommendations(summary_dict: Dict[int, Dict[str, Any]], output_dir: Path):
    pine_path = output_dir / "PINE_READY_RECOMMENDATIONS.txt"
    with open(pine_path, "w") as f:
        f.write("// SetupID Baseline Profile Engine - Pine Script Recommendations\n")
        f.write(f"// Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        for sid in sorted(summary_dict.keys()):
            rules = summary_dict[sid]["deployable_rules"]
            f.write(f"// ===== SID {sid} =====\n")
            if not rules:
                f.write(f"// No deployable rules for SID {sid}\n\n")
                continue
            for i, rule in enumerate(rules, 1):
                cond_text = conditions_to_rule_string(rule["conditions"])
                f.write(f"// Rule {i}: {cond_text}\n")
                f.write(f"// Validation expectancy: {rule['val_eval']['expectancy']*100:.2f}%\n")
                f.write(f"// Validation trades: {rule['val_eval']['trades']}\n")
                f.write(f"// Score: {rule['composite_score']:.2f}\n\n")
    print(f"✓ Pine-ready recommendations saved: {pine_path}")


def save_csv_outputs(summary_dict: Dict[int, Dict[str, Any]], output_dir: Path):
    # baseline_by_sid.csv
    baseline_rows = []
    relevance_rows = []
    preferred_rows = []
    avoid_rows = []
    candidate_rows = []
    deployable_rows = []
    debug_rows = []

    for sid in sorted(summary_dict.keys()):
        baseline = summary_dict[sid]["baseline"]
        baseline_rows.append(
            {
                "SID": sid,
                "Trades": baseline["trades"],
                "Wins": baseline["wins"],
                "Win_Rate": baseline["win_rate"],
                "Expectancy_Pct": baseline["expectancy"] * 100,
                "Median_Return_Pct": baseline["median_return"] * 100,
                "Avg_Hold_Days": baseline["avg_hold_days"],
                "Payoff_Ratio": baseline["payoff_ratio"],
                "Runner_Rate": baseline["runner_rate"],
            }
        )

        for factor, rel in summary_dict[sid]["relevance"].items():
            relevance_rows.append(
                {
                    "SID": sid,
                    "Factor": factor,
                    "Relevant": "Yes" if rel.get("relevant") else "No",
                    "Label": rel.get("label"),
                    "Effect_Size": rel.get("effect_size"),
                    "Relevance_Score": rel.get("relevance_score"),
                    "Pattern": rel.get("pattern"),
                    "Reason": rel.get("reason"),
                }
            )

        for factor, zones in summary_dict[sid]["preferred_avoid"].items():
            for z in zones.get("preferred", []):
                preferred_rows.append(
                    {
                        "SID": sid,
                        "Factor": factor,
                        "Range": z.get("range"),
                        "Range_Min": z.get("range_min"),
                        "Range_Max": z.get("range_max"),
                        "Category": z.get("category"),
                        "Expectancy_Pct": z.get("expectancy", np.nan) * 100 if z.get("expectancy") is not None else np.nan,
                        "Uplift_Pct": z.get("uplift", np.nan) * 100 if z.get("uplift") is not None else np.nan,
                        "Trades": z.get("trades"),
                    }
                )
            for z in zones.get("avoid", []):
                avoid_rows.append(
                    {
                        "SID": sid,
                        "Factor": factor,
                        "Zone": z.get("zone"),
                        "Zone_Min": z.get("zone_min"),
                        "Zone_Max": z.get("zone_max"),
                        "Category": z.get("category"),
                        "Expectancy_Pct": z.get("expectancy", np.nan) * 100 if z.get("expectancy") is not None else np.nan,
                        "Downside_Pct": z.get("downside", np.nan) * 100 if z.get("downside") is not None else np.nan,
                        "Trades": z.get("trades"),
                    }
                )

        for i, rule in enumerate(summary_dict[sid]["candidate_rules"], 1):
            candidate_rows.append(
                {
                    "SID": sid,
                    "Rule_Number": i,
                    "Rule": conditions_to_rule_string(rule["conditions"]),
                    "Num_Factors": rule["num_factors"],
                    "Composite_Score": rule["composite_score"],
                    "Val_Expectancy_Pct": rule["val_eval"]["expectancy"] * 100,
                    "Val_Win_Rate": rule["val_eval"]["win_rate"],
                    "Val_Trades": rule["val_eval"]["trades"],
                    "Train_Val_Drift_Pct": rule["drift_pct"] * 100,
                    "Expectancy_Uplift_Pct": rule["exp_uplift_pct"] * 100,
                    "Median_Uplift_Pct": rule["median_uplift_pct"] * 100,
                    "Promoted": rule["promoted"],
                }
            )

        for i, rule in enumerate(summary_dict[sid]["deployable_rules"], 1):
            deployable_rows.append(
                {
                    "SID": sid,
                    "Rule_Number": i,
                    "Rule": conditions_to_rule_string(rule["conditions"]),
                    "Num_Factors": rule["num_factors"],
                    "Composite_Score": rule["composite_score"],
                    "Val_Expectancy_Pct": rule["val_eval"]["expectancy"] * 100,
                    "Val_Win_Rate": rule["val_eval"]["win_rate"],
                    "Val_Trades": rule["val_eval"]["trades"],
                    "Train_Val_Drift_Pct": rule["drift_pct"] * 100,
                    "Expectancy_Uplift_Pct": rule["exp_uplift_pct"] * 100,
                    "Median_Uplift_Pct": rule["median_uplift_pct"] * 100,
                }
            )

        debug = summary_dict[sid].get("debug", {})
        debug_row = {
            "SID": sid,
            "Relevant_Factors": debug.get("relevant_factor_count", 0),
            "Preferred_Ranges": debug.get("preferred_range_count", 0),
            "Avoid_Zones": debug.get("avoid_zone_count", 0),
            "Rule_Combos_1F": debug.get("rule_combos_attempted_1f", 0),
            "Rule_Combos_2F": debug.get("rule_combos_attempted_2f", 0),
            "Rule_Combos_3F": debug.get("rule_combos_attempted_3f", 0),
            "Rule_Combos_Total": debug.get("rule_combos_attempted_total", 0),
            "Filtered_Insufficient_Train": debug.get("filtered_insufficient_train", 0),
            "Filtered_Insufficient_Validation": debug.get("filtered_insufficient_validation", 0),
            "Candidates_Survived": debug.get("candidates_survived", 0),
            "Candidate_Rules": debug.get("candidate_rule_count", len(summary_dict[sid]["candidate_rules"])),
            "Deployable_Rules": debug.get("deployable_rule_count", len(summary_dict[sid]["deployable_rules"])),
        }
        seed_counts = debug.get("seed_counts", {})
        for factor in PRIMARY_FACTORS:
            debug_row[f"Seed_{factor}"] = int(seed_counts.get(factor, 0))
        debug_rows.append(debug_row)

    pd.DataFrame(baseline_rows).to_csv(output_dir / "baseline_by_sid.csv", index=False)
    pd.DataFrame(relevance_rows).to_csv(output_dir / "factor_relevance_by_sid.csv", index=False)
    pd.DataFrame(preferred_rows).to_csv(output_dir / "preferred_ranges_by_sid.csv", index=False)
    pd.DataFrame(avoid_rows).to_csv(output_dir / "avoid_zones_by_sid.csv", index=False)
    pd.DataFrame(candidate_rows).to_csv(output_dir / "candidate_rules_by_sid.csv", index=False)
    pd.DataFrame(deployable_rows).to_csv(output_dir / "deployable_rules_by_sid.csv", index=False)
    pd.DataFrame(debug_rows).to_csv(output_dir / "debug_rule_pipeline_by_sid.csv", index=False)

    print("✓ Saved: baseline_by_sid.csv")
    print("✓ Saved: factor_relevance_by_sid.csv")
    print("✓ Saved: preferred_ranges_by_sid.csv")
    print("✓ Saved: avoid_zones_by_sid.csv")
    print("✓ Saved: candidate_rules_by_sid.csv")
    print("✓ Saved: deployable_rules_by_sid.csv")
    print("✓ Saved: debug_rule_pipeline_by_sid.csv")


# =============================================================================
# CHARTS
# =============================================================================

def create_response_curve_chart(sid: int, factor_name: str, buckets_df: pd.DataFrame, baseline_expectancy: float, output_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"SID {sid} - Factor: {factor_name}", fontsize=14, fontweight="bold")

    bucket_labels = [str(b) for b in buckets_df.iloc[:, 0].astype(str)]

    ax1 = axes[0]
    ax1.bar(range(len(buckets_df)), buckets_df["expectancy"] * 100)
    ax1.axhline(baseline_expectancy * 100, linestyle="--", linewidth=2)
    ax1.set_ylabel("Expectancy (%)")
    ax1.set_xlabel("Bucket/Category")
    ax1.set_title("Expectancy")
    ax1.set_xticks(range(len(buckets_df)))
    ax1.set_xticklabels(bucket_labels, rotation=45, ha="right")
    ax1.grid(axis="y", alpha=0.3)

    ax2 = axes[1]
    ax2_twin = ax2.twinx()
    ax2.bar(range(len(buckets_df)), buckets_df["trades"], alpha=0.6)
    ax2_twin.plot(range(len(buckets_df)), buckets_df["win_rate"] * 100, marker="o", linewidth=2)
    ax2.set_ylabel("Trade Count")
    ax2_twin.set_ylabel("Win Rate (%)")
    ax2.set_xlabel("Bucket/Category")
    ax2.set_title("Trade Count & Win Rate")
    ax2.set_xticks(range(len(buckets_df)))
    ax2.set_xticklabels(bucket_labels, rotation=45, ha="right")
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    chart_path = output_dir / f"chart_sid{sid}_{factor_name}.png"
    plt.savefig(chart_path, dpi=100, facecolor="white", edgecolor="none")
    plt.close()
    return chart_path


# =============================================================================
# MAIN
# =============================================================================

def main():
    args = parse_arguments()

    print("\n" + "=" * 70)
    print("SetupID Baseline Profile Engine")
    print("=" * 70)

    df = load_and_validate_data(args.input, args.min_trades_per_sid)
    if df is None:
        print("\nERROR: Data loading failed. Exiting.")
        sys.exit(1)

    output_dir = ensure_output_directory(args.output)
    print(f"\nOutput directory: {output_dir}\n")

    summary_dict: Dict[int, Dict[str, Any]] = {}

    for sid in sorted(df["sid"].dropna().unique()):
        print("\n" + "=" * 70)
        print(f"ANALYZING SID {sid}")
        print("=" * 70)

        df_sid = df[df["sid"] == sid].copy()
        baseline_stats = compute_baseline_stats(df_sid)

        print(f"Total trades: {baseline_stats['trades']:,}")
        print(f"Baseline Expectancy: {baseline_stats['expectancy']*100:.2f}%")
        print(f"Baseline Win Rate:   {baseline_stats['win_rate']:.1%}")

        relevance_assessment: Dict[str, Dict[str, Any]] = {}
        factor_analysis: Dict[str, Optional[pd.DataFrame]] = {}

        print("\nFactor Relevance Assessment:")
        for factor in PRIMARY_FACTORS:
            if factor not in df_sid.columns:
                relevance_assessment[factor] = {"relevant": False, "label": "not_relevant", "reason": "not_in_dataset"}
                print(f"  {factor}: NOT IN DATASET")
                factor_analysis[factor] = None
                continue

            rel = assess_factor_relevance(
                df_sid,
                factor,
                baseline_stats["expectancy"],
                args.train_fraction,
                args.min_trades_per_bucket,
            )
            relevance_assessment[factor] = rel

            if factor in NUMERIC_FACTORS:
                table, _ = analyze_numeric_factor(df_sid, factor, args.min_trades_per_bucket)
            else:
                table, _ = analyze_categorical_factor(df_sid, factor, args.min_trades_per_bucket)
            factor_analysis[factor] = table

            status = rel.get("label", "not_relevant").upper()
            if rel.get("relevant"):
                print(f"  {factor}: {status} (score={rel['relevance_score']:.2f}, pattern={rel['pattern']})")
            else:
                print(f"  {factor}: {status} ({rel.get('reason', '')})")

        print("\nIdentifying Preferred Ranges & Avoid Zones:")
        preferred_avoid: Dict[str, Dict[str, Any]] = {}
        for factor in PRIMARY_FACTORS:
            if factor not in df_sid.columns:
                continue
            if not relevance_assessment[factor].get("relevant"):
                preferred_avoid[factor] = {"preferred": [], "avoid": []}
                continue

            zones = determine_preferred_and_avoid_zones(
                df_sid=df_sid,
                factor_name=factor,
                baseline_expectancy=baseline_stats["expectancy"],
                min_trades_per_bucket=args.min_trades_per_bucket,
            )
            preferred_avoid[factor] = zones
            print(f"  {factor}: {len(zones['preferred'])} preferred, {len(zones['avoid'])} avoid")

        print("\nMining and Validating Rules:")
        df_train, df_val = split_train_validation(df_sid, args.train_fraction)
        print(f"  Train: {len(df_train):,} trades | Val: {len(df_val):,} trades")

        relevant_factors = [f for f, rel in relevance_assessment.items() if rel.get("relevant")]
        preferred_count = sum(len(z.get("preferred", [])) for z in preferred_avoid.values())
        avoid_count = sum(len(z.get("avoid", [])) for z in preferred_avoid.values())
        print(f"  Relevant factors found: {len(relevant_factors)}")
        print(f"  Preferred ranges found: {preferred_count}")
        print(f"  Avoid zones found: {avoid_count}")
        print(f"  Relevant factors for rule mining: {', '.join(relevant_factors) if relevant_factors else 'None'}")

        if len(relevant_factors) >= 1:
            candidates, debug_info, seeds = mine_candidate_rules(
                relevant_factors=relevant_factors,
                preferred_avoid=preferred_avoid,
                df_sid=df_sid,
                df_train=df_train,
                df_val=df_val,
                baseline_stats=baseline_stats,
                min_trades_per_bucket=args.min_trades_per_bucket,
                min_trades_per_rule=args.min_trades_per_rule,
            )
            debug_info["preferred_range_count"] = preferred_count
            debug_info["avoid_zone_count"] = avoid_count

            print("  Seed conditions by factor:")
            for factor in relevant_factors:
                print(f"    {factor}: {len(seeds.get(factor, []))}")

            print(
                "  Rule combinations attempted: "
                f"1F={debug_info['rule_combos_attempted_1f']}, "
                f"2F={debug_info['rule_combos_attempted_2f']}, "
                f"3F={debug_info['rule_combos_attempted_3f']}"
            )
            print(f"  Filtered (insufficient train trades): {debug_info['filtered_insufficient_train']}")
            print(f"  Filtered (insufficient validation trades): {debug_info['filtered_insufficient_validation']}")
            print(f"  Survived into scoring: {debug_info['candidates_survived']}")

            ranked_candidates = score_and_rank_rules(
                candidates,
                args.min_trades_per_rule
            )
            deployable_rules = select_deployable_rules(
                ranked_candidates,
                args.max_promoted_rules,
                sid,
            )
            debug_info["candidate_rule_count"] = len(ranked_candidates)
            debug_info["deployable_rule_count"] = len(deployable_rules)
            print(f"  Candidate rule count: {len(ranked_candidates)}")
            print(f"  Deployable rule count: {len(deployable_rules)}")
        else:
            ranked_candidates = []
            deployable_rules = []
            debug_info = {
                "relevant_factor_count": 0,
                "available_factor_count": 0,
                "seed_counts": {},
                "preferred_range_count": preferred_count,
                "avoid_zone_count": avoid_count,
                "rule_combos_attempted_1f": 0,
                "rule_combos_attempted_2f": 0,
                "rule_combos_attempted_3f": 0,
                "rule_combos_attempted_total": 0,
                "filtered_insufficient_train": 0,
                "filtered_insufficient_validation": 0,
                "candidates_survived": 0,
                "candidate_rule_count": 0,
                "deployable_rule_count": 0,
            }
            print("  Insufficient relevant factors for rule mining")

        for i, rule in enumerate(deployable_rules[:3], 1):
            print(
                f"    {i}. {conditions_to_rule_string(rule['conditions'])} "
                f"| val_exp={rule['val_eval']['expectancy']*100:.2f}% "
                f"| uplift=+{rule['exp_uplift_pct']*100:.2f}%"
            )

        summary_dict[sid] = {
            "baseline": baseline_stats,
            "relevance": relevance_assessment,
            "preferred_avoid": preferred_avoid,
            "candidate_rules": ranked_candidates,
            "deployable_rules": deployable_rules,
            "debug": debug_info,
        }

        generate_per_sid_report(
            sid=sid,
            baseline_stats=baseline_stats,
            relevance_assessment=relevance_assessment,
            preferred_avoid=preferred_avoid,
            candidate_rules=ranked_candidates,
            deployable_rules=deployable_rules,
            output_dir=output_dir,
        )

        if args.save_charts:
            print("\nGenerating charts:")
            for factor in relevant_factors:
                table = factor_analysis.get(factor)
                if table is not None and not table.empty:
                    chart_file = create_response_curve_chart(sid, factor, table, baseline_stats["expectancy"], output_dir)
                    print(f"  ✓ Chart: {chart_file.name}")

    print("\n" + "=" * 70)
    print("GENERATING SUMMARY OUTPUTS")
    print("=" * 70 + "\n")

    generate_overall_summary(summary_dict, output_dir)
    generate_pine_ready_recommendations(summary_dict, output_dir)
    save_csv_outputs(summary_dict, output_dir)

    print("\n" + "=" * 70)
    print("RESEARCH COMPLETE")
    print("=" * 70 + "\n")
    print(f"All outputs saved to: {output_dir}\n")

    print("Output files:")
    for file in sorted(output_dir.glob("*")):
        if file.is_file():
            size = file.stat().st_size
            size_str = f"{size:,} bytes" if size < 1024 * 1024 else f"{size/(1024*1024):.1f} MB"
            print(f"  {file.name:<50} {size_str:>15}")


if __name__ == "__main__":
    main()