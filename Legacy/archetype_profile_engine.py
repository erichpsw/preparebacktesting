#!/usr/bin/env python3
"""
Archetype Profile Engine (Stage 3)
==================================

Purpose:
  - Extend SetupID baseline profiles into archetype-specific adjustments
  - Learn whether different stock archetypes require factor preference changes
  - Generate archetype adjustments anchored to baseline, not independent systems
  - Validate archetype rules and adjustments on held-out data

Core Principle:
    Adjustments are relative to the validated Stage 2 baseline, not replacements for it.
  Stage 2 is designed to discover when a specific stock archetype justifies tighter,
  looser, or category-specific changes to the baseline profile, while preserving the
  original SetupID structure and avoiding overfitting.

What this script does:
  1) Loads the clean campaign-level trade dataset
  2) Loads Stage 2 baseline outputs:
     - baseline_by_sid.csv
     - factor_relevance_by_sid.csv
     - preferred_ranges_by_sid.csv
     - avoid_zones_by_sid.csv
      - deployable_rules_by_sid.csv
  3) Reconstructs a baseline profile for each SetupID from Stage 2 research
  4) Builds stock archetype dimensions using:
     - DDV liquidity buckets
     - RMV volatility buckets
     - RS/TR leadership buckets
     - optional composite archetypes
  5) Splits each archetype chronologically into train / validation sets
  6) Measures whether factor relevance, preferred ranges, and avoid zones shift
     inside each archetype versus the baseline profile
  7) Proposes archetype-specific baseline-relative adjustments
  8) Mines small archetype-specific rules using only baseline-consistent seed conditions
  9) Promotes only adjustments and rules that survive validation with sufficient sample size
 10) Exports archetype research outputs for downstream deployment in Stage 3

Key output files:
  - archetype_assignments.csv
  - archetype_baseline_by_sid.csv
  - archetype_relevance_by_sid.csv
  - archetype_preferred_ranges_by_sid.csv
  - archetype_avoid_zones_by_sid.csv
  - archetype_adjustments_by_sid.csv
  - archetype_candidate_rules_by_sid.csv
    - archetype_deployable_rules_by_sid.csv
  - report_sid_<n>_archetypes.txt
  - SUMMARY_ARCHETYPES.txt

Usage:
python archetype_profile_engine.py \
  --input campaigns_clean.csv \
  --baseline-dir results_profiles \
  --output results_archetypes \
  --train-fraction 0.70 \
  --min-trades-per-sid 80 \
  --min-trades-per-archetype 40 \
  --min-trades-per-bucket 15 \
  --min-trades-per-rule 12 \
  --max-promoted-rules 5 \
  --ddv-bins 1000000,5000000,20000000
"""

from __future__ import annotations

import argparse
import math
import os
import re
import sys
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

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

DEFAULT_LIQUIDITY_BINS = [0, 50, 100, 200, np.inf]
DEFAULT_LIQUIDITY_LABELS = ["0-50M", "50-100M", "100-200M", "200M+"]

VOLATILITY_LABELS = ["Low", "Medium", "High"]
LEADERSHIP_LABELS = ["Moderate", "Strong", "Elite"]
ADR_LABELS = ["LowADR", "MediumADR", "HighADR"]

MIN_COMPOSITE_TRADES = 50

ADJUSTMENT_SCORING_WEIGHTS = {
    "validation_improvement": 0.35,
    "trade_count": 0.15,
    "stability": 0.25,
    "consistency": 0.15,
    "simplicity_bonus": 0.10,
}

MIN_ARCHETYPE_EFFECT_SIZE = 0.005
MIN_ADJUSTMENT_UPLIFT = 0.01
MAX_DRIFT_TOLERANCE = 0.03
MIN_FACTOR_EFFECT_SHIFT = 0.003
DEFAULT_RULE_TOP_K_PER_FACTOR = 2

# stable schemas for empty outputs
RELEVANCE_SHIFT_COLS = [
    "factor",
    "relevance_shift",
    "baseline_relevant",
    "baseline_effect_size",
    "archetype_effect_size",
    "note",
]

PREFERRED_SHIFT_COLS = [
    "factor",
    "baseline_min",
    "baseline_max",
    "archetype_min",
    "archetype_max",
    "uplift",
    "trades_in_range",
    "baseline_categories",
    "archetype_categories",
]

AVOID_SHIFT_COLS = [
    "factor",
    "baseline_avoid_min",
    "baseline_avoid_max",
    "archetype_avoid_min",
    "archetype_avoid_max",
    "downside",
    "trades_in_zone",
    "baseline_avoid_categories",
    "archetype_avoid_categories",
]

ADJUSTMENT_COLS = [
    "sid",
    "archetype_dimension",
    "archetype_label",
    "factor",
    "adjustment_type",
    "baseline_value",
    "proposed_value",
    "direction",
    "train_expectancy",
    "val_expectancy",
    "train_trades",
    "val_trades",
    "uplift_vs_baseline",
    "drift",
    "score",
    "promoted",
    "note",
]

ARCH_BASELINE_COLS = [
    "SID",
    "Archetype",
    "Trades",
    "Expectancy_Pct",
    "Win_Rate",
    "Median_Return_Pct",
    "Payoff_Ratio",
]

ARCH_CANDIDATE_RULE_COLS = [
    "SID",
    "Archetype",
    "Rule",
    "Train_Expectancy_Pct",
    "Val_Expectancy_Pct",
    "Val_Trades",
    "Val_Uplift_Pct",
    "Median_Uplift_Pct",
    "Drift_Pct",
    "Score",
    "Promoted",
]

ARCH_DEPLOYABLE_RULE_COLS = [
    "SID",
    "Archetype",
    "Rule",
    "Val_Expectancy_Pct",
    "Val_Trades",
    "Val_Uplift_Pct",
    "Median_Uplift_Pct",
    "Drift_Pct",
    "Score",
]

# =============================================================================
# HELPERS
# =============================================================================


def safe_upper(x: Any) -> str:
    return str(x).strip().upper()


def pretty_num(x: Any, ndigits: int = 2) -> str:
    if x is None:
        return "NA"
    try:
        xf = float(x)
        if math.isnan(xf) or math.isinf(xf):
            return "NA"
        return f"{xf:.{ndigits}f}"
    except Exception:
        return str(x)


def standardize_factor_name(x: Any) -> str:
    s = safe_upper(x)
    aliases = {
        "SQZSTATE": "SQZ",
        "SQUEEZE": "SQZ",
        "SQUEEZE_STATE": "SQZ",
    }
    return aliases.get(s, s)


def first_existing(df: pd.DataFrame, names: Iterable[str]) -> Optional[str]:
    if df is None or df.empty or len(df.columns) == 0:
        return None
    lower_map = {c.lower(): c for c in df.columns}
    for name in names:
        if name.lower() in lower_map:
            return lower_map[name.lower()]
    return None


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def safe_read_csv(path: Path, required: bool = False) -> pd.DataFrame:
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Required file missing: {path}")
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
    except Exception:
        if required:
            raise
        return pd.DataFrame()


def split_train_val(df_subset: pd.DataFrame, train_fraction: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_sorted = df_subset.sort_values("entry_dt").reset_index(drop=True)
    split_idx = max(1, min(len(df_sorted) - 1, int(len(df_sorted) * train_fraction)))
    return df_sorted.iloc[:split_idx].copy(), df_sorted.iloc[split_idx:].copy()


def normalize_category_set(x: Any) -> List[str]:
    if x is None:
        return []
    if isinstance(x, float) and math.isnan(x):
        return []
    s = str(x).strip()
    if not s:
        return []
    parts = re.split(r"[|,;/]+", s)
    return [p.strip() for p in parts if p.strip()]


def bucket_to_bounds(bucket_str: str) -> Tuple[Optional[float], Optional[float]]:
    if not isinstance(bucket_str, str):
        return None, None
    nums = re.findall(r"-?\d+(?:\.\d+)?", bucket_str)
    if len(nums) >= 2:
        return float(nums[0]), float(nums[1])
    return None, None


def parse_rule_conditions(rule_text: str) -> List[Dict[str, Any]]:
    if not isinstance(rule_text, str) or not rule_text.strip():
        return []

    conditions: List[Dict[str, Any]] = []
    chunks = re.split(r"\s+AND\s+", rule_text.strip(), flags=re.IGNORECASE)

    for chunk in chunks:
        chunk = chunk.strip()

        m_num = re.match(r"^([A-Za-z_]+)\s*(>=|<=|>|<|=)\s*(-?\d+(?:\.\d+)?)$", chunk)
        if m_num:
            factor = standardize_factor_name(m_num.group(1))
            op = m_num.group(2)
            val = float(m_num.group(3))
            conditions.append({"factor": factor, "type": "numeric", "op": op, "value": val})
            continue

        m_cat = re.match(r"^([A-Za-z_]+)\s+(?:IN|in)\s+(.+)$", chunk)
        if m_cat:
            factor = standardize_factor_name(m_cat.group(1))
            vals = normalize_category_set(m_cat.group(2))
            conditions.append({"factor": factor, "type": "categorical", "op": "in", "values": vals})
            continue

        m_cat_eq = re.match(r"^([A-Za-z_]+)\s*=\s*(.+)$", chunk)
        if m_cat_eq:
            factor = standardize_factor_name(m_cat_eq.group(1))
            rhs = m_cat_eq.group(2).strip()
            try:
                val = float(rhs)
                conditions.append({"factor": factor, "type": "numeric", "op": "=", "value": val})
            except ValueError:
                vals = normalize_category_set(rhs)
                conditions.append({"factor": factor, "type": "categorical", "op": "in", "values": vals})

    return conditions


def format_condition(cond: Dict[str, Any]) -> str:
    if cond["type"] == "numeric":
        return f"{cond['factor']} {cond['op']} {pretty_num(cond['value'])}"
    return f"{cond['factor']} in {'|'.join(cond['values'])}"


def format_rule(conds: List[Dict[str, Any]]) -> str:
    return " AND ".join(format_condition(c) for c in conds)


def empty_df(columns: List[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=columns)


# =============================================================================
# DATA STRUCTURES
# =============================================================================


@dataclass
class FactorDef:
    factor: str
    relevant: bool = False
    relevance_reason: str = ""
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    preferred_min: Optional[float] = None
    preferred_max: Optional[float] = None
    avoid_min: Optional[float] = None
    avoid_max: Optional[float] = None
    preferred_categories: List[str] = field(default_factory=list)
    avoid_categories: List[str] = field(default_factory=list)
    source_notes: List[str] = field(default_factory=list)


@dataclass
class BaselineProfile:
    sid: int
    metrics: Dict[str, Any]
    factor_defs: Dict[str, FactorDef]
    deployable_rule_conditions: List[List[Dict[str, Any]]]


# =============================================================================
# ARGUMENTS
# =============================================================================


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Archetype Profile Engine (Stage 3)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", "-i", required=True, help="Path to clean campaign CSV")
    parser.add_argument("--baseline-dir", "-bd", required=True, help="Stage 2 output directory")
    parser.add_argument("--output", "-o", default="results_archetypes", help="Output directory")
    parser.add_argument("--train-fraction", "-tf", type=float, default=0.70)
    parser.add_argument("--min-trades-per-sid", type=int, default=50)
    parser.add_argument("--min-trades-per-archetype", type=int, default=30)
    parser.add_argument("--min-trades-per-bucket", type=int, default=15)
    parser.add_argument("--min-trades-per-rule", type=int, default=10)
    parser.add_argument("--max-promoted-rules", type=int, default=5)
    parser.add_argument("--enable-composite-archetypes", action="store_true")
    parser.add_argument("--save-charts", action="store_true")
    parser.add_argument("--ddv-bins", type=str, default=None)
    parser.add_argument("--adr-bins", type=int, default=3)
    parser.add_argument("--volatility-bins", type=int, default=3)
    parser.add_argument("--leadership-bins", type=int, default=3)
    return parser.parse_args()


# =============================================================================
# LOAD DATA
# =============================================================================


def load_campaign_data(csv_path: str, min_trades_per_sid: int) -> Optional[pd.DataFrame]:
    print("\n" + "=" * 70)
    print("LOADING CAMPAIGN DATA")
    print("=" * 70)

    if not os.path.exists(csv_path):
        print(f"ERROR: File not found: {csv_path}")
        return None

    try:
        df = safe_read_csv(Path(csv_path), required=True)
    except Exception as e:
        print(f"ERROR: Could not load CSV: {e}")
        return None

    required_cols = {
        "sid",
        "entry_dt",
        "return_pct",
        "RS",
        "CMP",
        "TR",
        "VQS",
        "RMV",
        "DCR",
        "ORC",
        "SQZ",
        "DDV",
        "ADR",
    }
    missing = required_cols - set(df.columns)
    if missing:
        print(f"ERROR: Missing required columns: {sorted(missing)}")
        return None

    try:
        df["entry_dt"] = pd.to_datetime(df["entry_dt"])
    except Exception as e:
        print(f"ERROR: Could not parse entry_dt: {e}")
        return None

    valid_sid_values = sorted(df["sid"].dropna().unique().tolist())
    print(f"✓ Found SetupIDs in dataset: {valid_sid_values}")

    sid_counts = df["sid"].value_counts()
    valid_sids = sid_counts[sid_counts >= min_trades_per_sid].index.tolist()
    if not valid_sids:
        print(f"ERROR: No SetupIDs meet minimum trade count of {min_trades_per_sid}")
        return None

    df = df[df["sid"].isin(valid_sids)].copy()
    df = df.dropna(subset=["return_pct", "RS", "CMP", "TR", "VQS", "RMV", "DCR", "ORC", "SQZ", "DDV", "ADR"]).copy()

    print(f"✓ Loaded {len(df):,} usable rows")
    print(f"✓ Proceeding with SetupIDs: {sorted(valid_sids)}")
    return df


def load_baseline_outputs(baseline_dir: str) -> Optional[Dict[str, pd.DataFrame]]:
    print("\n" + "=" * 70)
    print("LOADING BASELINE OUTPUTS")
    print("=" * 70)

    bdir = Path(baseline_dir)

    required_files = [
        "baseline_by_sid.csv",
        "factor_relevance_by_sid.csv",
        "preferred_ranges_by_sid.csv",
        "avoid_zones_by_sid.csv",
        "deployable_rules_by_sid.csv",
    ]
    missing_files = [name for name in required_files if not (bdir / name).exists()]
    if missing_files:
        print("ERROR: Missing required Stage 2 baseline files:")
        for name in missing_files:
            print(f"  - {name}")
        return None

    try:
        data = {
            "baseline": safe_read_csv(bdir / "baseline_by_sid.csv", required=True),
            "relevance": safe_read_csv(bdir / "factor_relevance_by_sid.csv", required=False),
            "rules": safe_read_csv(bdir / "deployable_rules_by_sid.csv", required=True),
            "preferred": safe_read_csv(bdir / "preferred_ranges_by_sid.csv", required=False),
            "avoid": safe_read_csv(bdir / "avoid_zones_by_sid.csv", required=False),
            "candidates": safe_read_csv(bdir / "candidate_rules_by_sid.csv", required=False),
        }
    except Exception as e:
        print(f"ERROR: Failed loading baseline outputs: {e}")
        return None

    print("✓ Baseline outputs loaded")
    return data


# =============================================================================
# BASELINE PROFILE PARSING
# =============================================================================


def _extract_sid_col(df: pd.DataFrame) -> Optional[str]:
    if df is None or df.empty or len(df.columns) == 0:
        return None
    return first_existing(df, ["SID", "sid"])


def parse_baseline_profiles(baselines: Dict[str, pd.DataFrame]) -> Dict[int, BaselineProfile]:
    baseline_df = baselines["baseline"]
    relevance_df = baselines["relevance"]
    rules_df = baselines["rules"]
    preferred_df = baselines["preferred"]
    avoid_df = baselines["avoid"]

    sid_col_base = _extract_sid_col(baseline_df)
    if sid_col_base is None:
        raise ValueError("Baseline file must have SID column")

    sid_col_rel = _extract_sid_col(relevance_df)
    sid_col_rules = _extract_sid_col(rules_df)
    sid_col_pref = _extract_sid_col(preferred_df)
    sid_col_avoid = _extract_sid_col(avoid_df)

    profiles: Dict[int, BaselineProfile] = {}

    trades_col = first_existing(baseline_df, ["Trades"])
    win_col = first_existing(baseline_df, ["Win_Rate"])
    exp_col = first_existing(baseline_df, ["Expectancy_Pct"])
    med_col = first_existing(baseline_df, ["Median_Return_Pct"])
    payoff_col = first_existing(baseline_df, ["Payoff_Ratio"])

    if not all([trades_col, win_col, exp_col, payoff_col]):
        raise ValueError("baseline_by_sid.csv is missing required columns")

    for _, row in baseline_df.iterrows():
        sid = int(row[sid_col_base])
        metrics = {
            "trades": int(row[trades_col]),
            "win_rate": float(row[win_col]),
            "expectancy": float(row[exp_col]) / 100.0,
            "median_return": float(row[med_col]) / 100.0 if med_col and pd.notna(row[med_col]) else np.nan,
            "payoff_ratio": float(row[payoff_col]),
        }

        factor_defs = {f: FactorDef(factor=f) for f in PRIMARY_FACTORS}

        if sid_col_rel is not None:
            rel_sid = relevance_df[relevance_df[sid_col_rel] == sid].copy()
            factor_col_rel = first_existing(rel_sid, ["Factor"])
            relevant_col = first_existing(rel_sid, ["Relevant"])
            reason_col = first_existing(rel_sid, ["Reason"])

            if factor_col_rel and relevant_col:
                for _, r in rel_sid.iterrows():
                    factor = standardize_factor_name(r[factor_col_rel])
                    if factor not in factor_defs:
                        continue
                    factor_defs[factor].relevant = str(r[relevant_col]).strip().lower() == "yes"
                    factor_defs[factor].relevance_reason = str(r[reason_col]) if reason_col and pd.notna(r[reason_col]) else ""

        if sid_col_pref is not None:
            pref_sid = preferred_df[preferred_df[sid_col_pref] == sid].copy()
            factor_col_pref = first_existing(pref_sid, ["Factor"])
            min_col = first_existing(pref_sid, ["Preferred_Min", "Min_Value", "Range_Min", "archetype_min"])
            max_col = first_existing(pref_sid, ["Preferred_Max", "Max_Value", "Range_Max", "archetype_max"])
            range_col = first_existing(pref_sid, ["Range", "Preferred_Range", "Category", "Preferred_Category"])

            if factor_col_pref:
                for _, r in pref_sid.iterrows():
                    factor = standardize_factor_name(r[factor_col_pref])
                    if factor not in factor_defs:
                        continue
                    fd = factor_defs[factor]

                    if factor in NUMERIC_FACTORS:
                        min_v = float(r[min_col]) if min_col and pd.notna(r[min_col]) else None
                        max_v = float(r[max_col]) if max_col and pd.notna(r[max_col]) else None

                        if min_v is None and max_v is None and range_col and pd.notna(r[range_col]):
                            txt = str(r[range_col])
                            nums = re.findall(r"-?\d+(?:\.\d+)?", txt)
                            if ">=" in txt or ">" in txt:
                                min_v = float(nums[0]) if nums else None
                            elif "<=" in txt or "<" in txt:
                                max_v = float(nums[0]) if nums else None
                            elif len(nums) >= 2:
                                min_v = float(nums[0])
                                max_v = float(nums[1])

                        if min_v is not None:
                            fd.preferred_min = min_v if fd.preferred_min is None else max(fd.preferred_min, min_v)
                        if max_v is not None:
                            fd.preferred_max = max_v if fd.preferred_max is None else min(fd.preferred_max, max_v)
                        fd.source_notes.append("preferred_csv")
                    else:
                        if range_col and pd.notna(r[range_col]):
                            vals = normalize_category_set(r[range_col])
                            fd.preferred_categories = sorted(list(set(fd.preferred_categories).union(vals)))
                            fd.source_notes.append("preferred_csv")

        if sid_col_avoid is not None:
            avoid_sid = avoid_df[avoid_df[sid_col_avoid] == sid].copy()
            factor_col_avoid = first_existing(avoid_sid, ["Factor"])
            min_col = first_existing(avoid_sid, ["Avoid_Min", "Min_Value", "Zone_Min", "archetype_avoid_min"])
            max_col = first_existing(avoid_sid, ["Avoid_Max", "Max_Value", "Zone_Max", "archetype_avoid_max"])
            zone_col = first_existing(avoid_sid, ["Zone", "Avoid_Zone", "Category", "Avoid_Category"])

            if factor_col_avoid:
                for _, r in avoid_sid.iterrows():
                    factor = standardize_factor_name(r[factor_col_avoid])
                    if factor not in factor_defs:
                        continue
                    fd = factor_defs[factor]

                    if factor in NUMERIC_FACTORS:
                        min_v = float(r[min_col]) if min_col and pd.notna(r[min_col]) else None
                        max_v = float(r[max_col]) if max_col and pd.notna(r[max_col]) else None

                        if min_v is None and max_v is None and zone_col and pd.notna(r[zone_col]):
                            txt = str(r[zone_col])
                            nums = re.findall(r"-?\d+(?:\.\d+)?", txt)
                            if ">=" in txt or ">" in txt:
                                min_v = float(nums[0]) if nums else None
                                max_v = np.inf
                            elif "<=" in txt or "<" in txt:
                                min_v = -np.inf
                                max_v = float(nums[0]) if nums else None
                            elif len(nums) >= 2:
                                min_v = float(nums[0])
                                max_v = float(nums[1])

                        if min_v is not None:
                            fd.avoid_min = min_v if fd.avoid_min is None else min(fd.avoid_min, min_v)
                        if max_v is not None:
                            fd.avoid_max = max_v if fd.avoid_max is None else max(fd.avoid_max, max_v)
                        fd.source_notes.append("avoid_csv")
                    else:
                        if zone_col and pd.notna(r[zone_col]):
                            vals = normalize_category_set(r[zone_col])
                            fd.avoid_categories = sorted(list(set(fd.avoid_categories).union(vals)))
                            fd.source_notes.append("avoid_csv")

        deployable_rule_conditions: List[List[Dict[str, Any]]] = []
        if sid_col_rules is not None:
            rules_sid = rules_df[rules_df[sid_col_rules] == sid].copy()
            rule_col = first_existing(rules_sid, ["Rule"])

            if rule_col:
                for _, r in rules_sid.iterrows():
                    if pd.notna(r[rule_col]):
                        conds = parse_rule_conditions(str(r[rule_col]))
                        if conds:
                            deployable_rule_conditions.append(conds)
                            for cond in conds:
                                f = cond["factor"]
                                if f not in factor_defs:
                                    continue
                                fd = factor_defs[f]
                                if cond["type"] == "numeric":
                                    if cond["op"] in (">", ">="):
                                        fd.min_value = cond["value"] if fd.min_value is None else max(fd.min_value, cond["value"])
                                    elif cond["op"] in ("<", "<="):
                                        fd.max_value = cond["value"] if fd.max_value is None else min(fd.max_value, cond["value"])
                                    elif cond["op"] == "=":
                                        fd.min_value = cond["value"]
                                        fd.max_value = cond["value"]
                                else:
                                    fd.preferred_categories = sorted(list(set(fd.preferred_categories).union(cond["values"])))
                                fd.source_notes.append("deployable_rule")

        for fd in factor_defs.values():
            if fd.min_value is None and fd.preferred_min is not None:
                fd.min_value = fd.preferred_min
            if fd.max_value is None and fd.preferred_max is not None:
                fd.max_value = fd.preferred_max

        profiles[sid] = BaselineProfile(
            sid=sid,
            metrics=metrics,
            factor_defs=factor_defs,
            deployable_rule_conditions=deployable_rule_conditions,
        )

    return profiles


# =============================================================================
# ARCHETYPES
# =============================================================================


def build_archetype_dimensions(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print("BUILDING ARCHETYPE DIMENSIONS")
    print("=" * 70)

    out = df.copy()

    if args.ddv_bins:
        try:
            bins = [float(x) for x in args.ddv_bins.split(",")]
            labels = [f"{int(bins[i])}-{int(bins[i + 1])}M" if np.isfinite(bins[i + 1]) else f"{int(bins[i])}M+" for i in range(len(bins) - 1)]
        except Exception:
            bins = DEFAULT_LIQUIDITY_BINS
            labels = DEFAULT_LIQUIDITY_LABELS
    else:
        bins = DEFAULT_LIQUIDITY_BINS
        labels = DEFAULT_LIQUIDITY_LABELS

    out["liquidity_arch"] = pd.cut(out["DDV"], bins=bins, labels=labels, include_lowest=True)

    aq = np.linspace(0, 1, args.adr_bins + 1)
    alabels = ADR_LABELS[: args.adr_bins]
    out["adr_arch"] = pd.qcut(
        pd.to_numeric(out["ADR"], errors="coerce"),
        q=aq,
        labels=alabels,
        duplicates="drop",
    )

    vq = np.linspace(0, 1, args.volatility_bins + 1)
    vlabels = VOLATILITY_LABELS[: args.volatility_bins]
    out["volatility_arch"] = pd.qcut(out["RMV"], q=vq, labels=vlabels, duplicates="drop")

    rs_norm = (out["RS"] - out["RS"].min()) / max(out["RS"].max() - out["RS"].min(), 1e-9)
    tr_norm = (out["TR"] - out["TR"].min()) / max(out["TR"].max() - out["TR"].min(), 1e-9)
    out["leadership_score"] = (rs_norm + tr_norm) / 2.0
    lq = np.linspace(0, 1, args.leadership_bins + 1)
    llabels = LEADERSHIP_LABELS[: args.leadership_bins]
    out["leadership_arch"] = pd.qcut(out["leadership_score"], q=lq, labels=llabels, duplicates="drop")

    if args.enable_composite_archetypes:
        out["composite_arch"] = (
            out["liquidity_arch"].astype(str)
            + "_"
            + out["volatility_arch"].astype(str)
            + "_"
            + out["leadership_arch"].astype(str)
        )
    else:
        out["composite_arch"] = None

    print("✓ Archetype dimensions built")
    return out


# =============================================================================
# METRICS / FACTOR ANALYSIS
# =============================================================================


def compute_baseline_metrics(df_subset: pd.DataFrame) -> Dict[str, Any]:
    returns = df_subset["return_pct"].to_numpy()
    wins = (returns > 0).sum()
    losses = returns[returns < 0]
    positives = returns[returns > 0]
    return {
        "trades": int(len(df_subset)),
        "wins": int(wins),
        "win_rate": float(wins / len(df_subset)) if len(df_subset) else 0.0,
        "expectancy": float(np.mean(returns)) if len(returns) else np.nan,
        "median_return": float(np.median(returns)) if len(returns) else np.nan,
        "avg_hold_days": float(df_subset["hold_days"].mean()) if "hold_days" in df_subset.columns else np.nan,
        "payoff_ratio": float(positives.mean() / abs(losses.mean())) if len(positives) and len(losses) else np.nan,
        "runner_rate": float((df_subset["RUNNER"] == 1).mean()) if "RUNNER" in df_subset.columns else np.nan,
    }


def analyze_numeric_factor(df_subset: pd.DataFrame, factor: str, min_trades_per_bucket: int) -> Optional[pd.DataFrame]:
    if factor not in df_subset.columns:
        return None
    ds = df_subset.dropna(subset=[factor]).copy()
    if len(ds) < min_trades_per_bucket:
        return None
    try:
        ds["bucket"] = pd.qcut(ds[factor], q=5, duplicates="drop")
    except Exception:
        ds["bucket"] = pd.cut(ds[factor], bins=5, duplicates="drop")

    rows = []
    for bucket in ds["bucket"].cat.categories:
        block = ds[ds["bucket"] == bucket]
        if len(block) < min_trades_per_bucket:
            continue
        rets = block["return_pct"].to_numpy()
        rows.append(
            {
                "bucket": str(bucket),
                "trades": int(len(block)),
                "win_rate": float((rets > 0).mean()),
                "expectancy": float(np.mean(rets)),
                "median_return": float(np.median(rets)),
                "mean_factor": float(block[factor].mean()),
            }
        )
    return pd.DataFrame(rows) if rows else None


def analyze_categorical_factor(df_subset: pd.DataFrame, factor: str, min_trades_per_bucket: int) -> Optional[pd.DataFrame]:
    if factor not in df_subset.columns:
        return None
    ds = df_subset.dropna(subset=[factor]).copy()
    if len(ds) < min_trades_per_bucket:
        return None

    rows = []
    for cat, block in ds.groupby(factor):
        if len(block) < min_trades_per_bucket:
            continue
        rets = block["return_pct"].to_numpy()
        rows.append(
            {
                "category": str(cat),
                "trades": int(len(block)),
                "win_rate": float((rets > 0).mean()),
                "expectancy": float(np.mean(rets)),
                "median_return": float(np.median(rets)),
            }
        )
    return pd.DataFrame(rows) if rows else None


def summarize_factor_behavior(df_subset: pd.DataFrame, factor: str, min_trades_per_bucket: int) -> Optional[Dict[str, Any]]:
    if factor in NUMERIC_FACTORS:
        table = analyze_numeric_factor(df_subset, factor, min_trades_per_bucket)
        if table is None or table.empty:
            return None
        exps = table["expectancy"].to_numpy()
        return {
            "type": "numeric",
            "table": table,
            "effect_size": float(np.nanmax(exps) - np.nanmin(exps)),
            "std_expectancy": float(np.nanstd(exps)),
            "best_idx": int(table["expectancy"].idxmax()),
            "worst_idx": int(table["expectancy"].idxmin()),
        }

    if factor in CATEGORICAL_FACTORS:
        table = analyze_categorical_factor(df_subset, factor, min_trades_per_bucket)
        if table is None or table.empty:
            return None
        exps = table["expectancy"].to_numpy()
        return {
            "type": "categorical",
            "table": table,
            "effect_size": float(np.nanmax(exps) - np.nanmin(exps)),
            "std_expectancy": float(np.nanstd(exps)),
            "best_idx": int(table["expectancy"].idxmax()),
            "worst_idx": int(table["expectancy"].idxmin()),
        }
    return None


# =============================================================================
# SHIFT ANALYSIS
# =============================================================================


def analyze_factor_relevance_shift(
    train_arch: pd.DataFrame,
    baseline_profile: BaselineProfile,
    min_trades_per_bucket: int,
) -> pd.DataFrame:
    rows = []
    for factor in PRIMARY_FACTORS:
        fd = baseline_profile.factor_defs[factor]
        if not fd.relevant:
            continue

        summary = summarize_factor_behavior(train_arch, factor, min_trades_per_bucket)
        if summary is None:
            rows.append(
                {
                    "factor": factor,
                    "relevance_shift": "inconclusive",
                    "baseline_relevant": True,
                    "baseline_effect_size": None,
                    "archetype_effect_size": None,
                    "note": "insufficient_archetype_data",
                }
            )
            continue

        arch_eff = summary["effect_size"]
        base_strength = 0.0
        if fd.preferred_min is not None or fd.preferred_max is not None or fd.preferred_categories:
            base_strength += 0.01
        if fd.avoid_min is not None or fd.avoid_max is not None or fd.avoid_categories:
            base_strength += 0.005

        delta = arch_eff - base_strength
        if delta >= MIN_FACTOR_EFFECT_SHIFT:
            shift = "stronger_in_archetype"
        elif delta <= -MIN_FACTOR_EFFECT_SHIFT:
            shift = "weaker_in_archetype"
        else:
            shift = "unchanged"

        rows.append(
            {
                "factor": factor,
                "relevance_shift": shift,
                "baseline_relevant": True,
                "baseline_effect_size": base_strength,
                "archetype_effect_size": arch_eff,
                "note": "",
            }
        )
    return pd.DataFrame(rows, columns=RELEVANCE_SHIFT_COLS)


def analyze_preferred_range_shifts(
    train_arch: pd.DataFrame,
    baseline_profile: BaselineProfile,
    baseline_expectancy: float,
    min_trades_per_bucket: int,
) -> pd.DataFrame:
    rows = []

    for factor in PRIMARY_FACTORS:
        fd = baseline_profile.factor_defs[factor]
        if not fd.relevant:
            continue

        summary = summarize_factor_behavior(train_arch, factor, min_trades_per_bucket)
        if summary is None:
            continue

        if factor in NUMERIC_FACTORS:
            table = summary["table"]
            good = table[table["expectancy"] >= baseline_expectancy + MIN_ADJUSTMENT_UPLIFT].copy()
            if good.empty:
                continue

            mins: List[float] = []
            maxs: List[float] = []
            for b in good["bucket"]:
                mn, mx = bucket_to_bounds(str(b))
                if mn is not None:
                    mins.append(mn)
                if mx is not None:
                    maxs.append(mx)

            if not mins and not maxs:
                continue

            rows.append(
                {
                    "factor": factor,
                    "baseline_min": fd.preferred_min,
                    "baseline_max": fd.preferred_max,
                    "archetype_min": min(mins) if mins else None,
                    "archetype_max": max(maxs) if maxs else None,
                    "uplift": float(good["expectancy"].max() - baseline_expectancy),
                    "trades_in_range": int(good["trades"].sum()),
                    "baseline_categories": None,
                    "archetype_categories": None,
                }
            )

        else:
            table = summary["table"]
            good = table[table["expectancy"] >= baseline_expectancy + MIN_ADJUSTMENT_UPLIFT].copy()
            if good.empty:
                continue

            rows.append(
                {
                    "factor": factor,
                    "baseline_min": None,
                    "baseline_max": None,
                    "archetype_min": None,
                    "archetype_max": None,
                    "uplift": float(good["expectancy"].max() - baseline_expectancy),
                    "trades_in_range": int(good["trades"].sum()),
                    "baseline_categories": "|".join(fd.preferred_categories),
                    "archetype_categories": "|".join(sorted(good["category"].astype(str).tolist())),
                }
            )

    return pd.DataFrame(rows, columns=PREFERRED_SHIFT_COLS)


def analyze_avoid_zone_shifts(
    train_arch: pd.DataFrame,
    baseline_profile: BaselineProfile,
    baseline_expectancy: float,
    min_trades_per_bucket: int,
) -> pd.DataFrame:
    rows = []

    for factor in PRIMARY_FACTORS:
        fd = baseline_profile.factor_defs[factor]
        if not fd.relevant:
            continue

        summary = summarize_factor_behavior(train_arch, factor, min_trades_per_bucket)
        if summary is None:
            continue

        if factor in NUMERIC_FACTORS:
            table = summary["table"]
            bad = table[table["expectancy"] <= baseline_expectancy - MIN_ADJUSTMENT_UPLIFT].copy()
            if bad.empty:
                continue

            mins: List[float] = []
            maxs: List[float] = []
            for b in bad["bucket"]:
                mn, mx = bucket_to_bounds(str(b))
                if mn is not None:
                    mins.append(mn)
                if mx is not None:
                    maxs.append(mx)

            rows.append(
                {
                    "factor": factor,
                    "baseline_avoid_min": fd.avoid_min,
                    "baseline_avoid_max": fd.avoid_max,
                    "archetype_avoid_min": min(mins) if mins else None,
                    "archetype_avoid_max": max(maxs) if maxs else None,
                    "downside": float(baseline_expectancy - bad["expectancy"].min()),
                    "trades_in_zone": int(bad["trades"].sum()),
                    "baseline_avoid_categories": None,
                    "archetype_avoid_categories": None,
                }
            )
        else:
            table = summary["table"]
            bad = table[table["expectancy"] <= baseline_expectancy - MIN_ADJUSTMENT_UPLIFT].copy()
            if bad.empty:
                continue

            rows.append(
                {
                    "factor": factor,
                    "baseline_avoid_min": None,
                    "baseline_avoid_max": None,
                    "archetype_avoid_min": None,
                    "archetype_avoid_max": None,
                    "downside": float(baseline_expectancy - bad["expectancy"].min()),
                    "trades_in_zone": int(bad["trades"].sum()),
                    "baseline_avoid_categories": "|".join(fd.avoid_categories),
                    "archetype_avoid_categories": "|".join(sorted(bad["category"].astype(str).tolist())),
                }
            )

    return pd.DataFrame(rows, columns=AVOID_SHIFT_COLS)


# =============================================================================
# RULES / ADJUSTMENTS
# =============================================================================


def evaluate_rule(df_data: pd.DataFrame, conditions: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if df_data.empty:
        return None

    mask = pd.Series(True, index=df_data.index)

    for cond in conditions:
        factor = cond["factor"]
        if factor not in df_data.columns:
            return None

        if cond["type"] == "numeric":
            val = cond["value"]
            op = cond["op"]
            if op == ">":
                mask &= df_data[factor] > val
            elif op == ">=":
                mask &= df_data[factor] >= val
            elif op == "<":
                mask &= df_data[factor] < val
            elif op == "<=":
                mask &= df_data[factor] <= val
            elif op == "=":
                mask &= df_data[factor] == val
        else:
            vals = set(cond["values"])
            mask &= df_data[factor].astype(str).isin(vals)

    sub = df_data[mask].copy()
    if sub.empty:
        return None

    rets = sub["return_pct"].to_numpy()
    return {
        "trades": int(len(sub)),
        "win_rate": float((rets > 0).mean()),
        "expectancy": float(np.mean(rets)),
        "median_return": float(np.median(rets)),
    }


def build_adjustment_candidates(
    sid: int,
    arch_dim: str,
    arch_label: str,
    train_arch: pd.DataFrame,
    val_arch: pd.DataFrame,
    baseline_profile: BaselineProfile,
    preferred_shift_df: pd.DataFrame,
    avoid_shift_df: pd.DataFrame,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    base_exp = baseline_profile.metrics["expectancy"]

    if preferred_shift_df is None or preferred_shift_df.empty:
        preferred_shift_df = empty_df(PREFERRED_SHIFT_COLS)
    if avoid_shift_df is None or avoid_shift_df.empty:
        avoid_shift_df = empty_df(AVOID_SHIFT_COLS)

    for _, r in preferred_shift_df.iterrows():
        factor = r["factor"]
        if factor in NUMERIC_FACTORS:
            fd = baseline_profile.factor_defs[factor]

            if pd.notna(r.get("archetype_min")):
                proposed = float(r["archetype_min"])
                baseline_value = fd.min_value
                if baseline_value is not None and proposed > baseline_value:
                    cond = [{"factor": factor, "type": "numeric", "op": ">=", "value": proposed}]
                    tr = evaluate_rule(train_arch, cond)
                    va = evaluate_rule(val_arch, cond)
                    if tr and va:
                        uplift = va["expectancy"] - base_exp
                        drift = abs(tr["expectancy"] - va["expectancy"])
                        if uplift >= MIN_ADJUSTMENT_UPLIFT and drift <= MAX_DRIFT_TOLERANCE:
                            rows.append(
                                {
                                    "sid": sid,
                                    "archetype_dimension": arch_dim,
                                    "archetype_label": arch_label,
                                    "factor": factor,
                                    "adjustment_type": "tighten_min",
                                    "baseline_value": baseline_value,
                                    "proposed_value": proposed,
                                    "direction": "increase",
                                    "train_expectancy": tr["expectancy"],
                                    "val_expectancy": va["expectancy"],
                                    "train_trades": tr["trades"],
                                    "val_trades": va["trades"],
                                    "uplift_vs_baseline": uplift,
                                    "drift": drift,
                                    "score": 0.0,
                                    "promoted": False,
                                    "note": f"{factor} >= {pretty_num(proposed)}",
                                }
                            )

            if pd.notna(r.get("archetype_max")):
                proposed = float(r["archetype_max"])
                baseline_value = fd.max_value
                if baseline_value is not None and proposed < baseline_value:
                    cond = [{"factor": factor, "type": "numeric", "op": "<=", "value": proposed}]
                    tr = evaluate_rule(train_arch, cond)
                    va = evaluate_rule(val_arch, cond)
                    if tr and va:
                        uplift = va["expectancy"] - base_exp
                        drift = abs(tr["expectancy"] - va["expectancy"])
                        if uplift >= MIN_ADJUSTMENT_UPLIFT and drift <= MAX_DRIFT_TOLERANCE:
                            rows.append(
                                {
                                    "sid": sid,
                                    "archetype_dimension": arch_dim,
                                    "archetype_label": arch_label,
                                    "factor": factor,
                                    "adjustment_type": "tighten_max",
                                    "baseline_value": baseline_value,
                                    "proposed_value": proposed,
                                    "direction": "decrease",
                                    "train_expectancy": tr["expectancy"],
                                    "val_expectancy": va["expectancy"],
                                    "train_trades": tr["trades"],
                                    "val_trades": va["trades"],
                                    "uplift_vs_baseline": uplift,
                                    "drift": drift,
                                    "score": 0.0,
                                    "promoted": False,
                                    "note": f"{factor} <= {pretty_num(proposed)}",
                                }
                            )

        else:
            fd = baseline_profile.factor_defs[factor]
            vals = normalize_category_set(r.get("archetype_categories"))
            if vals:
                cond = [{"factor": factor, "type": "categorical", "op": "in", "values": vals}]
                tr = evaluate_rule(train_arch, cond)
                va = evaluate_rule(val_arch, cond)
                if tr and va:
                    uplift = va["expectancy"] - base_exp
                    drift = abs(tr["expectancy"] - va["expectancy"])
                    if uplift >= MIN_ADJUSTMENT_UPLIFT and drift <= MAX_DRIFT_TOLERANCE:
                        rows.append(
                            {
                                "sid": sid,
                                "archetype_dimension": arch_dim,
                                "archetype_label": arch_label,
                                "factor": factor,
                                "adjustment_type": "preferred_categories",
                                "baseline_value": "|".join(fd.preferred_categories),
                                "proposed_value": "|".join(vals),
                                "direction": "categorical",
                                "train_expectancy": tr["expectancy"],
                                "val_expectancy": va["expectancy"],
                                "train_trades": tr["trades"],
                                "val_trades": va["trades"],
                                "uplift_vs_baseline": uplift,
                                "drift": drift,
                                "score": 0.0,
                                "promoted": False,
                                "note": f"{factor} in {'|'.join(vals)}",
                            }
                        )

    for _, r in avoid_shift_df.iterrows():
        factor = r["factor"]
        if factor not in NUMERIC_FACTORS:
            continue
        fd = baseline_profile.factor_defs[factor]

        if pd.notna(r.get("archetype_avoid_min")) and fd.avoid_min is not None:
            proposed = float(r["archetype_avoid_min"])
            baseline_value = fd.avoid_min
            if proposed < baseline_value:
                rows.append(
                    {
                        "sid": sid,
                        "archetype_dimension": arch_dim,
                        "archetype_label": arch_label,
                        "factor": factor,
                        "adjustment_type": "avoid_min_shift",
                        "baseline_value": baseline_value,
                        "proposed_value": proposed,
                        "direction": "decrease",
                        "train_expectancy": np.nan,
                        "val_expectancy": np.nan,
                        "train_trades": np.nan,
                        "val_trades": np.nan,
                        "uplift_vs_baseline": 0.0,
                        "drift": 0.0,
                        "score": 0.0,
                        "promoted": False,
                        "note": f"avoid {factor} >= {pretty_num(proposed)}",
                    }
                )

        if pd.notna(r.get("archetype_avoid_max")) and fd.avoid_max is not None and np.isfinite(fd.avoid_max):
            proposed = float(r["archetype_avoid_max"])
            baseline_value = fd.avoid_max
            if proposed > baseline_value:
                rows.append(
                    {
                        "sid": sid,
                        "archetype_dimension": arch_dim,
                        "archetype_label": arch_label,
                        "factor": factor,
                        "adjustment_type": "avoid_max_shift",
                        "baseline_value": baseline_value,
                        "proposed_value": proposed,
                        "direction": "increase",
                        "train_expectancy": np.nan,
                        "val_expectancy": np.nan,
                        "train_trades": np.nan,
                        "val_trades": np.nan,
                        "uplift_vs_baseline": 0.0,
                        "drift": 0.0,
                        "score": 0.0,
                        "promoted": False,
                        "note": f"avoid {factor} <= {pretty_num(proposed)}",
                    }
                )

    df = pd.DataFrame(rows, columns=ADJUSTMENT_COLS)
    if df.empty:
        return df

    scores = []
    for _, r in df.iterrows():
        uplift = float(r["uplift_vs_baseline"]) if pd.notna(r["uplift_vs_baseline"]) else 0.0
        drift = float(r["drift"]) if pd.notna(r["drift"]) else 0.0
        val_trades = float(r["val_trades"]) if pd.notna(r["val_trades"]) else 0.0
        consistency = 1.0

        if pd.notna(r["baseline_value"]) and pd.notna(r["proposed_value"]):
            if r["adjustment_type"] == "tighten_min" and r["proposed_value"] < r["baseline_value"]:
                consistency = 0.0
            if r["adjustment_type"] == "tighten_max" and r["proposed_value"] > r["baseline_value"]:
                consistency = 0.0

        score = (
            ADJUSTMENT_SCORING_WEIGHTS["validation_improvement"] * min(max(uplift / MIN_ADJUSTMENT_UPLIFT, 0.0), 3.0)
            + ADJUSTMENT_SCORING_WEIGHTS["trade_count"] * min(val_trades / 50.0, 1.0)
            + ADJUSTMENT_SCORING_WEIGHTS["stability"] * max(0.0, 1.0 - drift / MAX_DRIFT_TOLERANCE)
            + ADJUSTMENT_SCORING_WEIGHTS["consistency"] * consistency
            + ADJUSTMENT_SCORING_WEIGHTS["simplicity_bonus"] * 1.0
        )
        scores.append(score)

    df["score"] = scores
    df["promoted"] = (
        (df["uplift_vs_baseline"] >= MIN_ADJUSTMENT_UPLIFT)
        & (df["drift"].fillna(0.0) <= MAX_DRIFT_TOLERANCE)
        & (df["val_trades"].fillna(0) >= 10)
    )
    return df.sort_values(["promoted", "score"], ascending=[False, False]).reset_index(drop=True)


def build_rule_seed_conditions(
    baseline_profile: BaselineProfile,
    relevance_shift_df: pd.DataFrame,
    preferred_shift_df: pd.DataFrame,
) -> Dict[str, List[Dict[str, Any]]]:
    seeds: Dict[str, List[Dict[str, Any]]] = {}

    if relevance_shift_df is None or relevance_shift_df.empty or "factor" not in relevance_shift_df.columns:
        relevance_shift_df = empty_df(RELEVANCE_SHIFT_COLS)
    if preferred_shift_df is None or preferred_shift_df.empty or "factor" not in preferred_shift_df.columns:
        preferred_shift_df = empty_df(PREFERRED_SHIFT_COLS)

    shifts = {row["factor"]: row["relevance_shift"] for _, row in relevance_shift_df.iterrows()}

    for factor in PRIMARY_FACTORS:
        fd = baseline_profile.factor_defs[factor]
        if not fd.relevant:
            continue

        factor_conds: List[Dict[str, Any]] = []

        if factor in NUMERIC_FACTORS:
            if fd.min_value is not None:
                factor_conds.append({"factor": factor, "type": "numeric", "op": ">=", "value": float(fd.min_value)})
            if fd.max_value is not None:
                factor_conds.append({"factor": factor, "type": "numeric", "op": "<=", "value": float(fd.max_value)})

            pref = preferred_shift_df[preferred_shift_df["factor"] == factor]
            if not pref.empty:
                r = pref.iloc[0]
                if pd.notna(r.get("archetype_min")) and fd.min_value is not None and float(r["archetype_min"]) > fd.min_value:
                    factor_conds.append({"factor": factor, "type": "numeric", "op": ">=", "value": float(r["archetype_min"])})
                if pd.notna(r.get("archetype_max")) and fd.max_value is not None and float(r["archetype_max"]) < fd.max_value:
                    factor_conds.append({"factor": factor, "type": "numeric", "op": "<=", "value": float(r["archetype_max"])})
        else:
            cats = fd.preferred_categories
            if cats:
                factor_conds.append({"factor": factor, "type": "categorical", "op": "in", "values": cats})

        if shifts.get(factor) == "stronger_in_archetype":
            seeds[factor] = factor_conds[:DEFAULT_RULE_TOP_K_PER_FACTOR]
        elif factor_conds:
            seeds[factor] = factor_conds[:1]

    return seeds


def score_rule_candidate(candidate: Dict[str, Any], baseline_profile: BaselineProfile) -> Dict[str, Any]:
    tr = candidate["train_eval"]
    va = candidate["val_eval"]
    base_exp = baseline_profile.metrics["expectancy"]
    base_med = baseline_profile.metrics["median_return"]

    uplift = va["expectancy"] - base_exp
    drift = abs(tr["expectancy"] - va["expectancy"])
    median_uplift = va["median_return"] - base_med
    trade_score = min(va["trades"] / 50.0, 1.0)
    stability = max(0.0, 1.0 - drift / MAX_DRIFT_TOLERANCE)

    consistency = 1.0
    for cond in candidate["conditions"]:
        fd = baseline_profile.factor_defs.get(cond["factor"])
        if not fd:
            continue
        if cond["type"] == "numeric":
            if cond["op"] in (">", ">=") and fd.min_value is not None and cond["value"] < fd.min_value:
                consistency = 0.0
            if cond["op"] in ("<", "<=") and fd.max_value is not None and cond["value"] > fd.max_value:
                consistency = 0.0

    simplicity_bonus = 1.05 if len(candidate["conditions"]) == 2 else 1.0

    score = (
        ADJUSTMENT_SCORING_WEIGHTS["validation_improvement"] * min(max(uplift / MIN_ADJUSTMENT_UPLIFT, 0.0), 3.0)
        + ADJUSTMENT_SCORING_WEIGHTS["trade_count"] * trade_score
        + ADJUSTMENT_SCORING_WEIGHTS["stability"] * stability
        + ADJUSTMENT_SCORING_WEIGHTS["consistency"] * consistency
        + ADJUSTMENT_SCORING_WEIGHTS["simplicity_bonus"] * simplicity_bonus
    )

    candidate["val_uplift"] = uplift
    candidate["drift"] = drift
    candidate["median_uplift"] = median_uplift
    candidate["composite_score"] = score
    candidate["promoted"] = (
        uplift >= MIN_ADJUSTMENT_UPLIFT
        and va["trades"] >= 10
        and drift <= MAX_DRIFT_TOLERANCE
        and va["median_return"] >= base_med
    )
    return candidate


def generate_archetype_rules(
    train_arch: pd.DataFrame,
    val_arch: pd.DataFrame,
    baseline_profile: BaselineProfile,
    relevance_shift_df: pd.DataFrame,
    preferred_shift_df: pd.DataFrame,
    min_trades_per_rule: int,
    max_promoted_rules: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    seeds = build_rule_seed_conditions(baseline_profile, relevance_shift_df, preferred_shift_df)
    if len(seeds) < 2:
        return [], []

    all_candidates: List[Dict[str, Any]] = []
    factors = list(seeds.keys())

    for k in (2, 3):
        for combo in combinations(factors, k):
            cond_lists = [seeds[f] for f in combo]
            combos = [[]]
            for cond_list in cond_lists:
                next_combos = []
                for prefix in combos:
                    for cond in cond_list:
                        next_combos.append(prefix + [cond])
                combos = next_combos

            for conds in combos:
                tr = evaluate_rule(train_arch, conds)
                va = evaluate_rule(val_arch, conds)
                if not tr or not va:
                    continue
                if va["trades"] < min_trades_per_rule:
                    continue
                cand = {"conditions": conds, "train_eval": tr, "val_eval": va}
                cand = score_rule_candidate(cand, baseline_profile)
                all_candidates.append(cand)

    all_candidates = sorted(all_candidates, key=lambda x: x["composite_score"], reverse=True)
    promoted = [c for c in all_candidates if c["promoted"]][:max_promoted_rules]
    return all_candidates, promoted


# =============================================================================
# OUTPUTS
# =============================================================================


def save_outputs(output_dir: Path, archetype_assignments: pd.DataFrame, all_results: Dict[int, Dict[str, Any]]) -> None:
    archetype_assignments.to_csv(output_dir / "archetype_assignments.csv", index=False)

    baseline_rows = []
    relevance_rows = []
    pref_rows = []
    avoid_rows = []
    adj_rows = []
    candidate_rows = []
    deployable_rows = []

    for sid, results in all_results.items():
        for arch_name, arch_data in results.items():
            metrics = arch_data["baseline_metrics"]
            baseline_rows.append(
                {
                    "SID": sid,
                    "Archetype": arch_name,
                    "Trades": metrics["trades"],
                    "Expectancy_Pct": metrics["expectancy"] * 100,
                    "Win_Rate": metrics["win_rate"],
                    "Median_Return_Pct": metrics["median_return"] * 100,
                    "Payoff_Ratio": metrics["payoff_ratio"],
                }
            )

            rel_df = arch_data.get("relevance", empty_df(RELEVANCE_SHIFT_COLS))
            if not rel_df.empty:
                tmp = rel_df.copy()
                tmp.insert(0, "Archetype", arch_name)
                tmp.insert(0, "SID", sid)
                relevance_rows.extend(tmp.to_dict("records"))

            pref_df = arch_data.get("preferred_ranges", empty_df(PREFERRED_SHIFT_COLS))
            if not pref_df.empty:
                tmp = pref_df.copy()
                tmp.insert(0, "Archetype", arch_name)
                tmp.insert(0, "SID", sid)
                pref_rows.extend(tmp.to_dict("records"))

            avoid_df = arch_data.get("avoid_zones", empty_df(AVOID_SHIFT_COLS))
            if not avoid_df.empty:
                tmp = avoid_df.copy()
                tmp.insert(0, "Archetype", arch_name)
                tmp.insert(0, "SID", sid)
                avoid_rows.extend(tmp.to_dict("records"))

            adj_df = arch_data.get("adjustments", empty_df(ADJUSTMENT_COLS))
            if not adj_df.empty:
                adj_rows.extend(adj_df.to_dict("records"))

            for cand in arch_data.get("candidate_rules", []):
                candidate_rows.append(
                    {
                        "SID": sid,
                        "Archetype": arch_name,
                        "Rule": format_rule(cand["conditions"]),
                        "Train_Expectancy_Pct": cand["train_eval"]["expectancy"] * 100,
                        "Val_Expectancy_Pct": cand["val_eval"]["expectancy"] * 100,
                        "Val_Trades": cand["val_eval"]["trades"],
                        "Val_Uplift_Pct": cand["val_uplift"] * 100,
                        "Median_Uplift_Pct": cand["median_uplift"] * 100,
                        "Drift_Pct": cand["drift"] * 100,
                        "Score": cand["composite_score"],
                        "Promoted": cand["promoted"],
                    }
                )

            for cand in arch_data.get("promoted_rules", []):
                deployable_rows.append(
                    {
                        "SID": sid,
                        "Archetype": arch_name,
                        "Rule": format_rule(cand["conditions"]),
                        "Val_Expectancy_Pct": cand["val_eval"]["expectancy"] * 100,
                        "Val_Trades": cand["val_eval"]["trades"],
                        "Val_Uplift_Pct": cand["val_uplift"] * 100,
                        "Median_Uplift_Pct": cand["median_uplift"] * 100,
                        "Drift_Pct": cand["drift"] * 100,
                        "Score": cand["composite_score"],
                    }
                )

    pd.DataFrame(baseline_rows, columns=ARCH_BASELINE_COLS).to_csv(output_dir / "archetype_baseline_by_sid.csv", index=False)
    pd.DataFrame(relevance_rows, columns=["SID", "Archetype"] + RELEVANCE_SHIFT_COLS).to_csv(output_dir / "archetype_relevance_by_sid.csv", index=False)
    pd.DataFrame(pref_rows, columns=["SID", "Archetype"] + PREFERRED_SHIFT_COLS).to_csv(output_dir / "archetype_preferred_ranges_by_sid.csv", index=False)
    pd.DataFrame(avoid_rows, columns=["SID", "Archetype"] + AVOID_SHIFT_COLS).to_csv(output_dir / "archetype_avoid_zones_by_sid.csv", index=False)
    pd.DataFrame(adj_rows, columns=ADJUSTMENT_COLS).to_csv(output_dir / "archetype_adjustments_by_sid.csv", index=False)
    pd.DataFrame(candidate_rows, columns=ARCH_CANDIDATE_RULE_COLS).to_csv(output_dir / "archetype_candidate_rules_by_sid.csv", index=False)
    pd.DataFrame(deployable_rows, columns=ARCH_DEPLOYABLE_RULE_COLS).to_csv(output_dir / "archetype_deployable_rules_by_sid.csv", index=False)


def write_reports(output_dir: Path, all_results: Dict[int, Dict[str, Any]], baseline_profiles: Dict[int, BaselineProfile]) -> None:
    for sid, results in all_results.items():
        baseline_profile = baseline_profiles[sid]
        report_path = output_dir / f"report_sid_{sid}_archetypes.txt"
        with open(report_path, "w") as f:
            f.write(f"SID {sid} - ARCHETYPE PROFILE REPORT\n")
            f.write("=" * 70 + "\n\n")
            f.write("BASELINE ANCHOR\n")
            f.write("-" * 70 + "\n")
            f.write(f"Trades: {baseline_profile.metrics['trades']}\n")
            f.write(f"Expectancy: {baseline_profile.metrics['expectancy']*100:.2f}%\n")
            f.write(f"Win Rate: {baseline_profile.metrics['win_rate']:.1%}\n")
            f.write(f"Payoff Ratio: {baseline_profile.metrics['payoff_ratio']:.2f}\n\n")

            for arch_name, arch_data in results.items():
                m = arch_data["baseline_metrics"]
                f.write(f"{arch_name}\n")
                f.write("-" * 70 + "\n")
                f.write(f"Trades: {m['trades']}\n")
                f.write(f"Expectancy: {m['expectancy']*100:.2f}%\n")
                f.write(f"Win Rate: {m['win_rate']:.1%}\n")

                adj_df = arch_data.get("adjustments", empty_df(ADJUSTMENT_COLS))
                promoted_adj = adj_df[adj_df["promoted"] == True] if not adj_df.empty else empty_df(ADJUSTMENT_COLS)
                if promoted_adj.empty:
                    f.write("Adjustments: baseline unchanged\n")
                else:
                    f.write("Promoted Adjustments:\n")
                    for _, r in promoted_adj.iterrows():
                        f.write(
                            f"  - {r['factor']}: {r['adjustment_type']} | baseline={pretty_num(r['baseline_value'])} "
                            f"-> proposed={pretty_num(r['proposed_value'])} | uplift={r['uplift_vs_baseline']*100:.2f}%\n"
                        )

                promoted_rules = arch_data.get("promoted_rules", [])
                if promoted_rules:
                    f.write("Deployable Rules:\n")
                    for pr in promoted_rules:
                        f.write(
                            f"  - {format_rule(pr['conditions'])} | val_exp={pr['val_eval']['expectancy']*100:.2f}% "
                            f"| uplift={pr['val_uplift']*100:.2f}% | trades={pr['val_eval']['trades']}\n"
                        )
                else:
                    f.write("Deployable Rules: none\n")
                f.write("\n")

    summary_path = output_dir / "SUMMARY_ARCHETYPES.txt"
    with open(summary_path, "w") as f:
        f.write("ARCHETYPE PROFILE ENGINE - OVERALL SUMMARY\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        total_examined = 0
        total_with_shift = 0
        total_with_adj = 0
        total_with_rules = 0

        for sid, results in all_results.items():
            examined = len(results)
            with_shift = sum(1 for v in results.values() if not v.get("relevance", empty_df(RELEVANCE_SHIFT_COLS)).empty)
            with_adj = sum(1 for v in results.values() if not v.get("adjustments", empty_df(ADJUSTMENT_COLS)).empty and v["adjustments"]["promoted"].any())
            with_rules = sum(len(v.get("promoted_rules", [])) for v in results.values())

            total_examined += examined
            total_with_shift += with_shift
            total_with_adj += with_adj
            total_with_rules += with_rules

            f.write(f"SID {sid}\n")
            f.write(f"  Archetypes examined: {examined}\n")
            f.write(f"  Archetypes with meaningful factor shifts: {with_shift}\n")
            f.write(f"  Archetypes with promoted adjustments: {with_adj}\n")
            f.write(f"  Deployable archetype rules: {with_rules}\n\n")

        f.write("TOTALS\n")
        f.write(f"  Archetypes examined: {total_examined}\n")
        f.write(f"  Archetypes with meaningful factor shifts: {total_with_shift}\n")
        f.write(f"  Archetypes with promoted adjustments: {total_with_adj}\n")
        f.write(f"  Deployable archetype rules: {total_with_rules}\n")


def write_environment_summary(output_dir: Path, all_results: Dict[int, Dict[str, Any]]) -> None:
    rows = []

    for sid, results in all_results.items():
        for arch_name, arch_data in results.items():
            m = arch_data["baseline_metrics"]
            rows.append(
                {
                    "SID": sid,
                    "Archetype": arch_name,
                    "Trades": m["trades"],
                    "Expectancy_Pct": m["expectancy"] * 100 if pd.notna(m["expectancy"]) else np.nan,
                    "Win_Rate": m["win_rate"],
                    "Payoff_Ratio": m["payoff_ratio"],
                    "Promoted_Adjustments": int(
                        0 if arch_data.get("adjustments", pd.DataFrame()).empty
                        else arch_data["adjustments"]["promoted"].fillna(False).sum()
                    ),
                    "Deployable_Rules": len(arch_data.get("promoted_rules", [])),
                }
            )

    env_df = pd.DataFrame(rows)
    env_df.to_csv(output_dir / "environment_summary.csv", index=False)

    txt_path = output_dir / "ENVIRONMENT_SUMMARY.txt"
    with open(txt_path, "w") as f:
        f.write("ENVIRONMENT SUMMARY\n")
        f.write("=" * 70 + "\n\n")

        if env_df.empty:
            f.write("No archetype environments available.\n")
            return

        ranked = env_df.sort_values(
            ["Expectancy_Pct", "Trades"],
            ascending=[False, False]
        )

        f.write("TOP ENVIRONMENTS BY EXPECTANCY\n")
        f.write("-" * 70 + "\n")
        for _, row in ranked.head(20).iterrows():
            f.write(
                f"SID {int(row['SID'])} | {row['Archetype']} | "
                f"Exp={row['Expectancy_Pct']:.2f}% | "
                f"Trades={int(row['Trades'])} | "
                f"Rules={int(row['Deployable_Rules'])} | "
                f"Adj={int(row['Promoted_Adjustments'])}\n"
            )


def maybe_save_chart(output_dir: Path, sid: int, arch_name: str, factor: str, summary: Dict[str, Any]) -> None:
    if summary["type"] == "numeric":
        table = summary["table"]
        x = np.arange(len(table))
        labels = table["bucket"].astype(str).tolist()
    else:
        table = summary["table"]
        x = np.arange(len(table))
        labels = table["category"].astype(str).tolist()

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(x, table["expectancy"] * 100)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Expectancy (%)")
    ax.set_title(f"SID {sid} | {arch_name} | {factor}")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / f"chart_sid{sid}_{arch_name}_{factor}.png", dpi=120, facecolor="white")
    plt.close(fig)


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    args = parse_arguments()

    print("\n" + "=" * 70)
    print("Archetype Profile Engine (Stage 3)")
    print("=" * 70)

    df = load_campaign_data(args.input, args.min_trades_per_sid)
    if df is None:
        sys.exit(1)

    baselines = load_baseline_outputs(args.baseline_dir)
    if baselines is None:
        sys.exit(1)

    baseline_profiles = parse_baseline_profiles(baselines)
    df_arch = build_archetype_dimensions(df, args)

    output_dir = Path(args.output)
    ensure_dir(output_dir)

    assignment_cols = [
        "entry_dt",
        "sid",
        "symbol",
        "liquidity_arch",
        "adr_arch",
        "volatility_arch",
        "leadership_arch",
    ]
    if args.enable_composite_archetypes:
        assignment_cols.append("composite_arch")
    archetype_assignments = df_arch[[c for c in assignment_cols if c in df_arch.columns]].copy()

    all_results: Dict[int, Dict[str, Any]] = {}

    for sid in sorted(df_arch["sid"].dropna().unique()):
        if sid not in baseline_profiles:
            continue
        df_sid = df_arch[df_arch["sid"] == sid].copy()
        if df_sid.empty:
            continue

        bp = baseline_profiles[sid]
        print("\n" + "=" * 70)
        print(f"ANALYZING SID {sid}")
        print("=" * 70)
        print(f"Baseline expectancy: {bp.metrics['expectancy']*100:.2f}%")
        print(f"Relevant factors: {', '.join([f for f in PRIMARY_FACTORS if bp.factor_defs[f].relevant]) or 'None'}")

        sid_results: Dict[str, Any] = {}

        archetype_dimensions = ["liquidity_arch", "adr_arch", "volatility_arch", "leadership_arch"]
        if args.enable_composite_archetypes:
            archetype_dimensions.append("composite_arch")

        for arch_dim in archetype_dimensions:
            values = df_sid[arch_dim].dropna().astype(str).unique().tolist()
            for arch_value in values:
                df_group = df_sid[df_sid[arch_dim].astype(str) == arch_value].copy()
                min_needed = MIN_COMPOSITE_TRADES if arch_dim == "composite_arch" else args.min_trades_per_archetype
                if len(df_group) < min_needed:
                    continue

                arch_name = f"{arch_dim.replace('_arch', '')}_{arch_value}"
                train_arch, val_arch = split_train_val(df_group, args.train_fraction)

                metrics = compute_baseline_metrics(df_group)
                relevance_shift_df = analyze_factor_relevance_shift(train_arch, bp, args.min_trades_per_bucket)
                preferred_shift_df = analyze_preferred_range_shifts(train_arch, bp, bp.metrics["expectancy"], args.min_trades_per_bucket)
                avoid_shift_df = analyze_avoid_zone_shifts(train_arch, bp, bp.metrics["expectancy"], args.min_trades_per_bucket)

                adjustments_df = build_adjustment_candidates(
                    sid=sid,
                    arch_dim=arch_dim.replace("_arch", ""),
                    arch_label=arch_value,
                    train_arch=train_arch,
                    val_arch=val_arch,
                    baseline_profile=bp,
                    preferred_shift_df=preferred_shift_df,
                    avoid_shift_df=avoid_shift_df,
                )

                candidate_rules, promoted_rules = generate_archetype_rules(
                    train_arch=train_arch,
                    val_arch=val_arch,
                    baseline_profile=bp,
                    relevance_shift_df=relevance_shift_df,
                    preferred_shift_df=preferred_shift_df,
                    min_trades_per_rule=args.min_trades_per_rule,
                    max_promoted_rules=args.max_promoted_rules,
                )

                sid_results[arch_name] = {
                    "baseline_metrics": metrics,
                    "relevance": relevance_shift_df,
                    "preferred_ranges": preferred_shift_df,
                    "avoid_zones": avoid_shift_df,
                    "adjustments": adjustments_df,
                    "candidate_rules": candidate_rules,
                    "promoted_rules": promoted_rules,
                }

                if args.save_charts:
                    for factor in PRIMARY_FACTORS:
                        summary = summarize_factor_behavior(train_arch, factor, args.min_trades_per_bucket)
                        if summary is not None:
                            maybe_save_chart(output_dir, sid, arch_name, factor, summary)

        all_results[sid] = sid_results

    save_outputs(output_dir, archetype_assignments, all_results)
    write_reports(output_dir, all_results, baseline_profiles)
    write_environment_summary(output_dir, all_results)

    print("\n" + "=" * 70)
    print("ARCHETYPE RESEARCH COMPLETE")
    print("=" * 70)
    print(f"Outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()