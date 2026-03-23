#!/usr/bin/env python3
"""
STAGE 4 - RULE STABILITY / ROBUSTNESS
=====================================

Purpose:
- Read Stage 3 best factor combinations.
- Evaluate robustness and consistency on campaign-level trades.
- Flag which combinations appear stable enough for downstream deployment consideration.

This stage does NOT own:
- winner discovery
- factor-combo generation
- deployment formatting
- Pine integration

Stage 4 inputs (from Stage 3):
- best_factor_combos_by_sid.csv
- factor_combo_validation_by_sid.csv

Core Stage 4 outputs:
- stable_factor_combos_by_sid.csv
- rule_stability_report.csv
- stage_4_stability_summary.txt

Compatibility outputs (optional via --include-compatibility-outputs):
- stable_rules.csv
- stable_environments.csv (header-only placeholder)

python stage_4_rule_stability_engine.py \
  --input-dir Stage_3_Report_ORC_RMV \
  --campaign-input campaigns_clean.csv \
  --output-dir Stage_4_Report_Final
  
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

SID_ALIASES = ["sid", "setupid", "setup_id", "SID"]
RETURN_ALIASES = ["return_pct", "return", "ret_pct"]
DATE_ALIASES = ["entry_dt", "entry_date", "date", "timestamp"]


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
        return int(float(x))
    except Exception:
        return None


def find_column(df: pd.DataFrame, aliases: List[str]) -> Optional[str]:
    lookup = {str(c).strip().lower(): c for c in df.columns}
    for alias in aliases:
        col = lookup.get(alias.lower())
        if col is not None:
            return col
    return None


def parse_setup_ids(raw: str) -> Optional[List[int]]:
    txt = str(raw).strip()
    if not txt:
        return None
    out: List[int] = []
    for token in txt.split(","):
        t = token.strip()
        if t:
            out.append(int(t))
    return sorted(set(out))


def normalize_factor_name(x: Any) -> str:
    txt = str(x).strip()
    aliases = {
        "squeeze": "SQZ",
        "squeeze_state": "SQZ",
        "sqzstate": "SQZ",
        "setupid": "sid",
        "setup_id": "sid",
    }
    return aliases.get(txt.lower(), txt)


def to_sqz_binary(s: pd.Series) -> pd.Series:
    def _map(x: Any) -> Optional[int]:
        if pd.isna(x):
            return None
        txt = str(x).strip().lower()
        if txt in ("1", "true", "on", "yes", "sqon", "fired", "exp"):
            return 1
        if txt in ("0", "false", "off", "no", "none"):
            return 0
        try:
            return 1 if float(txt) >= 0.5 else 0
        except Exception:
            return None

    return s.apply(_map)


def parse_rule_string(rule_str: Any) -> List[Dict[str, Any]]:
    if rule_str is None or (not isinstance(rule_str, str) and pd.isna(rule_str)):
        return []

    txt = str(rule_str).strip()
    if not txt:
        return []

    conditions: List[Dict[str, Any]] = []
    parts = re.split(r"\s+AND\s+", txt, flags=re.IGNORECASE)

    for part in parts:
        p = part.strip()

        m_num = re.match(r"^([A-Za-z_]+)\s*(>=|<=|>|<|=)\s*(-?\d+(?:\.\d+)?)$", p)
        if m_num:
            conditions.append(
                {
                    "factor": normalize_factor_name(m_num.group(1)),
                    "operator": m_num.group(2),
                    "value": float(m_num.group(3)),
                    "type": "numeric",
                }
            )
            continue

        m_range = re.match(
            r"^([A-Za-z_]+)\s+(?:in|IN)\s*([\[\(])\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*([\]\)])$",
            p,
        )
        if m_range:
            conditions.append(
                {
                    "factor": normalize_factor_name(m_range.group(1)),
                    "operator": "range",
                    "lower": float(m_range.group(3)),
                    "upper": float(m_range.group(4)),
                    "lower_inclusive": m_range.group(2) == "[",
                    "upper_inclusive": m_range.group(5) == "]",
                    "type": "numeric_range",
                }
            )
            continue

    return conditions


def apply_conditions(df: pd.DataFrame, conditions: List[Dict[str, Any]]) -> pd.Series:
    mask = pd.Series(True, index=df.index)

    for cond in conditions:
        factor = cond["factor"]
        if factor not in df.columns:
            return pd.Series(False, index=df.index)

        if cond["type"] == "numeric":
            col = pd.to_numeric(df[factor], errors="coerce")
            op = cond["operator"]
            val = cond["value"]
            if op == ">=":
                mask &= col >= val
            elif op == "<=":
                mask &= col <= val
            elif op == ">":
                mask &= col > val
            elif op == "<":
                mask &= col < val
            elif op == "=":
                mask &= col == val
            else:
                return pd.Series(False, index=df.index)

        elif cond["type"] == "numeric_range":
            col = pd.to_numeric(df[factor], errors="coerce")
            lower_mask = (col >= cond["lower"]) if cond["lower_inclusive"] else (col > cond["lower"])
            upper_mask = (col <= cond["upper"]) if cond["upper_inclusive"] else (col < cond["upper"])
            mask &= lower_mask & upper_mask

        else:
            return pd.Series(False, index=df.index)

    return mask.fillna(False)


def calc_expectancy_pct(df: pd.DataFrame, ret_col: str) -> float:
    r = pd.to_numeric(df[ret_col], errors="coerce").dropna()
    if r.empty:
        return np.nan
    return float(r.mean() * 100.0)


def chronological_slices(df: pd.DataFrame, dt_col: Optional[str], n_slices: int) -> List[pd.DataFrame]:
    if dt_col is None or dt_col not in df.columns or df.empty:
        return []

    dfx = df.dropna(subset=[dt_col]).sort_values(dt_col).reset_index(drop=True)
    if dfx.empty:
        return []

    idx_parts = np.array_split(np.arange(len(dfx)), max(1, n_slices))
    return [dfx.iloc[idx].copy() for idx in idx_parts if len(idx) > 0]


def load_stage3_inputs(input_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    best_path = input_dir / "best_factor_combos_by_sid.csv"
    validation_path = input_dir / "factor_combo_validation_by_sid.csv"

    if not best_path.exists():
        raise FileNotFoundError(f"Missing Stage 3 file: {best_path}")
    if not validation_path.exists():
        raise FileNotFoundError(f"Missing Stage 3 file: {validation_path}")

    best_df = pd.read_csv(best_path)
    validation_df = pd.read_csv(validation_path)

    required_best = {"SID", "Rule"}
    if not required_best.issubset(set(best_df.columns)):
        raise ValueError(f"best_factor_combos_by_sid.csv missing required columns: {sorted(required_best)}")

    return best_df, validation_df


def load_campaign_dataset(path: str, return_col_arg: str, date_col_arg: str) -> Tuple[pd.DataFrame, str, Optional[str]]:
    campaign_path = Path(path)
    if not campaign_path.exists():
        raise FileNotFoundError(f"Campaign input file not found: {campaign_path}")

    df = pd.read_csv(campaign_path)

    sid_col = find_column(df, SID_ALIASES)
    if sid_col is None:
        raise ValueError(f"Missing SetupID column. Tried aliases: {SID_ALIASES}")

    if return_col_arg:
        if return_col_arg not in df.columns:
            raise ValueError(f"Missing return column: {return_col_arg}")
        ret_col = return_col_arg
    else:
        ret_col = find_column(df, RETURN_ALIASES)
        if ret_col is None:
            raise ValueError(f"Missing return column. Tried aliases: {RETURN_ALIASES}")

    dt_col: Optional[str]
    if date_col_arg:
        dt_col = date_col_arg if date_col_arg in df.columns else None
    else:
        dt_col = find_column(df, DATE_ALIASES)

    df[sid_col] = pd.to_numeric(df[sid_col], errors="coerce")
    df[ret_col] = pd.to_numeric(df[ret_col], errors="coerce")
    df = df.dropna(subset=[sid_col, ret_col]).copy()
    df[sid_col] = df[sid_col].astype(int)

    for col in ["RS", "CMP", "TR", "VQS", "RMV", "DCR", "ORC"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "SQZ" in df.columns:
        df["SQZ"] = to_sqz_binary(df["SQZ"])

    if dt_col is not None:
        df[dt_col] = pd.to_datetime(df[dt_col], errors="coerce")

    return df, ret_col, dt_col


def classify_combo(
    trades: int,
    retention_pct: float,
    lift_pct: float,
    slices_evaluated: int,
    positive_slices: int,
    baseline_beating_slices: int,
    worst_slice_expectancy_pct: float,
    recent_slice_expectancy_pct: float,
    recent_collapse: bool,
    stability_score: float,
    min_trades: int,
    min_retention: float,
    min_lift: float,
    min_positive_slice_ratio: float,
    min_slice_expectancy_pct: float,
    stability_threshold: float,
) -> Tuple[str, str, str, bool]:
    material_lift_threshold = max(min_lift, 0.20)
    deploy_positive_ratio = max(min_positive_slice_ratio, 0.60)

    positive_ratio = (positive_slices / slices_evaluated) if slices_evaluated > 0 else 0.0
    baseline_beating_ratio = (
        baseline_beating_slices / slices_evaluated if slices_evaluated > 0 else 0.0
    )

    if trades < min_trades or slices_evaluated < 2:
        return "Reject", "fragile_low_sample", "reject", False

    if retention_pct < min_retention:
        return "Reject", "overfiltered", "reject", False

    if lift_pct < material_lift_threshold:
        return "Reject", "insufficient_lift", "reject", False

    if recent_collapse:
        if (
            trades >= (2 * min_trades)
            and retention_pct >= min_retention
            and lift_pct >= material_lift_threshold
        ):
            return "Caution", "unstable_recent_performance", "monitor", False
        return "Reject", "unstable_recent_performance", "reject", False

    is_deploy_consistent = (
        positive_ratio >= deploy_positive_ratio
        and baseline_beating_ratio >= 0.50
        and worst_slice_expectancy_pct >= min_slice_expectancy_pct
        and recent_slice_expectancy_pct >= min_slice_expectancy_pct
    )
    if is_deploy_consistent and stability_score >= stability_threshold:
        return "Stable", "stable_and_deployable", "deploy", True

    is_monitor_promising = (
        positive_ratio >= 0.50
        and worst_slice_expectancy_pct >= (min_slice_expectancy_pct - 0.40)
        and lift_pct >= material_lift_threshold
    )
    if is_monitor_promising:
        return "Caution", "promising_but_time_inconsistent", "monitor", False

    return "Reject", "reject", "reject", False


def evaluate_row(
    row: pd.Series,
    df_sid: pd.DataFrame,
    ret_col: str,
    dt_col: Optional[str],
    args: argparse.Namespace,
    sid_baseline_exp_pct: float,
) -> Dict[str, Any]:
    sid = int(row["SID"])
    rule_txt = str(row["Rule"])

    conditions = parse_rule_string(rule_txt)
    if not conditions:
        return {
            "SID": sid,
            "Rule": rule_txt,
            "Factors": row.get("Factors"),
            "Num_Factors": safe_int(row.get("Num_Factors")),
            "Stage3_Filtered_Trades": safe_int(row.get("Filtered_Trades")),
            "Stage3_Retention_Pct": safe_float(row.get("Retention_Pct")),
            "Stage3_Expectancy_Lift_Pct": safe_float(row.get("Expectancy_Lift_Pct")),
            "Total_SID_Trades": int(len(df_sid)),
            "Filtered_Trades": 0,
            "Retention_Pct": 0.0,
            "Filtered_Expectancy_Pct": np.nan,
            "Baseline_Expectancy_Pct": sid_baseline_exp_pct,
            "Expectancy_Lift_Pct": np.nan,
            "Slice_Min_Expectancy_Pct": np.nan,
            "Slice_Max_Expectancy_Pct": np.nan,
            "Positive_Slice_Ratio": np.nan,
            "Time_Slices_Evaluated": 0,
            "Positive_Slices": 0,
            "Baseline_Beating_Slices": 0,
            "Worst_Slice_Expectancy_Pct": np.nan,
            "Best_Slice_Expectancy_Pct": np.nan,
            "Recent_Slice_Expectancy_Pct": np.nan,
            "Stability_Score": 0.0,
            "Stability_Label": "Reject",
            "Stability_Reason": "reject",
            "Detailed_Stability_Reason": "reject",
            "Deployment_Recommendation": "reject",
            "Promoted_Stable": False,
        }

    mask = apply_conditions(df_sid, conditions)
    filtered = df_sid.loc[mask].copy()

    total_trades = int(len(df_sid))
    filtered_trades = int(len(filtered))
    retention_pct = (filtered_trades / total_trades * 100.0) if total_trades > 0 else np.nan
    filtered_exp_pct = calc_expectancy_pct(filtered, ret_col)
    lift_pct = filtered_exp_pct - sid_baseline_exp_pct if not np.isnan(filtered_exp_pct) else np.nan

    slice_expectancies: List[float] = []
    baseline_beating_slices = 0
    slices_evaluated = 0
    if dt_col is not None:
        min_slice_size = max(3, math.ceil(args.min_trades / max(1, args.time_slices)))
        for sl in chronological_slices(filtered, dt_col, args.time_slices):
            if len(sl) < min_slice_size:
                continue
            e = calc_expectancy_pct(sl, ret_col)
            if not np.isnan(e):
                slices_evaluated += 1
                slice_expectancies.append(e)
                if e > sid_baseline_exp_pct:
                    baseline_beating_slices += 1

    if slice_expectancies:
        slice_min = float(np.min(slice_expectancies))
        slice_max = float(np.max(slice_expectancies))
        positive_slices = int(np.sum(np.array(slice_expectancies) > 0.0))
        positive_slice_ratio = float(positive_slices / slices_evaluated) if slices_evaluated > 0 else np.nan
        baseline_beating_ratio = (
            float(baseline_beating_slices / slices_evaluated) if slices_evaluated > 0 else np.nan
        )
        recent_slice_expectancy = float(slice_expectancies[-1])
    else:
        slice_min = np.nan
        slice_max = np.nan
        positive_slices = 0
        baseline_beating_ratio = np.nan
        recent_slice_expectancy = np.nan
        positive_slice_ratio = np.nan
        slices_evaluated = 0

    recent_collapse = False
    if not np.isnan(recent_slice_expectancy):
        recent_collapse = (
            recent_slice_expectancy < args.min_slice_expectancy_pct
            or (not np.isnan(filtered_exp_pct) and recent_slice_expectancy < (filtered_exp_pct - 0.75))
            or recent_slice_expectancy < (sid_baseline_exp_pct - 0.50)
        )

    trade_score = min(filtered_trades / float(max(args.min_trades, 1) * 2), 1.0)
    retention_score = min(retention_pct / float(max(args.min_retention, 1e-9)), 1.0) if pd.notna(retention_pct) else 0.0
    material_lift_threshold = max(args.min_lift, 0.20)
    lift_score = min(max(lift_pct, 0.0) / material_lift_threshold, 1.0) if pd.notna(lift_pct) else 0.0
    consistency_score = positive_slice_ratio if pd.notna(positive_slice_ratio) else 0.0
    baseline_score = baseline_beating_ratio if pd.notna(baseline_beating_ratio) else 0.0
    recent_score = 0.0 if recent_collapse else 1.0

    stability_score = 100.0 * (
        0.30 * trade_score +
        0.20 * retention_score +
        0.20 * consistency_score +
        0.10 * baseline_score +
        0.10 * recent_score +
        0.10 * lift_score
    )

    label, reason, recommendation, promoted = classify_combo(
        trades=filtered_trades,
        retention_pct=float(retention_pct) if pd.notna(retention_pct) else 0.0,
        lift_pct=float(lift_pct) if pd.notna(lift_pct) else -np.inf,
        slices_evaluated=slices_evaluated,
        positive_slices=positive_slices,
        baseline_beating_slices=baseline_beating_slices,
        worst_slice_expectancy_pct=float(slice_min) if pd.notna(slice_min) else -np.inf,
        recent_slice_expectancy_pct=float(recent_slice_expectancy) if pd.notna(recent_slice_expectancy) else -np.inf,
        recent_collapse=recent_collapse,
        stability_score=stability_score,
        min_trades=args.min_trades,
        min_retention=args.min_retention,
        min_lift=args.min_lift,
        min_positive_slice_ratio=args.min_positive_slice_ratio,
        min_slice_expectancy_pct=args.min_slice_expectancy_pct,
        stability_threshold=args.stability_threshold,
    )

    return {
        "SID": sid,
        "Rule": rule_txt,
        "Factors": row.get("Factors"),
        "Num_Factors": safe_int(row.get("Num_Factors")),
        "Stage3_Filtered_Trades": safe_int(row.get("Filtered_Trades")),
        "Stage3_Retention_Pct": safe_float(row.get("Retention_Pct")),
        "Stage3_Expectancy_Lift_Pct": safe_float(row.get("Expectancy_Lift_Pct")),
        "Total_SID_Trades": total_trades,
        "Filtered_Trades": filtered_trades,
        "Retention_Pct": retention_pct,
        "Filtered_Expectancy_Pct": filtered_exp_pct,
        "Baseline_Expectancy_Pct": sid_baseline_exp_pct,
        "Expectancy_Lift_Pct": lift_pct,
        "Slice_Min_Expectancy_Pct": slice_min,
        "Slice_Max_Expectancy_Pct": slice_max,
        "Positive_Slice_Ratio": positive_slice_ratio,
        "Time_Slices_Evaluated": slices_evaluated,
        "Positive_Slices": positive_slices,
        "Baseline_Beating_Slices": baseline_beating_slices,
        "Worst_Slice_Expectancy_Pct": slice_min,
        "Best_Slice_Expectancy_Pct": slice_max,
        "Recent_Slice_Expectancy_Pct": recent_slice_expectancy,
        "Stability_Score": stability_score,
        "Stability_Label": label,
        "Stability_Reason": reason,
        "Detailed_Stability_Reason": reason,
        "Deployment_Recommendation": recommendation,
        "Promoted_Stable": promoted,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage 4 - Rule Stability / Robustness")
    p.add_argument("--input-dir", default="results_refinement", help="Stage 3 output directory")
    p.add_argument("--campaign-input", default="campaigns_clean.csv", help="Campaign-level dataset")
    p.add_argument("--output-dir", default="results_stability", help="Stage 4 output directory")
    p.add_argument("--setup-ids", default="", help="Comma-separated SetupIDs; empty = all available")

    p.add_argument("--return-col", default="return_pct", help="Return column in campaign input")
    p.add_argument("--date-col", default="entry_dt", help="Date/time column for chronological consistency checks")

    p.add_argument("--min-trades", type=int, default=30, help="Minimum filtered trades")
    p.add_argument("--min-retention", type=float, default=15.0, help="Minimum retention percentage")
    p.add_argument("--min-lift", type=float, default=0.0, help="Minimum lift over baseline expectancy (percentage points)")
    p.add_argument("--stability-threshold", type=float, default=60.0, help="Minimum stability score for stable promotion")

    p.add_argument("--time-slices", type=int, default=4, help="Chronological slice count for consistency checks")
    p.add_argument("--min-positive-slice-ratio", type=float, default=0.50, help="Minimum ratio of positive-expectancy slices")
    p.add_argument("--min-slice-expectancy-pct", type=float, default=-0.20, help="Minimum acceptable worst-slice expectancy percent")
    p.add_argument(
        "--include-compatibility-outputs",
        action="store_true",
        help="Also write compatibility files stable_rules.csv and stable_environments.csv",
    )
    return p.parse_args()


def run_pipeline(args: argparse.Namespace) -> None:
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    best_df, validation_df = load_stage3_inputs(input_dir)
    campaign_df, ret_col, dt_col = load_campaign_dataset(args.campaign_input, args.return_col, args.date_col)

    selected_sids = parse_setup_ids(args.setup_ids)
    if selected_sids is not None:
        best_df = best_df[pd.to_numeric(best_df["SID"], errors="coerce").isin(selected_sids)].copy()

    best_df["SID"] = pd.to_numeric(best_df["SID"], errors="coerce")
    best_df = best_df.dropna(subset=["SID", "Rule"]).copy()
    best_df["SID"] = best_df["SID"].astype(int)

    if best_df.empty:
        raise ValueError("No Stage 3 best combinations available for stability evaluation.")

    rows: List[Dict[str, Any]] = []

    print("\n" + "=" * 70)
    print("STAGE 4 - RULE STABILITY / ROBUSTNESS")
    print("=" * 70)
    print(f"Stage 3 input dir: {input_dir}")
    print(f"Campaign input: {args.campaign_input}")

    for sid in sorted(best_df["SID"].unique().tolist()):
        sid_best = best_df[best_df["SID"] == sid].copy()
        sid_trades = campaign_df[campaign_df[find_column(campaign_df, SID_ALIASES)] == sid].copy()
        if sid_trades.empty:
            print(f"SID {sid}: skipped, no campaign rows")
            continue

        baseline_exp_pct = calc_expectancy_pct(sid_trades, ret_col)

        for _, row in sid_best.iterrows():
            rows.append(
                evaluate_row(
                    row=row,
                    df_sid=sid_trades,
                    ret_col=ret_col,
                    dt_col=dt_col,
                    args=args,
                    sid_baseline_exp_pct=baseline_exp_pct,
                )
            )

    stability_df = pd.DataFrame(rows)
    if stability_df.empty:
        raise ValueError("No Stage 4 stability rows were produced.")

    candidate_counts = (
        validation_df.assign(SID=pd.to_numeric(validation_df.get("SID"), errors="coerce"))
        .dropna(subset=["SID"])
        .groupby("SID")
        .size()
        .rename("Stage3_Total_Candidates")
        .reset_index()
    )
    candidate_counts["SID"] = candidate_counts["SID"].astype(int)
    stability_df = stability_df.merge(candidate_counts, on="SID", how="left")

    stability_df = stability_df.sort_values(
        ["Promoted_Stable", "Stability_Label", "Stability_Score", "Expectancy_Lift_Pct", "Filtered_Trades"],
        ascending=[False, True, False, False, False],
        na_position="last",
    )

    stable_df = stability_df[stability_df["Deployment_Recommendation"] == "deploy"].copy()

    stability_df.to_csv(output_dir / "rule_stability_report.csv", index=False)
    stable_df.to_csv(output_dir / "stable_factor_combos_by_sid.csv", index=False)

    if args.include_compatibility_outputs:
        # Compatibility output retained for legacy consumers that still read stable_rules.csv.
        if stable_df.empty:
            stable_rules_compat = pd.DataFrame(
                columns=[
                    "subset",
                    "SID",
                    "sid",
                    "Rule",
                    "Trades",
                    "Validation_Expectancy",
                    "Validation_Expectancy_Pct",
                    "Stability_Score",
                    "Promoted_Stable",
                    "Stability_Label",
                    "Stability_Reason",
                ]
            )
        else:
            stable_rules_compat = pd.DataFrame(
                {
                    "subset": [f"sid{int(v)}" for v in stable_df["SID"]],
                    "SID": stable_df["SID"],
                    "sid": stable_df["SID"],
                    "Rule": stable_df["Rule"],
                    "Trades": stable_df["Filtered_Trades"],
                    "Validation_Expectancy": pd.to_numeric(stable_df["Filtered_Expectancy_Pct"], errors="coerce") / 100.0,
                    "Validation_Expectancy_Pct": stable_df["Filtered_Expectancy_Pct"],
                    "Stability_Score": stable_df["Stability_Score"],
                    "Promoted_Stable": True,
                    "Stability_Label": stable_df["Stability_Label"],
                    "Stability_Reason": stable_df["Stability_Reason"],
                }
            )

        stable_rules_compat.to_csv(output_dir / "stable_rules.csv", index=False)

        # Compatibility placeholder retained only for legacy file-check workflows.
        pd.DataFrame(
            columns=["SID", "GROUP_TYPE", "Trades", "Expectancy", "Environment_Stability_Score", "Promoted_Stable"]
        ).to_csv(output_dir / "stable_environments.csv", index=False)
    else:
        print("Compatibility outputs disabled (stable_rules.csv, stable_environments.csv)")

    summary_lines = [
        "STAGE 4 - RULE STABILITY / ROBUSTNESS SUMMARY",
        "=" * 70,
        f"Stage 3 input dir: {input_dir}",
        f"Campaign input: {args.campaign_input}",
        "",
        f"Total Stage 3 best combos reviewed: {len(stability_df)}",
        f"Deploy recommendations: {int((stability_df['Deployment_Recommendation'] == 'deploy').sum())}",
        f"Monitor recommendations: {int((stability_df['Deployment_Recommendation'] == 'monitor').sum())}",
        f"Reject recommendations: {int((stability_df['Deployment_Recommendation'] == 'reject').sum())}",
        f"Stable promoted combos: {len(stable_df)}",
        f"Caution combos: {int((stability_df['Stability_Label'] == 'Caution').sum())}",
        f"Rejected combos: {int((stability_df['Stability_Label'] == 'Reject').sum())}",
        "",
        "Deployable combinations:",
    ]

    if stable_df.empty:
        summary_lines.append("- none")
    else:
        for _, row in stable_df.head(25).iterrows():
            summary_lines.append(
                f"- SID {int(row['SID'])}: {row['Rule']} | recommendation={row['Deployment_Recommendation']} | "
                f"reason={row['Detailed_Stability_Reason']} | trades={int(row['Filtered_Trades'])} | "
                f"retention={row['Retention_Pct']:.1f}% | lift={row['Expectancy_Lift_Pct']:.2f}% | "
                f"score={row['Stability_Score']:.1f}"
            )

    summary_lines.append("")
    summary_lines.append("Monitor/reject combinations:")
    weak_df = stability_df[~stability_df["Promoted_Stable"]].head(25)
    if weak_df.empty:
        summary_lines.append("- none")
    else:
        for _, row in weak_df.iterrows():
            summary_lines.append(
                f"- SID {int(row['SID'])}: {row['Rule']} | recommendation={row['Deployment_Recommendation']} | "
                f"reason={row['Detailed_Stability_Reason']} | label={row['Stability_Label']} | "
                f"trades={int(row['Filtered_Trades'])} | "
                f"retention={row['Retention_Pct']:.1f}% | lift={row['Expectancy_Lift_Pct']:.2f}%"
            )

    (output_dir / "stage_4_stability_summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")

    print("\n" + "=" * 70)
    print("EXPORT COMPLETE")
    print("=" * 70)
    print(f"Outputs written to: {output_dir}")


def main() -> None:
    args = parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
