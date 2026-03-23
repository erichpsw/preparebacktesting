#!/usr/bin/env python3
"""
stage_5_profile_exporter.py

STAGE 5 — PROFILE EXPORT / PACKAGING

Purpose
-------
Read approved upstream outputs from Stages 2, 3, and 4, consolidate them
by SetupID, and package the results into clean deployment-oriented files.

Inputs
------
Stage 2 (--profiles-dir):
  - baseline_by_sid.csv                        (required)
  - winner_single_rule_validation_by_sid.csv   (optional)
  - winner_distribution_by_sid.csv             (optional)
  - deployable_rules_by_sid.csv                (optional)

Stage 3 (--refinement-dir):
  - best_factor_combos_by_sid.csv              (optional)
  - factor_combo_validation_by_sid.csv         (optional)

Stage 4 (--stability-dir):
  - stable_factor_combos_by_sid.csv            (primary, preferred over Stage 3)
  - rule_stability_report.csv                  (optional, full report)

Outputs
-------
Primary outputs:
  - final_profile_package.csv        Per-SID consolidated profile with best stable combo
  - final_rule_package.csv           Flat list of all stable/promoted rules per SID
  - setup_summary_export.csv         Overview: one row per SID

Text summary:
  - stage_5_export_summary.txt

Compatibility outputs (for Stage 6 — pine_integration_engine.py):
  - deployment_baseline_profiles.csv      Factor bounds per SID derived from best stable rule
    - deployment_archetype_adjustments.csv  Header-only placeholder (optional via --include-compatibility-outputs)

Run Example
-----------
python stage_5_profile_exporter.py \\
  --profiles-dir results_profiles \\
  --refinement-dir results_refinement \\
  --stability-dir results_stability \\
  --output-dir results_export

This stage does NOT own:
  - winner discovery (Stage 2)
  - factor combination generation (Stage 3)
  - robustness testing (Stage 4)
  - Pine integration (Stage 6)
"""

from __future__ import annotations

import argparse
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

DEFAULT_DECIMAL_PRECISION = 2

FACTOR_ORDER = ["rs", "cmp", "tr", "vqs", "rmv", "dcr", "orc", "sqz"]
NUMERIC_FACTORS = {"rs", "cmp", "tr", "vqs", "rmv", "dcr", "orc"}
CATEGORICAL_FACTORS = {"sqz"}

# Numeric sanity bounds per factor: (min_allowed, max_allowed). None means no bound.
FACTOR_SANITY_BOUNDS: Dict[str, Tuple[Optional[float], Optional[float]]] = {
    "orc": (0.0, 2.0),
    "rs":  (0.0, 300.0),
    "cmp": (0.0, 300.0),
    "tr":  (0.0, 1000.0),
    "vqs": (0.0, 300.0),
    "rmv": (0.0, 300.0),
    "dcr": (0.0, 300.0),
}


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

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


def normalize_factor_name(x: Any) -> str:
    s = str(x).strip().lower()
    aliases = {
        "squeeze": "sqz",
        "squeeze_state": "sqz",
        "sqzstate": "sqz",
    }
    return aliases.get(s, s)


def round_or_none(x: Any, decimals: int) -> Optional[float]:
    val = safe_float(x)
    if val is None:
        return None
    return round(val, decimals)


def normalize_text_list(x: Any) -> List[str]:
    if x is None or pd.isna(x):
        return []
    s = str(x).strip()
    if not s:
        return []
    parts = re.split(r"[|,;/]+", s)
    return [p.strip() for p in parts if p.strip()]


def normalize_range(
    min_v: Optional[float], max_v: Optional[float]
) -> Tuple[Optional[float], Optional[float]]:
    if min_v is None or max_v is None:
        return min_v, max_v
    if min_v > max_v:
        min_v, max_v = max_v, min_v
    return min_v, max_v


def sanity_check_numeric(factor: str, val: Optional[float]) -> Optional[float]:
    """Return val only if it falls within FACTOR_SANITY_BOUNDS; otherwise return None."""
    if val is None:
        return None
    bounds = FACTOR_SANITY_BOUNDS.get(factor)
    if bounds is None:
        return val
    lo, hi = bounds
    if lo is not None and val < lo:
        return None
    if hi is not None and val > hi:
        return None
    return val


def parse_rule_string(rule_str: Any) -> List[Dict[str, Any]]:
    """Parse a rule string (e.g. 'cmp >= 60 AND dcr >= 80') into condition dicts."""
    if rule_str is None or pd.isna(rule_str):
        return []
    txt = str(rule_str).strip()
    if not txt:
        return []

    conditions: List[Dict[str, Any]] = []
    parts = re.split(r"\s+AND\s+", txt, flags=re.IGNORECASE)

    for part in parts:
        part = part.strip()

        m_num = re.match(
            r"^([A-Za-z_]+)\s*(>=|<=|>|<|=)\s*(-?\d+(?:\.\d+)?)$", part
        )
        if m_num:
            conditions.append({
                "factor": normalize_factor_name(m_num.group(1)),
                "operator": m_num.group(2),
                "value": float(m_num.group(3)),
                "type": "numeric",
            })
            continue

        m_range = re.match(
            r"^([A-Za-z_]+)\s+(?:in|IN)\s*([\[\(])\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*([\]\)])$",
            part,
        )
        if m_range:
            conditions.append({
                "factor": normalize_factor_name(m_range.group(1)),
                "operator": "range",
                "lower": float(m_range.group(3)),
                "upper": float(m_range.group(4)),
                "lower_inclusive": m_range.group(2) == "[",
                "upper_inclusive": m_range.group(5) == "]",
                "type": "numeric_range",
            })
            continue

        m_cat = re.match(r"^([A-Za-z_]+)\s+(?:in|IN)\s+(.+)$", part)
        if m_cat:
            conditions.append({
                "factor": normalize_factor_name(m_cat.group(1)),
                "operator": "in",
                "values": normalize_text_list(m_cat.group(2)),
                "type": "categorical",
            })
            continue

    return conditions


RUNNER_SCORE_MIN_COLS = ["runner_score_min", "Runner_Score_Min"]
RUNNER_GATE_ENABLED_COLS = ["runner_gate_enabled", "Runner_Gate_Enabled"]
RUNNER_PROFILE_LABEL_COLS = ["runner_profile_label", "Runner_Profile_Label"]


def _first_present_value(row: pd.Series, columns: List[str]) -> Any:
    for col in columns:
        if col in row.index:
            return row.get(col)
    return None


def _parse_bool_or_none(x: Any) -> Optional[bool]:
    if x is None or pd.isna(x):
        return None
    if isinstance(x, bool):
        return x
    txt = str(x).strip().lower()
    if txt in ("1", "true", "yes", "y", "on"):
        return True
    if txt in ("0", "false", "no", "n", "off"):
        return False
    return None


def extract_runner_gate_from_row(row: pd.Series) -> Dict[str, Any]:
    """Pass through runner gate fields when present in approved upstream rows.

    No inference/scoring is performed here: values are exported only when
    upstream approved outputs already carry them.
    """
    min_val = safe_float(_first_present_value(row, RUNNER_SCORE_MIN_COLS))
    enabled_raw = _first_present_value(row, RUNNER_GATE_ENABLED_COLS)
    enabled = _parse_bool_or_none(enabled_raw)
    label_raw = _first_present_value(row, RUNNER_PROFILE_LABEL_COLS)
    label = None if label_raw is None or pd.isna(label_raw) else str(label_raw)

    if enabled is None:
        enabled = min_val is not None

    source_has_cols = any(
        c in row.index
        for c in (RUNNER_SCORE_MIN_COLS + RUNNER_GATE_ENABLED_COLS + RUNNER_PROFILE_LABEL_COLS)
    )
    source = "upstream" if source_has_cols else "not_available"

    return {
        "Runner_Gate_Enabled": bool(enabled),
        "Runner_Score_Min": min_val,
        "Runner_Profile_Label": label,
        "Runner_Gate_Source": source,
    }


def load_csv_if_exists(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
    except Exception as e:
        print(f"Warning: Could not read {path}: {e}")
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Input loaders
# ---------------------------------------------------------------------------

def load_stage2_inputs(profiles_dir: str) -> Optional[Dict[str, Optional[pd.DataFrame]]]:
    """Load Stage 2 winner profile discovery outputs. baseline_by_sid.csv is required."""
    pdir = Path(profiles_dir)
    baseline_path = pdir / "baseline_by_sid.csv"
    if not baseline_path.exists():
        print(f"ERROR: Required Stage 2 file not found: {baseline_path}")
        return None
    try:
        data: Dict[str, Optional[pd.DataFrame]] = {
            "baseline": pd.read_csv(baseline_path),
            "winner_single_rules": load_csv_if_exists(pdir / "winner_single_rule_validation_by_sid.csv"),
            "winner_distribution": load_csv_if_exists(pdir / "winner_distribution_by_sid.csv"),
            "deployable_rules": load_csv_if_exists(pdir / "deployable_rules_by_sid.csv"),
        }
        n_sids = len(data["baseline"])
        print(f"✓ Loaded Stage 2 inputs from {profiles_dir}  ({n_sids} baseline rows)")
        return data
    except Exception as e:
        print(f"ERROR: Could not load Stage 2 inputs: {e}")
        return None


def load_stage3_inputs(refinement_dir: Optional[str]) -> Optional[Dict[str, Optional[pd.DataFrame]]]:
    """Load Stage 3 factor rule refinement outputs. Fully optional."""
    if not refinement_dir:
        return None
    rdir = Path(refinement_dir)
    if not rdir.exists():
        print(f"WARNING: Stage 3 refinement directory not found: {refinement_dir}")
        return None
    data: Dict[str, Optional[pd.DataFrame]] = {
        "best_combos": load_csv_if_exists(rdir / "best_factor_combos_by_sid.csv"),
        "combo_validation": load_csv_if_exists(rdir / "factor_combo_validation_by_sid.csv"),
    }
    if data["best_combos"] is not None and not data["best_combos"].empty:
        print(f"✓ Loaded Stage 3 inputs from {refinement_dir}  ({len(data['best_combos'])} combos)")
    else:
        print(f"WARNING: No Stage 3 best combos found in {refinement_dir}")
    return data


def load_stage4_inputs(stability_dir: Optional[str]) -> Optional[Dict[str, Optional[pd.DataFrame]]]:
    """Load Stage 4 rule stability outputs. Optional; if absent Stage 5 falls back to Stage 3."""
    if not stability_dir:
        return None
    sdir = Path(stability_dir)
    if not sdir.exists():
        print(f"WARNING: Stage 4 stability directory not found: {stability_dir}")
        return None
    data: Dict[str, Optional[pd.DataFrame]] = {
        "stable_combos": load_csv_if_exists(sdir / "stable_factor_combos_by_sid.csv"),
        "stability_report": load_csv_if_exists(sdir / "rule_stability_report.csv"),
    }
    if data["stable_combos"] is not None and not data["stable_combos"].empty:
        print(f"✓ Loaded Stage 4 inputs from {stability_dir}  ({len(data['stable_combos'])} stable combos)")
    else:
        print(f"WARNING: No Stage 4 stable combos found in {stability_dir}; will fall back to Stage 3")
    return data


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def resolve_best_combos(
    stage3_data: Optional[Dict[str, Optional[pd.DataFrame]]],
    stage4_data: Optional[Dict[str, Optional[pd.DataFrame]]],
) -> Tuple[Optional[pd.DataFrame], str]:
    """Return the best available combo dataframe and a label indicating its source.

    Stage 4 stable combos are preferred. If absent, Stage 3 best combos are used.
    """
    if stage4_data:
        df = stage4_data.get("stable_combos")
        if df is not None and not df.empty:
            return df, "stage4_stable"
    if stage3_data:
        df = stage3_data.get("best_combos")
        if df is not None and not df.empty:
            return df, "stage3_best"
    return None, "none"


def extract_factor_bounds_from_rule(
    rule_str: Any,
    decimal_precision: int,
) -> Dict[str, Any]:
    """Extract per-factor min/max/allowed bounds from a single rule string.

    Returns a flat dict with keys {factor}_min, {factor}_max, {factor}_allowed,
    {factor}_avoided for all FACTOR_ORDER entries.
    """
    bounds: Dict[str, Any] = {}
    for factor in FACTOR_ORDER:
        bounds[f"{factor}_min"] = None
        bounds[f"{factor}_max"] = None
        bounds[f"{factor}_allowed"] = None
        bounds[f"{factor}_avoided"] = None

    conditions = parse_rule_string(rule_str)
    for cond in conditions:
        factor = cond.get("factor")
        if factor not in FACTOR_ORDER:
            continue
        ctype = cond.get("type")

        if ctype == "numeric":
            op = cond.get("operator")
            val = sanity_check_numeric(factor, safe_float(cond.get("value")))
            if val is None:
                continue
            if op in (">", ">="):
                cur = bounds.get(f"{factor}_min")
                bounds[f"{factor}_min"] = round(
                    max(cur, val) if cur is not None else val, decimal_precision
                )
            elif op in ("<", "<="):
                cur = bounds.get(f"{factor}_max")
                bounds[f"{factor}_max"] = round(
                    min(cur, val) if cur is not None else val, decimal_precision
                )
            elif op == "=":
                bounds[f"{factor}_min"] = round(val, decimal_precision)
                bounds[f"{factor}_max"] = round(val, decimal_precision)

        elif ctype == "numeric_range":
            lo = sanity_check_numeric(factor, safe_float(cond.get("lower")))
            hi = sanity_check_numeric(factor, safe_float(cond.get("upper")))
            if lo is not None:
                bounds[f"{factor}_min"] = round(lo, decimal_precision)
            if hi is not None:
                bounds[f"{factor}_max"] = round(hi, decimal_precision)

        elif ctype == "categorical":
            vals = cond.get("values", [])
            if vals:
                bounds[f"{factor}_allowed"] = "|".join(vals)

    # Normalize any min > max after accumulating all conditions
    for factor in FACTOR_ORDER:
        lo, hi = normalize_range(
            bounds.get(f"{factor}_min"), bounds.get(f"{factor}_max")
        )
        bounds[f"{factor}_min"] = lo
        bounds[f"{factor}_max"] = hi

    return bounds


def _pick_best_combo_for_sid(
    sid_combos: pd.DataFrame,
) -> pd.Series:
    """Sort combos for a single SID and return the best row.

    Priority: Stable > Caution > other, then highest Expectancy_Lift_Pct.
    """
    df = sid_combos.copy()
    sort_cols: List[str] = []
    sort_asc: List[bool] = []

    if "Stability_Label" in df.columns:
        label_rank = {"Stable": 0, "Caution": 1, "Reject": 2}
        df["_label_rank"] = df["Stability_Label"].map(label_rank).fillna(3)
        sort_cols.append("_label_rank")
        sort_asc.append(True)

    if "Expectancy_Lift_Pct" in df.columns:
        sort_cols.append("Expectancy_Lift_Pct")
        sort_asc.append(False)

    if sort_cols:
        df = df.sort_values(sort_cols, ascending=sort_asc)

    return df.iloc[0]


def consolidate_by_sid(
    stage2_data: Dict[str, Optional[pd.DataFrame]],
    best_combos: Optional[pd.DataFrame],
    combo_source: str,
    decimal_precision: int,
) -> List[Dict[str, Any]]:
    """Consolidate upstream outputs into one summary row per SetupID.

    Each row carries:
    - Stage 2 baseline metrics
    - best available combo (Stage 4 or Stage 3) with its rule and performance metrics
    - factor bounds extracted from that best rule (used for Stage 6 compat export)
    """
    baseline_df = stage2_data["baseline"]
    sids = sorted(
        pd.to_numeric(baseline_df["SID"], errors="coerce")
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )

    rows: List[Dict[str, Any]] = []
    for sid in sids:
        base_match = baseline_df[
            pd.to_numeric(baseline_df["SID"], errors="coerce") == sid
        ]
        if base_match.empty:
            continue
        base = base_match.iloc[0]

        row: Dict[str, Any] = {
            "SID": sid,
            # --- Stage 2 baseline context ---
            "Baseline_Trades": safe_int(base.get("Trades")),
            "Baseline_Win_Rate": round_or_none(base.get("Win_Rate"), decimal_precision),
            "Baseline_Expectancy_Pct": round_or_none(base.get("Expectancy_Pct"), decimal_precision),
            "Baseline_Median_Return_Pct": round_or_none(base.get("Median_Return_Pct"), decimal_precision),
            "Baseline_Payoff_Ratio": round_or_none(base.get("Payoff_Ratio"), decimal_precision),
            # --- Best approved combo (Stage 4 preferred, Stage 3 fallback) ---
            "Best_Rule": None,
            "Best_Factors": None,
            "Best_Num_Factors": None,
            "Best_Filtered_Trades": None,
            "Best_Retention_Pct": None,
            "Best_Expectancy_Pct": None,
            "Best_Expectancy_Lift_Pct": None,
            "Best_Win_Rate": None,
            "Stability_Label": None,
            "Stability_Score": None,
            "Stability_Reason": None,
            "Combo_Source": combo_source,
            # Runner gate pass-through from upstream approved outputs only.
            "Runner_Gate_Enabled": False,
            "Runner_Score_Min": None,
            "Runner_Profile_Label": None,
            "Runner_Gate_Source": "not_available",
        }

        # Factor bounds (populated from best rule, used for Stage 6 compat)
        for factor in FACTOR_ORDER:
            row[f"{factor}_min"] = None
            row[f"{factor}_max"] = None
            row[f"{factor}_allowed"] = None
            row[f"{factor}_avoided"] = None

        if best_combos is not None and not best_combos.empty:
            sid_combos = best_combos[
                pd.to_numeric(best_combos["SID"], errors="coerce") == sid
            ]
            if not sid_combos.empty:
                best = _pick_best_combo_for_sid(sid_combos)
                rule_txt = best.get("Rule")
                row["Best_Rule"] = rule_txt
                row["Best_Factors"] = best.get("Factors")
                row["Best_Num_Factors"] = safe_int(best.get("Num_Factors"))
                row["Best_Filtered_Trades"] = safe_int(best.get("Filtered_Trades"))
                row["Best_Retention_Pct"] = round_or_none(
                    best.get("Retention_Pct"), decimal_precision
                )
                row["Best_Expectancy_Pct"] = round_or_none(
                    best.get("Filtered_Expectancy_Pct"), decimal_precision
                )
                row["Best_Expectancy_Lift_Pct"] = round_or_none(
                    best.get("Expectancy_Lift_Pct"), decimal_precision
                )
                row["Best_Win_Rate"] = round_or_none(
                    best.get("Filtered_Win_Rate"), decimal_precision
                )
                if "Stability_Label" in best.index:
                    row["Stability_Label"] = best.get("Stability_Label")
                    row["Stability_Score"] = round_or_none(
                        best.get("Stability_Score"), decimal_precision
                    )
                    row["Stability_Reason"] = best.get("Stability_Reason")
                elif "Promoted" in best.index:
                    # Stage 3 fallback: map Promoted flag to a label
                    row["Stability_Label"] = (
                        "Stable" if bool(best.get("Promoted")) else "Caution"
                    )

                runner_gate = extract_runner_gate_from_row(best)
                row.update(runner_gate)

                # Extract factor bounds for Stage 6 compat
                bounds = extract_factor_bounds_from_rule(rule_txt, decimal_precision)
                for factor in FACTOR_ORDER:
                    row[f"{factor}_min"] = bounds[f"{factor}_min"]
                    row[f"{factor}_max"] = bounds[f"{factor}_max"]
                    row[f"{factor}_allowed"] = bounds[f"{factor}_allowed"]
                    row[f"{factor}_avoided"] = bounds[f"{factor}_avoided"]

        rows.append(row)

    return rows


def build_final_rule_package(
    best_combos: Optional[pd.DataFrame],
    combo_source: str,
    decimal_precision: int,
) -> pd.DataFrame:
    """Flatten all stable/promoted combos into a deployment rule table.

    Excludes Reject-labelled combos. For Stage 3 fallback data, only includes
    rows where Promoted is True.
    """
    if best_combos is None or best_combos.empty:
        return pd.DataFrame()

    rows: List[Dict[str, Any]] = []
    for _, row in best_combos.iterrows():
        sid = safe_int(row.get("SID"))
        rule_txt = row.get("Rule")
        label = row.get("Stability_Label")
        promoted = row.get("Promoted_Stable", row.get("Promoted"))

        # Skip explicitly rejected combos
        if label is not None and str(label) == "Reject":
            continue
        # For Stage 3 fallback data without a stability label, skip un-promoted rows
        if label is None and promoted is not None and not bool(promoted):
            continue

        r: Dict[str, Any] = {
            "SID": sid,
            "Rule": rule_txt,
            "Factors": row.get("Factors"),
            "Num_Factors": safe_int(row.get("Num_Factors")),
            "Filtered_Trades": safe_int(row.get("Filtered_Trades")),
            "Retention_Pct": round_or_none(row.get("Retention_Pct"), decimal_precision),
            "Filtered_Expectancy_Pct": round_or_none(
                row.get("Filtered_Expectancy_Pct"), decimal_precision
            ),
            "Baseline_Expectancy_Pct": round_or_none(
                row.get("Baseline_Expectancy_Pct"), decimal_precision
            ),
            "Expectancy_Lift_Pct": round_or_none(
                row.get("Expectancy_Lift_Pct"), decimal_precision
            ),
            "Filtered_Win_Rate": round_or_none(
                row.get("Filtered_Win_Rate"), decimal_precision
            ),
            "Stability_Label": label,
            "Stability_Score": round_or_none(
                row.get("Stability_Score"), decimal_precision
            ),
            "Stability_Reason": row.get("Stability_Reason"),
            "Promoted_Stable": bool(row.get("Promoted_Stable", row.get("Promoted", False))),
            "Combo_Source": combo_source,
        }
        r.update(extract_runner_gate_from_row(row))
        rows.append(r)

    return pd.DataFrame(rows)


def build_setup_summary(
    consolidated_rows: List[Dict[str, Any]],
    best_combos: Optional[pd.DataFrame],
) -> pd.DataFrame:
    """Build a one-row-per-SID overview table."""
    combo_counts: Dict[int, Dict[str, Any]] = {}
    if (
        best_combos is not None
        and not best_combos.empty
        and "Stability_Label" in best_combos.columns
    ):
        for sid_val, grp in best_combos.groupby(
            pd.to_numeric(best_combos["SID"], errors="coerce")
        ):
            sid_int = safe_int(sid_val)
            if sid_int is None:
                continue
            combo_counts[sid_int] = {
                "Stable_Combos": int((grp["Stability_Label"] == "Stable").sum()),
                "Caution_Combos": int((grp["Stability_Label"] == "Caution").sum()),
                "Reject_Combos": int((grp["Stability_Label"] == "Reject").sum()),
                "Total_Combos": len(grp),
            }

    rows: List[Dict[str, Any]] = []
    for cr in consolidated_rows:
        sid = cr["SID"]
        counts = combo_counts.get(sid, {})
        rows.append({
            "SID": sid,
            "Baseline_Trades": cr["Baseline_Trades"],
            "Baseline_Expectancy_Pct": cr["Baseline_Expectancy_Pct"],
            "Baseline_Win_Rate": cr["Baseline_Win_Rate"],
            "Best_Rule": cr["Best_Rule"],
            "Best_Filtered_Trades": cr["Best_Filtered_Trades"],
            "Best_Expectancy_Lift_Pct": cr["Best_Expectancy_Lift_Pct"],
            "Best_Stability_Label": cr["Stability_Label"],
            "Best_Stability_Score": cr["Stability_Score"],
            "Total_Combos": counts.get("Total_Combos"),
            "Stable_Combos": counts.get("Stable_Combos"),
            "Caution_Combos": counts.get("Caution_Combos"),
            "Reject_Combos": counts.get("Reject_Combos"),
            "Combo_Source": cr["Combo_Source"],
            "Runner_Gate_Enabled": cr.get("Runner_Gate_Enabled"),
            "Runner_Score_Min": cr.get("Runner_Score_Min"),
            "Runner_Profile_Label": cr.get("Runner_Profile_Label"),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Compatibility outputs for Stage 6
# ---------------------------------------------------------------------------

def build_deployment_baseline_profiles(
    consolidated_rows: List[Dict[str, Any]],
) -> pd.DataFrame:
    """Build deployment_baseline_profiles.csv for Stage 6 compatibility.

    Compatibility output: stage_6_pine_integration_engine.py reads this file.
    It expects columns: SID, {factor}_min, {factor}_max, {factor}_allowed,
    {factor}_avoided  (plus optional baseline metrics).

    Factor bounds are derived from the best stable rule per SID. If a SID has
    no approved combo, its factor bound columns are left empty (None/NaN).
    """
    rows: List[Dict[str, Any]] = []
    for cr in consolidated_rows:
        row: Dict[str, Any] = {
            "SID": cr["SID"],
            # Baseline metrics included for Stage 6 context
            "expectancy_pct": cr.get("Baseline_Expectancy_Pct"),
            "win_rate": cr.get("Baseline_Win_Rate"),
            "trades": cr.get("Baseline_Trades"),
            "payoff_ratio": cr.get("Baseline_Payoff_Ratio"),
            # Runner gate fields are pass-through only (no Stage 5 inference).
            "runner_gate_enabled": cr.get("Runner_Gate_Enabled"),
            "runner_score_min": cr.get("Runner_Score_Min"),
            "runner_profile_label": cr.get("Runner_Profile_Label"),
        }
        for factor in FACTOR_ORDER:
            row[f"{factor}_min"] = cr.get(f"{factor}_min")
            row[f"{factor}_max"] = cr.get(f"{factor}_max")
            row[f"{factor}_allowed"] = cr.get(f"{factor}_allowed")
            row[f"{factor}_avoided"] = cr.get(f"{factor}_avoided")
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Text summary
# ---------------------------------------------------------------------------

def write_stage5_summary_text(
    consolidated_rows: List[Dict[str, Any]],
    best_combos: Optional[pd.DataFrame],
    combo_source: str,
    output_dir: Path,
    include_compatibility_outputs: bool,
    runner_fields_present_upstream: bool,
) -> None:
    """Write a plain-text summary of the Stage 5 export."""
    path = output_dir / "stage_5_export_summary.txt"

    stable_count = caution_count = reject_count = 0
    if best_combos is not None and not best_combos.empty and "Stability_Label" in best_combos.columns:
        stable_count = int((best_combos["Stability_Label"] == "Stable").sum())
        caution_count = int((best_combos["Stability_Label"] == "Caution").sum())
        reject_count = int((best_combos["Stability_Label"] == "Reject").sum())

    with open(path, "w", encoding="utf-8") as f:
        f.write("STAGE 5 — PROFILE EXPORT SUMMARY\n")
        f.write("=" * 60 + "\n")
        f.write(f"Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Rule source: {combo_source}\n\n")
        f.write(f"SetupIDs exported : {len(consolidated_rows)}\n")
        if best_combos is not None and not best_combos.empty:
            total = len(best_combos)
            f.write(f"Total combos available: {total}\n")
            if "Stability_Label" in best_combos.columns:
                f.write(f"  Stable  : {stable_count}\n")
                f.write(f"  Caution : {caution_count}\n")
                f.write(f"  Reject  : {reject_count}\n")
        f.write("\n")

        f.write("PER-SETUPID SUMMARY\n")
        f.write("-" * 60 + "\n")
        for cr in consolidated_rows:
            sid = cr["SID"]
            f.write(f"\nSID {sid}:\n")
            f.write(
                f"  Baseline : {cr['Baseline_Trades']} trades, "
                f"{cr['Baseline_Expectancy_Pct']}% expectancy, "
                f"win rate {cr['Baseline_Win_Rate']}\n"
            )
            if cr.get("Best_Rule"):
                f.write(f"  Best Rule: {cr['Best_Rule']}\n")
                f.write(
                    f"  Filtered : {cr['Best_Filtered_Trades']} trades, "
                    f"retention {cr['Best_Retention_Pct']}%, "
                    f"lift {cr['Best_Expectancy_Lift_Pct']}%\n"
                )
                f.write(
                    f"  Stability: {cr['Stability_Label']} "
                    f"(score: {cr['Stability_Score']}, reason: {cr['Stability_Reason']})\n"
                )
            else:
                f.write("  Best Rule: None (no approved combos available for this SID)\n")
            f.write(
                f"  Runner Gate: enabled={cr.get('Runner_Gate_Enabled')} | "
                f"min={cr.get('Runner_Score_Min')} | label={cr.get('Runner_Profile_Label')}\n"
            )

        f.write("\nRUNNER GATE STATUS\n")
        f.write("-" * 60 + "\n")
        enabled_count = sum(1 for cr in consolidated_rows if bool(cr.get("Runner_Gate_Enabled")))
        f.write(f"Runner-gated SetupIDs: {enabled_count}/{len(consolidated_rows)}\n")
        if not runner_fields_present_upstream:
            f.write(
                "Runner Score gate is not present in current approved upstream outputs; "
                "Stage 5 exported runner gate fields as disabled/empty pass-through defaults.\n"
            )

        f.write("\n" + "=" * 60 + "\n")
        f.write("OUTPUT FILES\n")
        f.write("-" * 60 + "\n")
        f.write("  final_profile_package.csv           — per-SID consolidated profile\n")
        f.write("  final_rule_package.csv              — all stable/promoted rules\n")
        f.write("  setup_summary_export.csv            — overview table\n")
        f.write("  stage_5_export_summary.txt          — this file\n")
        f.write("  deployment_baseline_profiles.csv    — Stage 6 compat: factor bounds per SID\n")
        if include_compatibility_outputs:
            f.write("  deployment_archetype_adjustments.csv — Stage 6 compat: header-only placeholder\n")

    print("✓ Exported: stage_5_export_summary.txt")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 5 — Profile Export / Packaging",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Reads upstream outputs from Stages 2, 3, and 4.\n"
            "Packages approved stable rule combinations for deployment and Stage 6 integration."
        ),
    )
    parser.add_argument(
        "--profiles-dir", "-p",
        required=True,
        help="Stage 2 output directory (must contain baseline_by_sid.csv)",
    )
    parser.add_argument(
        "--refinement-dir", "-r",
        default=None,
        help="Stage 3 output directory containing best_factor_combos_by_sid.csv (optional)",
    )
    parser.add_argument(
        "--stability-dir", "-s",
        default=None,
        help="Stage 4 output directory containing stable_factor_combos_by_sid.csv (optional)",
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="results_export",
        help="Output directory for Stage 5 export files (default: results_export)",
    )
    parser.add_argument(
        "--setup-ids",
        default=None,
        help="Comma-separated list of SetupIDs to process (default: all from Stage 2 baseline)",
    )
    parser.add_argument(
        "--decimal-precision",
        type=int,
        default=DEFAULT_DECIMAL_PRECISION,
        help=f"Decimal places for numeric outputs (default: {DEFAULT_DECIMAL_PRECISION})",
    )
    parser.add_argument(
        "--include-compatibility-outputs",
        action="store_true",
        help="Also write compatibility files such as deployment_archetype_adjustments.csv",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------

def run_pipeline(args: argparse.Namespace) -> None:
    print("\n" + "=" * 70)
    print("STAGE 5 — PROFILE EXPORT / PACKAGING")
    print("=" * 70)

    # Load Stage 2 inputs (required)
    stage2_data = load_stage2_inputs(args.profiles_dir)
    if stage2_data is None:
        print("ERROR: Cannot proceed without Stage 2 baseline data.")
        raise SystemExit(1)

    # Load Stage 3 inputs (optional)
    stage3_data = load_stage3_inputs(args.refinement_dir)

    # Load Stage 4 inputs (optional)
    stage4_data = load_stage4_inputs(args.stability_dir)

    # Resolve best combos: Stage 4 preferred, Stage 3 fallback
    best_combos, combo_source = resolve_best_combos(stage3_data, stage4_data)
    if best_combos is None:
        print(
            "WARNING: No combo data available from Stage 3 or Stage 4. "
            "Export will contain baseline-only rows."
        )
    else:
        print(f"✓ Using combos from: {combo_source}  ({len(best_combos)} rows)")

    runner_fields_present_upstream = False
    if best_combos is not None:
        runner_fields_present_upstream = any(
            c in best_combos.columns
            for c in (RUNNER_SCORE_MIN_COLS + RUNNER_GATE_ENABLED_COLS + RUNNER_PROFILE_LABEL_COLS)
        )

    # Filter by requested setup IDs if provided
    if args.setup_ids:
        requested = {
            int(x.strip())
            for x in args.setup_ids.split(",")
            if x.strip().isdigit()
        }
        baseline_df = stage2_data["baseline"]
        stage2_data["baseline"] = baseline_df[
            pd.to_numeric(baseline_df["SID"], errors="coerce").isin(requested)
        ].copy()
        if best_combos is not None:
            best_combos = best_combos[
                pd.to_numeric(best_combos["SID"], errors="coerce").isin(requested)
            ].copy()
        print(f"Filtered to setup IDs: {sorted(requested)}")

    # Consolidate by SID
    consolidated_rows = consolidate_by_sid(
        stage2_data, best_combos, combo_source, args.decimal_precision
    )
    if not consolidated_rows:
        print("ERROR: No SetupIDs found in Stage 2 baseline. Nothing to export.")
        raise SystemExit(1)
    print(f"✓ Consolidated {len(consolidated_rows)} SetupIDs")

    # Build output dataframes
    final_profile_df = pd.DataFrame(consolidated_rows)
    final_rule_df = build_final_rule_package(best_combos, combo_source, args.decimal_precision)
    setup_summary_df = build_setup_summary(consolidated_rows, best_combos)
    # Compatibility output for Stage 6 — derived from consolidated factor bounds
    deployment_profiles_df = build_deployment_baseline_profiles(consolidated_rows)

    # Write outputs
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)
    print(f"\nOutput directory: {output_dir}\n")

    # Primary Stage 5 outputs
    final_profile_df.to_csv(output_dir / "final_profile_package.csv", index=False)
    print("✓ Exported: final_profile_package.csv")

    final_rule_df.to_csv(output_dir / "final_rule_package.csv", index=False)
    print("✓ Exported: final_rule_package.csv")

    setup_summary_df.to_csv(output_dir / "setup_summary_export.csv", index=False)
    print("✓ Exported: setup_summary_export.csv")

    write_stage5_summary_text(
        consolidated_rows,
        best_combos,
        combo_source,
        output_dir,
        include_compatibility_outputs=args.include_compatibility_outputs,
        runner_fields_present_upstream=runner_fields_present_upstream,
    )

    # Compatibility outputs for Stage 6 (pine_integration_engine.py)
    # Stage 6 reads deployment_baseline_profiles.csv for factor bounds per SID.
    deployment_profiles_df.to_csv(output_dir / "deployment_baseline_profiles.csv", index=False)
    print("✓ Exported: deployment_baseline_profiles.csv  [Stage 6 compat]")

    if args.include_compatibility_outputs:
        # Optional compatibility output for older Stage 6 workflows.
        # The archetype layer is no longer active in the default pipeline.
        _arch_cols = (
            ["sid", "archetype", "baseline_unchanged"]
            + [f"{f}_min_override" for f in FACTOR_ORDER]
            + [f"{f}_max_override" for f in FACTOR_ORDER]
            + [f"{f}_allowed_override" for f in FACTOR_ORDER]
            + [f"{f}_avoided_override" for f in FACTOR_ORDER]
        )
        pd.DataFrame(columns=_arch_cols).to_csv(
            output_dir / "deployment_archetype_adjustments.csv", index=False
        )
        print("✓ Exported: deployment_archetype_adjustments.csv  [Stage 6 compat, header-only]")
    else:
        print("Compatibility outputs disabled (deployment_archetype_adjustments.csv)")

    print("\n" + "=" * 70)
    print("EXPORT COMPLETE")
    print("=" * 70 + "\n")
    print(f"All export files saved to: {output_dir}\n")

    print("Output files:")
    for file in sorted(output_dir.glob("*")):
        if file.is_file():
            size = file.stat().st_size
            size_str = (
                f"{size:,} bytes" if size < 1024 * 1024 else f"{size / (1024 * 1024):.1f} MB"
            )
            print(f"  {file.name:<52} {size_str:>15}")


def main() -> None:
    args = parse_arguments()
    run_pipeline(args)


if __name__ == "__main__":
    main()
