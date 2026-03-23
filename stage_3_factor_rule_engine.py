#!/usr/bin/env python3
"""
STAGE 3 - FACTOR RULE REFINEMENT
================================

Purpose:
- Read Stage 2 candidate rules
- Test disciplined 2-factor / 3-factor combinations
- Identify practically useful rule sets by SetupID

This stage does NOT own:
- winner discovery
- stability testing
- deployment formatting
- Pine integration

Stage 3 inputs (from Stage 2):
- baseline_by_sid.csv
- winner_single_rule_validation_by_sid.csv

Core Stage 3 outputs:
- factor_combo_validation_by_sid.csv
- best_factor_combos_by_sid.csv
- stage_3_refinement_summary.txt

Compatibility outputs retained for downstream pipeline compatibility:
- deployable_rules.csv
- expectancy_lift_report.csv

Strict Mode
python stage_3_factor_rule_engine.py --input-dir Stage_2_Report --campaign-input campaigns_clean.csv --output-dir Stage_3_Report

Expanded Mode
python stage_3_factor_rule_engine.py --input-dir Stage_2_Report --campaign-input campaigns_clean.csv --output-dir Stage_3_Report --allow-tier-2-seeds

python stage_3_factor_rule_engine.py \
  --input-dir Stage_2_Report \
  --campaign-input campaigns_clean.csv \
  --output-dir Stage_3_Report_Tier2_3F \
  --allow-tier-2-seeds \
  --allow-3factor
"""

from __future__ import annotations

import argparse
import itertools
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

CORE_FACTORS = ["RS", "CMP", "TR", "VQS", "SQZ"]
SECONDARY_FACTORS = ["RMV", "DCR", "ORC"]
NUMERIC_FACTORS = {"RS", "CMP", "TR", "VQS", "RMV", "DCR", "ORC"}
BINARY_FACTORS = {"SQZ"}
MAX_COMBO_TESTS_PER_SID = 200

# Disciplined structure rules for 3-factor combinations.
THREE_FACTOR_CORE_GROUP = {"RS", "TR", "CMP", "VQS"}
THREE_FACTOR_SECONDARY_GROUP = {"ORC", "RMV", "SQZ"}

SID_ALIASES = ["sid", "setupid", "setup_id"]
RETURN_ALIASES = ["return_pct", "return", "ret_pct"]


@dataclass(frozen=True)
class Condition:
    factor: str
    kind: str  # ge, le, range, eq
    value: Any

    def to_rule_text(self) -> str:
        if self.kind == "ge":
            return f"{self.factor} >= {float(self.value):.2f}"
        if self.kind == "le":
            return f"{self.factor} <= {float(self.value):.2f}"
        if self.kind == "range":
            lo, hi = self.value
            return f"{self.factor} in [{float(lo):.2f}, {float(hi):.2f}]"
        if self.kind == "eq":
            return f"{self.factor} = {int(self.value)}"
        raise ValueError(f"Unsupported condition kind: {self.kind}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage 3 - Factor Rule Refinement")
    p.add_argument("--input-dir", default="results_profiles", help="Stage 2 output directory")
    p.add_argument("--campaign-input", default="campaigns_clean.csv", help="Campaign-level dataset")
    p.add_argument("--output-dir", default="results_refinement")
    p.add_argument("--setup-ids", default="", help="Comma-separated SetupIDs; empty = all available")

    p.add_argument("--top-k-single", type=int, default=8, help="Max Stage 2 single rules used as seeds per SID")
    p.add_argument("--top-k-combo", type=int, default=20, help="Max combos retained per SID")
    p.add_argument("--min-trades", type=int, default=30, help="Minimum filtered trades for practical combo use")
    p.add_argument("--min-retention", type=float, default=12.0, help="Minimum combo retention percentage")
    p.add_argument("--allow-3factor", action="store_true", help="Enable 3-factor combos in addition to 2-factor combos")
    p.add_argument("--include-secondary-factors", action="store_true", help="Allow RMV/DCR/ORC combos")
    p.add_argument("--allow-tier-2-seeds", action="store_true", help="Allow Tier 2 seeds to expand combo mining pool")

    p.add_argument("--seed-min-trades", type=int, default=50)
    p.add_argument("--seed-min-retention", type=float, default=12.0)
    p.add_argument("--seed-min-enrichment", type=float, default=1.02)
    p.add_argument("--seed-min-lift", type=float, default=0.00)

    return p.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def find_column(df: pd.DataFrame, aliases: List[str]) -> Optional[str]:
    lookup = {str(c).strip().lower(): c for c in df.columns}
    for alias in aliases:
        if alias.lower() in lookup:
            return lookup[alias.lower()]
    return None


def parse_setup_ids(raw: str) -> Optional[List[int]]:
    if not raw.strip():
        return None
    out: List[int] = []
    for token in raw.split(","):
        tok = token.strip()
        if tok:
            out.append(int(tok))
    return sorted(set(out))


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
            v = float(txt)
            return 1 if v >= 0.5 else 0
        except Exception:
            return None

    return s.apply(_map)


def basic_metrics(df: pd.DataFrame, ret_col: str) -> Dict[str, Any]:
    r = pd.to_numeric(df[ret_col], errors="coerce").dropna()
    if r.empty:
        return {
            "trades": 0,
            "win_rate": np.nan,
            "expectancy": np.nan,
            "median_return": np.nan,
        }

    return {
        "trades": int(len(r)),
        "win_rate": float((r > 0).mean()),
        "expectancy": float(r.mean()),
        "median_return": float(r.median()),
    }


def parse_rule_text(rule: str) -> List[Condition]:
    txt = str(rule).strip()
    if not txt:
        return []

    conditions: List[Condition] = []
    for part in txt.split(" AND "):
        p = part.strip()
        if " in [" in p and p.endswith("]"):
            factor, rhs = p.split(" in [", 1)
            lo_txt, hi_txt = rhs[:-1].split(",", 1)
            conditions.append(Condition(factor=factor.strip(), kind="range", value=(float(lo_txt.strip()), float(hi_txt.strip()))))
        elif ">=" in p:
            factor, rhs = p.split(">=", 1)
            conditions.append(Condition(factor=factor.strip(), kind="ge", value=float(rhs.strip())))
        elif "<=" in p:
            factor, rhs = p.split("<=", 1)
            conditions.append(Condition(factor=factor.strip(), kind="le", value=float(rhs.strip())))
        elif "=" in p:
            factor, rhs = p.split("=", 1)
            conditions.append(Condition(factor=factor.strip(), kind="eq", value=int(float(rhs.strip()))))
        else:
            raise ValueError(f"Unsupported rule text: {rule}")
    return conditions


def condition_mask(df: pd.DataFrame, col: str, cond: Condition) -> pd.Series:
    if cond.kind == "ge":
        s = pd.to_numeric(df[col], errors="coerce")
        return s >= float(cond.value)
    if cond.kind == "le":
        s = pd.to_numeric(df[col], errors="coerce")
        return s <= float(cond.value)
    if cond.kind == "range":
        lo, hi = cond.value
        s = pd.to_numeric(df[col], errors="coerce")
        return (s >= float(lo)) & (s <= float(hi))
    if cond.kind == "eq":
        s = pd.to_numeric(df[col], errors="coerce")
        return s == int(cond.value)
    raise ValueError(f"Unsupported condition kind: {cond.kind}")


def evaluate_combo(
    df_sid: pd.DataFrame,
    ret_col: str,
    baseline_expectancy: float,
    conditions: List[Tuple[Condition, str]],
) -> Dict[str, Any]:
    mask = pd.Series(True, index=df_sid.index)
    for cond, col in conditions:
        mask &= condition_mask(df_sid, col, cond).fillna(False)

    filtered = df_sid[mask].copy()
    stats = basic_metrics(filtered, ret_col)

    trades = int(stats["trades"])
    retention = (trades / len(df_sid) * 100.0) if len(df_sid) else np.nan
    expectancy = stats["expectancy"]
    lift = expectancy - baseline_expectancy if pd.notna(expectancy) and pd.notna(baseline_expectancy) else np.nan

    return {
        "rule": " AND ".join(c.to_rule_text() for c, _ in conditions),
        "num_factors": len(conditions),
        "trades": trades,
        "retention_pct": retention,
        "expectancy": expectancy,
        "win_rate": stats["win_rate"],
        "median_return": stats["median_return"],
        "lift": lift,
        "lift_pct": (lift * 100.0) if pd.notna(lift) else np.nan,
    }


def to_bool_series(s: pd.Series) -> pd.Series:
    def _as_bool(x: Any) -> bool:
        if pd.isna(x):
            return False
        txt = str(x).strip().lower()
        if txt in {"true", "1", "yes", "y", "t"}:
            return True
        if txt in {"false", "0", "no", "n", "f", ""}:
            return False
        return bool(x)

    return s.apply(_as_bool)


def choose_seed_pool(sid_singles: pd.DataFrame, allow_tier_2_seeds: bool, args: argparse.Namespace) -> pd.DataFrame:
    df = sid_singles.copy()

    for col in ["Filtered_Trades", "Retention_Pct", "Expectancy_Lift_Pct", "Winner_Enrichment"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    broad = df[
        (df["Filtered_Trades"] >= args.seed_min_trades)
        & (df["Retention_Pct"] >= args.seed_min_retention)
        & (df["Expectancy_Lift_Pct"] >= args.seed_min_lift)
    ].copy()

    if "Winner_Enrichment" in broad.columns:
        broad = broad[broad["Winner_Enrichment"] >= args.seed_min_enrichment].copy()

    broad_factor_count = broad["Factor"].astype(str).nunique() if "Factor" in broad.columns else 0
    fallback_base = df[
        (df["Filtered_Trades"] >= args.min_trades)
        & (df["Retention_Pct"] >= args.min_retention)
        & (df["Expectancy_Lift_Pct"] >= args.seed_min_lift)
    ].copy()
    if fallback_base.empty:
        fallback_base = df.copy()
    fallback = fallback_base.drop_duplicates(subset=["Factor"], keep="first").head(2).copy()

    if allow_tier_2_seeds:
        if broad_factor_count < 2:
            return fallback
        return broad

    strict = pd.DataFrame()

    if "Downstream_Eligible" in df.columns:
        strict = df[to_bool_series(df["Downstream_Eligible"])].copy()
    elif "Promoted" in df.columns:
        strict = df[to_bool_series(df["Promoted"])].copy()

    if strict.empty:
        if broad_factor_count < 2:
            return fallback
        return broad

    strict_factor_count = strict["Factor"].astype(str).nunique() if "Factor" in strict.columns else 0

    if strict_factor_count < 2:
        if broad_factor_count < 2:
            return fallback
        return broad

    return strict


def combo_structure_metadata(factors: List[str], combo_order: int) -> Tuple[int, bool, str]:
    """
    Return (core_factor_count, is_valid_structure, invalid_reason).
    For 2-factor combos, structure is always valid.
    For 3-factor combos, enforce disciplined composition constraints.
    """
    core_count = sum(1 for f in factors if f in THREE_FACTOR_CORE_GROUP)
    if combo_order != 3:
        return core_count, True, ""

    if len(set(factors)) != len(factors):
        return core_count, False, "duplicate_factor"
    if core_count < 2:
        return core_count, False, "insufficient_core_factors"

    secondary_count = sum(1 for f in factors if f in THREE_FACTOR_SECONDARY_GROUP)
    if secondary_count > 1:
        return core_count, False, "multiple_secondary_factors"

    disallowed = [f for f in factors if f not in THREE_FACTOR_CORE_GROUP and f not in THREE_FACTOR_SECONDARY_GROUP]
    if disallowed:
        return core_count, False, f"disallowed_factor_group:{'|'.join(disallowed)}"

    return core_count, True, ""


# ---------------------------------------------------------------------------
# Incremental factor contribution analysis helpers
# ---------------------------------------------------------------------------

def find_best_base_combo(
    sid_combo_rows: List[Dict[str, Any]],
    factors_3f: List[str],
) -> Optional[Dict[str, Any]]:
    """
    For a 3-factor combo described by `factors_3f`, find the best-performing
    2-factor subset already evaluated in `sid_combo_rows`.

    Selection priority (descending):
      1. Filtered_Expectancy_Pct
      2. Expectancy_Lift_Pct
      3. Retention_Pct
      4. Filtered_Trades

    Returns None if no matching 2-factor subset was evaluated.
    """
    factor_set = set(factors_3f)
    candidates: List[Dict[str, Any]] = []
    for row in sid_combo_rows:
        if row.get("Num_Factors") != 2:
            continue
        row_factors = set(str(row.get("Factors", "")).split("|"))
        if len(row_factors) != 2 or not row_factors.issubset(factor_set):
            continue
        candidates.append(row)

    if not candidates:
        return None

    def _sort_key(r: Dict[str, Any]) -> Tuple[float, float, float, float]:
        def _f(v: Any) -> float:
            n = pd.to_numeric(v, errors="coerce")
            return float(n) if pd.notna(n) else -1e9

        return (
            _f(r.get("Filtered_Expectancy_Pct")),
            _f(r.get("Expectancy_Lift_Pct")),
            _f(r.get("Retention_Pct")),
            _f(r.get("Filtered_Trades")),
        )

    return max(candidates, key=_sort_key)


def classify_incremental_contribution(
    incr_exp: float,
    incr_ret: float,
    new_trades: float,
) -> str:
    """Classify whether the 3rd factor is Helpful, Neutral, Harmful, or Unknown."""
    if any(pd.isna(v) for v in (incr_exp, incr_ret, new_trades)):
        return "Unknown"
    if incr_exp >= 0.05 and incr_ret > -10 and new_trades >= 100:
        return "Helpful"
    if incr_exp < 0 or incr_ret < -15 or new_trades < 30:
        return "Harmful"
    return "Neutral"


def annotate_incremental_3factor(sid_combo_rows: List[Dict[str, Any]]) -> None:
    """
    Mutate 3-factor combo dicts in-place to add incremental contribution columns.
    2-factor rows receive np.nan / "" placeholders for all 3F-specific columns.
    """
    _nan = np.nan
    _str_cols = {"Base_Combo_Rule", "Base_Combo_Factors", "Incremental_Factor", "Incremental_Contribution_Class"}
    _all_cols = [
        "Base_Combo_Rule", "Base_Combo_Factors", "Base_Combo_Order",
        "Base_Filtered_Trades", "Base_Retention_Pct",
        "Base_Filtered_Expectancy_Pct", "Base_Expectancy_Lift_Pct",
        "Incremental_Factor", "Incremental_Trades_Change",
        "Incremental_Retention_Change", "Incremental_Expectancy_Change",
        "Incremental_Lift_Change", "Incremental_Contribution_Class",
    ]
    # Default all rows so the CSV has consistent columns.
    for r in sid_combo_rows:
        for col in _all_cols:
            if col not in r:
                r[col] = "" if col in _str_cols else _nan

    def _n(v: Any) -> float:
        val = pd.to_numeric(v, errors="coerce")
        return float(val) if pd.notna(val) else _nan

    for r in sid_combo_rows:
        if r.get("Num_Factors") != 3:
            continue
        factors_3f = str(r["Factors"]).split("|")
        base = find_best_base_combo(sid_combo_rows, factors_3f)

        if base is None:
            r["Incremental_Contribution_Class"] = "Unknown"
            continue

        base_factors = set(str(base["Factors"]).split("|"))
        incr_factor_list = [f for f in factors_3f if f not in base_factors]
        incr_factor = incr_factor_list[0] if incr_factor_list else ""

        base_trades = _n(base.get("Filtered_Trades"))
        base_ret    = _n(base.get("Retention_Pct"))
        base_exp    = _n(base.get("Filtered_Expectancy_Pct"))
        base_lift   = _n(base.get("Expectancy_Lift_Pct"))
        new_trades  = _n(r.get("Filtered_Trades"))
        new_ret     = _n(r.get("Retention_Pct"))
        new_exp     = _n(r.get("Filtered_Expectancy_Pct"))
        new_lift    = _n(r.get("Expectancy_Lift_Pct"))

        r["Base_Combo_Rule"]               = str(base.get("Rule", ""))
        r["Base_Combo_Factors"]            = str(base.get("Factors", ""))
        r["Base_Combo_Order"]              = int(base.get("Combo_Order", 2))
        r["Base_Filtered_Trades"]          = base_trades
        r["Base_Retention_Pct"]            = base_ret
        r["Base_Filtered_Expectancy_Pct"]  = base_exp
        r["Base_Expectancy_Lift_Pct"]      = base_lift
        r["Incremental_Factor"]            = incr_factor
        r["Incremental_Trades_Change"]     = (new_trades - base_trades)    if pd.notna(new_trades) and pd.notna(base_trades) else _nan
        r["Incremental_Retention_Change"]  = (new_ret    - base_ret)       if pd.notna(new_ret)    and pd.notna(base_ret)    else _nan
        r["Incremental_Expectancy_Change"] = (new_exp    - base_exp)       if pd.notna(new_exp)    and pd.notna(base_exp)    else _nan
        r["Incremental_Lift_Change"]       = (new_lift   - base_lift)      if pd.notna(new_lift)   and pd.notna(base_lift)   else _nan
        r["Incremental_Contribution_Class"] = classify_incremental_contribution(
            r["Incremental_Expectancy_Change"],
            r["Incremental_Retention_Change"],
            new_trades,
        )


def annotate_2factor_vs_single(
    sid_combo_rows: List[Dict[str, Any]],
    sid_singles: pd.DataFrame,
) -> None:
    """
    Mutate 2-factor combo dicts in-place to add single-factor comparison columns.
    3-factor rows receive np.nan / "" placeholders.

    Uses all Stage 2 single-rule rows for the SID (not just promoted seeds) so
    that the best available single-factor evidence is used as the baseline.
    """
    _nan = np.nan
    _str_cols = {"Best_Single_Factor_Rule"}
    _all_cols = [
        "Best_Single_Factor_Rule", "Best_Single_Filtered_Expectancy_Pct",
        "Best_Single_Expectancy_Lift_Pct", "TwoFactor_Incremental_Expectancy_Change",
        "TwoFactor_Incremental_Lift_Change",
    ]
    for r in sid_combo_rows:
        for col in _all_cols:
            if col not in r:
                r[col] = "" if col in _str_cols else _nan

    if sid_singles.empty:
        return

    def _n(v: Any) -> float:
        val = pd.to_numeric(v, errors="coerce")
        return float(val) if pd.notna(val) else _nan

    for r in sid_combo_rows:
        if r.get("Num_Factors") != 2:
            continue
        factors_2f = str(r["Factors"]).split("|")
        best_rule = ""
        best_exp  = -1e9
        best_lift = _nan

        for factor in factors_2f:
            factor_rows = sid_singles[sid_singles["Factor"].astype(str) == factor]
            if factor_rows.empty:
                continue
            factor_rows_sorted = factor_rows.sort_values(
                ["Filtered_Expectancy_Pct", "Expectancy_Lift_Pct"],
                ascending=[False, False],
                na_position="last",
            )
            top = factor_rows_sorted.iloc[0]
            top_exp = _n(top.get("Filtered_Expectancy_Pct"))
            if pd.isna(top_exp):
                top_exp = -1e9
            if top_exp > best_exp:
                best_exp  = top_exp
                best_rule = str(top.get("Rule", ""))
                best_lift = _n(top.get("Expectancy_Lift_Pct"))

        if not best_rule:
            continue

        new_exp  = _n(r.get("Filtered_Expectancy_Pct"))
        new_lift = _n(r.get("Expectancy_Lift_Pct"))
        s_exp    = best_exp if best_exp > -1e9 else _nan

        r["Best_Single_Factor_Rule"]                 = best_rule
        r["Best_Single_Filtered_Expectancy_Pct"]     = s_exp
        r["Best_Single_Expectancy_Lift_Pct"]         = best_lift
        r["TwoFactor_Incremental_Expectancy_Change"] = (new_exp  - s_exp)    if pd.notna(new_exp)  and pd.notna(s_exp)    else _nan
        r["TwoFactor_Incremental_Lift_Change"]       = (new_lift - best_lift) if pd.notna(new_lift) and pd.notna(best_lift) else _nan


def load_stage2_inputs(input_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    baseline_path = input_dir / "baseline_by_sid.csv"
    singles_path = input_dir / "winner_single_rule_validation_by_sid.csv"

    if not baseline_path.exists():
        raise FileNotFoundError(f"Missing Stage 2 file: {baseline_path}")
    if not singles_path.exists():
        raise FileNotFoundError(f"Missing Stage 2 file: {singles_path}")

    baseline_df = pd.read_csv(baseline_path)
    singles_df = pd.read_csv(singles_path)

    needed_baseline = {"SID", "Expectancy_Pct", "Trades"}
    needed_singles = {"SID", "Factor", "Rule", "Filtered_Trades", "Retention_Pct", "Expectancy_Lift_Pct"}
    if not needed_baseline.issubset(set(baseline_df.columns)):
        raise ValueError(f"baseline_by_sid.csv missing required columns: {sorted(needed_baseline)}")
    if not needed_singles.issubset(set(singles_df.columns)):
        raise ValueError(f"winner_single_rule_validation_by_sid.csv missing required columns: {sorted(needed_singles)}")

    return baseline_df, singles_df


def load_campaign_dataset(path: str, include_secondary: bool) -> Tuple[pd.DataFrame, str, str, Dict[str, str]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Campaign input file not found: {path}")

    df = pd.read_csv(path)

    sid_col = find_column(df, SID_ALIASES)
    ret_col = find_column(df, RETURN_ALIASES)
    if sid_col is None:
        raise ValueError(f"Missing SetupID column. Tried aliases: {SID_ALIASES}")
    if ret_col is None:
        raise ValueError(f"Missing return column. Tried aliases: {RETURN_ALIASES}")

    desired = list(CORE_FACTORS)
    if include_secondary:
        desired += SECONDARY_FACTORS

    lookup = {str(c).strip().lower(): c for c in df.columns}
    factor_map: Dict[str, str] = {}
    for f in desired:
        col = lookup.get(f.lower())
        if col is not None:
            factor_map[f] = col

    if len([f for f in CORE_FACTORS if f in factor_map]) == 0:
        raise ValueError(f"None of core factors present in campaign input. Expected one of: {CORE_FACTORS}")

    df[sid_col] = pd.to_numeric(df[sid_col], errors="coerce")
    df[ret_col] = pd.to_numeric(df[ret_col], errors="coerce")
    for factor, col in factor_map.items():
        if factor in NUMERIC_FACTORS:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        elif factor in BINARY_FACTORS:
            df[col] = to_sqz_binary(df[col])

    needed = [sid_col, ret_col] + list(factor_map.values())
    df = df.dropna(subset=[sid_col, ret_col]).copy()
    df = df.dropna(subset=[c for c in needed if c in df.columns]).copy()
    df[sid_col] = df[sid_col].astype(int)

    return df, sid_col, ret_col, factor_map


def run_pipeline(args: argparse.Namespace) -> None:
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    baseline_df, singles_df = load_stage2_inputs(input_dir)
    campaigns_df, sid_col, ret_col, factor_map = load_campaign_dataset(args.campaign_input, args.include_secondary_factors)

    requested_sids = parse_setup_ids(args.setup_ids)
    available_sids = sorted(campaigns_df[sid_col].dropna().astype(int).unique().tolist())
    chosen_sids = [sid for sid in requested_sids if sid in set(available_sids)] if requested_sids else available_sids

    if not chosen_sids:
        raise ValueError("No SetupIDs selected after filtering.")

    combo_rows: List[Dict[str, Any]] = []
    best_rows: List[Dict[str, Any]] = []
    seed_stats_rows: List[Dict[str, Any]] = []

    print("\n" + "=" * 70)
    print("STAGE 3 - FACTOR RULE REFINEMENT")
    print("=" * 70)
    print(f"Stage 2 input dir: {input_dir}")
    print(f"Campaign input: {args.campaign_input}")
    print(f"Selected SetupIDs: {chosen_sids}")

    for sid in chosen_sids:
        df_sid = campaigns_df[campaigns_df[sid_col] == sid].copy()
        if df_sid.empty:
            continue

        baseline_hit = baseline_df[baseline_df["SID"] == sid]
        if baseline_hit.empty:
            print(f"SID {sid}: skipped, missing baseline row from Stage 2")
            continue
        baseline_expectancy = float(baseline_hit.iloc[0]["Expectancy_Pct"]) / 100.0

        sid_singles = singles_df[singles_df["SID"] == sid].copy()
        if sid_singles.empty:
            print(f"SID {sid}: skipped, no Stage 2 single-rule rows")
            continue

        allowed_factors = set(CORE_FACTORS + (SECONDARY_FACTORS if args.include_secondary_factors else []))
        sid_singles = sid_singles[sid_singles["Factor"].astype(str).isin(allowed_factors)].copy()

        sort_cols = []
        ascending = []

        for col in ["Winner_Enrichment", "Expectancy_Lift_Pct", "Retention_Pct", "Filtered_Trades"]:
            if col in sid_singles.columns:
                sort_cols.append(col)
                ascending.append(False)

        sid_singles = sid_singles.sort_values(sort_cols, ascending=ascending, na_position="last")

        seeds = choose_seed_pool(sid_singles, args.allow_tier_2_seeds, args)

        if seeds.empty:
            print(f"SID {sid}: skipped, no Stage 2 eligible seeds after tier gating")
            continue

        # Keep one strongest seed per factor first, then apply top-k.
        seeds = seeds.drop_duplicates(subset=["Factor"], keep="first").copy()
        seeds = seeds.head(args.top_k_single)

        parsed_seed: List[Tuple[str, Condition, str, bool, int]] = []
        for _, row in seeds.iterrows():
            factor = str(row["Factor"])
            if factor not in factor_map:
                continue
            conds = parse_rule_text(str(row["Rule"]))
            if len(conds) != 1:
                continue
            cond = conds[0]
            seed_tier = int(pd.to_numeric(row.get("Seed_Tier", 0), errors="coerce") or 0)
            used_tier2 = seed_tier >= 2 or ("Downstream_Eligible" in row and not bool(row.get("Downstream_Eligible", False)))
            parsed_seed.append((factor, cond, str(row["Rule"]), bool(used_tier2), seed_tier))

        tier2_seed_count = int(sum(1 for x in parsed_seed if x[3]))
        seed_stats_rows.append(
            {
                "SID": sid,
                "Valid_Seed_Count": int(len(parsed_seed)),
                "Tier2_Seed_Count": tier2_seed_count,
            }
        )

        if len(parsed_seed) < 2:
            print(f"SID {sid}: skipped, fewer than 2 valid seed factors after broad Stage 3 seed selection")
            continue

        widths = [2, 3] if args.allow_3factor else [2]
        # Part F (internal): allow slightly lower retention floor for combos only.
        # Minimum of 12% regardless of CLI flag value. Not exposed as a CLI argument.
        _combo_min_retention = max(12.0, args.min_retention - 3.0)
        # Part D: 3-factor combos use only top-ranked seeds (quality > quantity).
        _top_seeds_for_3factor = parsed_seed[: max(3, len(parsed_seed) // 2)]
        sid_combo_rows: List[Dict[str, Any]] = []
        tested = 0
        rejected_examples: Dict[str, str] = {}

        for k in widths:
            seed_pool = _top_seeds_for_3factor if k == 3 else parsed_seed
            if len(seed_pool) < k:
                continue
            for combo in itertools.combinations(seed_pool, k):
                if tested >= MAX_COMBO_TESTS_PER_SID:
                    break
                factors = [x[0] for x in combo]
                if len(set(factors)) != len(factors):
                    continue

                core_factor_count, combo_valid_structure, invalid_reason = combo_structure_metadata(factors, k)
                if not combo_valid_structure:
                    if invalid_reason not in rejected_examples:
                        rejected_examples[invalid_reason] = " AND ".join(x[1].to_rule_text() for x in combo)
                    continue

                conditions = [(x[1], factor_map[x[0]]) for x in combo]
                evaluated = evaluate_combo(df_sid, ret_col, baseline_expectancy, conditions)
                exp_pct = evaluated["expectancy"] * 100.0 if pd.notna(evaluated["expectancy"]) else np.nan
                baseline_pct = baseline_expectancy * 100.0
                combo_uses_tier2 = any(x[3] for x in combo)
                row = {
                    "SID": sid,
                    "Factors": "|".join(factors),
                    "Rule": evaluated["rule"],
                    "Num_Factors": evaluated["num_factors"],
                    "Filtered_Trades": evaluated["trades"],
                    "Retention_Pct": evaluated["retention_pct"],
                    "Filtered_Expectancy_Pct": exp_pct,
                    "Baseline_Expectancy_Pct": baseline_pct,
                    "Expectancy_Lift_Pct": evaluated["lift_pct"],
                    "Filtered_Win_Rate": evaluated["win_rate"],
                    "Filtered_Median_Return_Pct": evaluated["median_return"] * 100.0 if pd.notna(evaluated["median_return"]) else np.nan,
                    "Combo_Order": k,
                    "Used_Tier2_Seed": combo_uses_tier2,
                    "Core_Factor_Count": core_factor_count,
                    "Combo_Valid_Structure": combo_valid_structure,
                    # Part A: raise promotion standards — require meaningful lift and absolute expectancy.
                    "Promoted": (
                        evaluated["trades"] >= args.min_trades
                        and pd.notna(evaluated["retention_pct"]) and evaluated["retention_pct"] >= _combo_min_retention
                        and pd.notna(evaluated["lift_pct"]) and evaluated["lift_pct"] >= 0.05
                        and pd.notna(evaluated["expectancy"]) and exp_pct >= baseline_pct + 0.05
                    ),
                }
                sid_combo_rows.append(row)
                combo_rows.append(row)
                tested += 1

            if tested >= MAX_COMBO_TESTS_PER_SID:
                break

        if not sid_combo_rows:
            print(f"SID {sid}: no valid combinations produced")
            continue

        if rejected_examples:
            first_reason = sorted(rejected_examples.keys())[0]
            print(f"SID {sid}: rejected invalid 3-factor structure ({first_reason}) example: {rejected_examples[first_reason]}")

        # Annotate incremental contribution metrics in-place.
        # annotate_incremental_3factor mutates combo dicts, so combo_rows (which holds
        # the same dict objects) is updated automatically.
        annotate_incremental_3factor(sid_combo_rows)
        annotate_2factor_vs_single(sid_combo_rows, sid_singles)

        sid_combo_df = pd.DataFrame(sid_combo_rows)
        sid_combo_df = sid_combo_df.sort_values(
            ["Promoted", "Filtered_Expectancy_Pct", "Expectancy_Lift_Pct", "Retention_Pct", "Filtered_Trades"],
            ascending=[False, False, False, False, False],
            na_position="last",
        )

        promoted = sid_combo_df[sid_combo_df["Promoted"]].head(args.top_k_combo)
        if promoted.empty:
            promoted = sid_combo_df.head(min(args.top_k_combo, len(sid_combo_df)))

        best_rows.extend(promoted.to_dict(orient="records"))

    combo_df = pd.DataFrame(combo_rows)
    best_df = pd.DataFrame(best_rows)

    combo_cols = [
        "SID", "Factors", "Rule", "Num_Factors", "Filtered_Trades", "Retention_Pct",
        "Filtered_Expectancy_Pct", "Baseline_Expectancy_Pct", "Expectancy_Lift_Pct",
        "Filtered_Win_Rate", "Filtered_Median_Return_Pct",
        "Combo_Order", "Used_Tier2_Seed", "Core_Factor_Count", "Combo_Valid_Structure",
        "Promoted",
        # 3-factor incremental contribution columns
        "Base_Combo_Rule", "Base_Combo_Factors", "Base_Combo_Order",
        "Base_Filtered_Trades", "Base_Retention_Pct",
        "Base_Filtered_Expectancy_Pct", "Base_Expectancy_Lift_Pct",
        "Incremental_Factor", "Incremental_Trades_Change",
        "Incremental_Retention_Change", "Incremental_Expectancy_Change",
        "Incremental_Lift_Change", "Incremental_Contribution_Class",
        # 2-factor vs single-factor comparison columns
        "Best_Single_Factor_Rule", "Best_Single_Filtered_Expectancy_Pct",
        "Best_Single_Expectancy_Lift_Pct", "TwoFactor_Incremental_Expectancy_Change",
        "TwoFactor_Incremental_Lift_Change",
    ]
    best_cols = combo_cols

    pd.DataFrame(combo_df, columns=combo_cols).to_csv(output_dir / "factor_combo_validation_by_sid.csv", index=False)
    pd.DataFrame(best_df, columns=best_cols).to_csv(output_dir / "best_factor_combos_by_sid.csv", index=False)

    # Compatibility outputs retained for downstream stages that still expect these filenames.
    compat = best_df.copy()
    if compat.empty:
        compat = pd.DataFrame(columns=[
            "subset", "sid", "rule", "num_factors", "validation_trades", "validation_expectancy",
            "expectancy_lift", "expectancy_lift_pct", "trade_retention", "trade_retention_pct", "deployable", "score",
        ])
    else:
        compat = pd.DataFrame(
            {
                "subset": [f"sid{int(v)}" for v in compat["SID"]],
                "sid": compat["SID"],
                "rule": compat["Rule"],
                "num_factors": compat["Num_Factors"],
                "validation_trades": compat["Filtered_Trades"],
                "validation_expectancy": pd.to_numeric(compat["Filtered_Expectancy_Pct"], errors="coerce") / 100.0,
                "expectancy_lift": pd.to_numeric(compat["Expectancy_Lift_Pct"], errors="coerce") / 100.0,
                "expectancy_lift_pct": compat["Expectancy_Lift_Pct"],
                "trade_retention": pd.to_numeric(compat["Retention_Pct"], errors="coerce") / 100.0,
                "trade_retention_pct": compat["Retention_Pct"],
                "deployable": compat["Promoted"],
                "score": compat["Expectancy_Lift_Pct"],
            }
        )

    compat.to_csv(output_dir / "deployable_rules.csv", index=False)
    compat.to_csv(output_dir / "expectancy_lift_report.csv", index=False)

    summary_lines = [
        "STAGE 3 - FACTOR RULE REFINEMENT SUMMARY",
        "=" * 70,
        f"Stage 2 input dir: {input_dir}",
        f"Campaign input: {args.campaign_input}",
        f"Selected SetupIDs: {chosen_sids}",
        f"Allow Tier 2 seeds: {args.allow_tier_2_seeds}",
        f"Allow 3-factor: {args.allow_3factor}",
        "",
    ]

    if seed_stats_rows:
        summary_lines.append("Seed availability by SID:")
        for row in sorted(seed_stats_rows, key=lambda x: x["SID"]):
            summary_lines.append(
                f"- SID {row['SID']}: valid_seeds={row['Valid_Seed_Count']}, tier2_seeds={row['Tier2_Seed_Count']}"
            )
        summary_lines.append("")

    if best_df.empty:
        summary_lines.append("No combinations passed Stage 3 practical gates.")
    else:
        for sid in sorted(pd.to_numeric(best_df["SID"], errors="coerce").dropna().astype(int).unique().tolist()):
            sid_best = best_df[best_df["SID"] == sid].copy()
            summary_lines.append(f"SID {sid}")
            summary_lines.append("-" * 70)

            best_2 = sid_best[sid_best["Num_Factors"] == 2].head(3)
            best_3 = sid_best[sid_best["Num_Factors"] == 3].head(3)

            summary_lines.append("Best 2-factor combos:")
            if best_2.empty:
                summary_lines.append("- none")
            else:
                for _, row in best_2.iterrows():
                    summary_lines.append(
                        f"- {row['Rule']} | trades={int(row['Filtered_Trades'])} | retention={row['Retention_Pct']:.1f}% | "
                        f"exp={row['Filtered_Expectancy_Pct']:.2f}% | lift={row['Expectancy_Lift_Pct']:.2f}%"
                    )

            if args.allow_3factor:
                summary_lines.append("Best 3-factor combos:")
                if best_3.empty:
                    summary_lines.append("- none")
                else:
                    for _, row in best_3.iterrows():
                        summary_lines.append(
                            f"- {row['Rule']} | trades={int(row['Filtered_Trades'])} | retention={row['Retention_Pct']:.1f}% | "
                            f"exp={row['Filtered_Expectancy_Pct']:.2f}% | lift={row['Expectancy_Lift_Pct']:.2f}%"
                        )

            restrictive = sid_best[sid_best["Filtered_Trades"] < args.min_trades].head(2)
            summary_lines.append("Promising but restrictive:")
            if restrictive.empty:
                summary_lines.append("- none")
            else:
                for _, row in restrictive.iterrows():
                    summary_lines.append(
                        f"- {row['Rule']} | trades={int(row['Filtered_Trades'])} | retention={row['Retention_Pct']:.1f}% | "
                        f"lift={row['Expectancy_Lift_Pct']:.2f}%"
                    )
            summary_lines.append("")

    # ------------------------------------------------------------------
    # Incremental Contribution Highlights
    # ------------------------------------------------------------------
    summary_lines += [
        "Incremental Contribution Highlights (3-factor vs 2-factor base)",
        "=" * 70,
    ]

    if not args.allow_3factor:
        summary_lines.append("(3-factor mode not enabled — pass --allow-3factor to activate.)")
    elif combo_df.empty or "Incremental_Contribution_Class" not in combo_df.columns:
        summary_lines.append("No combo data available for incremental analysis.")
    else:
        three_f = combo_df[pd.to_numeric(combo_df["Num_Factors"], errors="coerce") == 3]
        if three_f.empty:
            summary_lines += [
                "No valid 3-factor combos exist in this run.",
                "(Each SID needs 3+ qualified seeds for a 3-factor combo to form.)",
                "All current SIDs have fewer than 3 qualified seeds — this is expected.",
            ]
        else:
            class_counts = three_f["Incremental_Contribution_Class"].value_counts()
            summary_lines.append(f"3-factor combos evaluated: {len(three_f)}")
            for cls in ("Helpful", "Neutral", "Harmful", "Unknown"):
                n = int(class_counts.get(cls, 0))
                if n:
                    summary_lines.append(f"  {cls}: {n}")

            helpful_rows = three_f[three_f["Incremental_Contribution_Class"] == "Helpful"].head(3)
            if not helpful_rows.empty:
                summary_lines.append("Examples — Helpful additions:")
                for _, r in helpful_rows.iterrows():
                    exp_chg  = r.get("Incremental_Expectancy_Change", np.nan)
                    ret_chg  = r.get("Incremental_Retention_Change",  np.nan)
                    factor   = r.get("Incremental_Factor", "?")
                    exp_chg_str  = f"{exp_chg:.2f}%" if pd.notna(exp_chg) else "n/a"
                    ret_chg_str  = f"{ret_chg:.1f}%" if pd.notna(ret_chg) else "n/a"
                    summary_lines.append(
                        f"  + [{factor}] {r['Rule']} | exp_change={exp_chg_str} | ret_change={ret_chg_str}"
                    )

            harmful_rows = three_f[three_f["Incremental_Contribution_Class"] == "Harmful"].head(3)
            if not harmful_rows.empty:
                summary_lines.append("Examples — Harmful/Redundant additions:")
                for _, r in harmful_rows.iterrows():
                    exp_chg  = r.get("Incremental_Expectancy_Change", np.nan)
                    ret_chg  = r.get("Incremental_Retention_Change",  np.nan)
                    factor   = r.get("Incremental_Factor", "?")
                    exp_chg_str  = f"{exp_chg:.2f}%" if pd.notna(exp_chg) else "n/a"
                    ret_chg_str  = f"{ret_chg:.1f}%" if pd.notna(ret_chg) else "n/a"
                    summary_lines.append(
                        f"  - [{factor}] {r['Rule']} | exp_change={exp_chg_str} | ret_change={ret_chg_str}"
                    )

    summary_lines.append("")

    # 2-factor vs single-factor highlights (brief)
    if not combo_df.empty and "TwoFactor_Incremental_Expectancy_Change" in combo_df.columns:
        two_f = combo_df[pd.to_numeric(combo_df["Num_Factors"], errors="coerce") == 2]
        two_f = two_f[two_f["Best_Single_Factor_Rule"].astype(str).str.strip() != ""]
        if not two_f.empty:
            summary_lines += [
                "2-factor vs best single-factor seed (top gains):",
            ]
            two_f_sorted = two_f.sort_values(
                "TwoFactor_Incremental_Expectancy_Change", ascending=False, na_position="last"
            ).head(3)
            for _, r in two_f_sorted.iterrows():
                chg = r.get("TwoFactor_Incremental_Expectancy_Change", np.nan)
                chg_str = f"{chg:+.2f}%" if pd.notna(chg) else "n/a"
                summary_lines.append(
                    f"  SID {int(r['SID'])} {r['Rule']} vs [{r['Best_Single_Factor_Rule']}] => {chg_str}"
                )
            summary_lines.append("")

    (output_dir / "stage_3_refinement_summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")

    print("\n" + "=" * 70)
    print("EXPORT COMPLETE")
    print("=" * 70)
    print(f"Outputs written to: {output_dir}")


def main() -> None:
    args = parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
