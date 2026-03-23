#!/usr/bin/env python3
"""
STAGE 2 - WINNER PROFILE DISCOVERY
==================================

Purpose:
- Identify where winners cluster for each SetupID
- Generate simple threshold/range candidate rules
- Validate those candidates on all trades for the same SetupID

This stage does NOT own:
- final combo mining
- stability testing
- deployment formatting
- Pine integration

Core outputs owned by Stage 2:
- baseline_by_sid.csv
- winner_distribution_by_sid.csv
- winner_single_rule_validation_by_sid.csv

Compatibility outputs (legacy schema, opt-in via --include-compatibility-outputs):
- factor_relevance_by_sid.csv
- candidate_rules_by_sid.csv
- deployable_rules_by_sid.csv
- preferred_ranges_by_sid.csv (header-only)
- avoid_zones_by_sid.csv (header-only)
- debug_rule_pipeline_by_sid.csv
- winner_combo_rule_validation_by_sid.csv (header-only)

python stage_2_winner_profile_engine.py \
  --input campaigns_clean.csv \
  --output-dir Stage_2_Report \
  --discovery-threshold 5.0

"""

from __future__ import annotations

import argparse
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

# Soft structural preference used only for ranking/tie-breaking of strong seeds.
STRUCTURAL_PRIORITY_FACTORS = {"CMP", "TR", "VQS", "RS"}
SECONDARY_PRIORITY_FACTORS = {"ORC", "RMV", "SQZ"}

# Two-tier seed system for single-factor rules
# Tier 1: Strict standalone seeds (Promoted, Downstream_Eligible, Combo_Eligible)
# Tier 2: Controlled expansion for Stage 3 combo testing (Combo_Eligible only)
TIER_1_LIFT_MIN = 0.15
TIER_1_ENRICHMENT_MIN = 1.05
TIER_2_LIFT_MIN = 0.08
TIER_2_ENRICHMENT_MIN = 1.02
TIER_2_RETENTION_MIN = 20.0  # Absolute floor for Tier 2
TIER_2_TRADES_MIN = 50      # Absolute floor for Tier 2

# Winner tier definitions for return_pct-based analysis
# Small Winner: positive but modest return
# Medium Winner: meaningful return
# Large Winner: strong return / runner-like
SMALL_WINNER_MAX = 1.0      # return_pct < 1.0
MEDIUM_WINNER_MIN = 1.0     # return_pct >= 1.0
MEDIUM_WINNER_MAX = 3.0     # return_pct < 3.0
LARGE_WINNER_MIN = 3.0      # return_pct >= 3.0

SID_ALIASES = ["sid", "setupid", "setup_id"]
RETURN_ALIASES = ["return_pct", "return", "ret_pct"]
DATE_ALIASES = ["entry_dt", "entry_date", "date"]


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
    p = argparse.ArgumentParser(
        description="Stage 2 - Winner Profile Discovery",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--input", "-i", default="campaigns_clean.csv")
    p.add_argument(
        "--output", "-o", "--output-dir",
        dest="output_dir",
        default="Stage_2_Report",
        help="Output directory (default: Stage_2_Report)",
    )
    p.add_argument("--setup-ids", default="")

    p.add_argument("--winner-mode", choices=["threshold", "percentile", "expectancy"], default="threshold")
    p.add_argument("--winner-threshold", type=float, default=1.2)
    p.add_argument("--discovery-threshold", type=float, default=3.0, help="Discovery threshold in percent units (default: 3.0)")
    p.add_argument("--winner-percentile", type=float, default=70.0)
    p.add_argument(
        "--winner-expectancy-cutoff",
        type=float,
        default=0.0,
        help="For winner-mode expectancy: winner if return_pct >= baseline_expectancy + cutoff",
    )

    p.add_argument("--min-trades", type=int, default=30, help="Minimum filtered trades for a validated candidate rule")
    p.add_argument("--min-retention", type=float, default=15.0, help="Minimum retention percentage for downstream-eligible single rules")
    p.add_argument("--min-trades-per-sid", type=int, default=30, help="Minimum total trades required to analyze a SetupID")
    p.add_argument("--min-winner-trades", type=int, default=15)
    p.add_argument("--top-k-single", type=int, default=10)
    p.add_argument("--include-secondary-factors", action="store_true")
    p.add_argument(
        "--include-compatibility-outputs",
        action="store_true",
        help="Also write legacy compatibility CSV outputs",
    )
    return p.parse_args()


def find_column(df: pd.DataFrame, aliases: List[str]) -> Optional[str]:
    lookup = {str(c).strip().lower(): c for c in df.columns}
    for a in aliases:
        if a.lower() in lookup:
            return lookup[a.lower()]
    return None


def factor_columns(df: pd.DataFrame, include_secondary: bool) -> Dict[str, str]:
    desired = list(CORE_FACTORS)
    if include_secondary:
        desired += SECONDARY_FACTORS

    lookup = {str(c).strip().lower(): c for c in df.columns}
    out: Dict[str, str] = {}
    for f in desired:
        col = lookup.get(f.lower())
        if col is not None:
            out[f] = col
    return out


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
            val = float(txt)
            return 1 if val >= 0.5 else 0
        except Exception:
            return None

    return s.apply(_map)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def parse_setup_ids(raw: str) -> Optional[List[int]]:
    if not raw.strip():
        return None
    out: List[int] = []
    for token in raw.split(","):
        tok = token.strip()
        if not tok:
            continue
        out.append(int(tok))
    return sorted(set(out))


def basic_metrics(df: pd.DataFrame, ret_col: str) -> Dict[str, Any]:
    r = pd.to_numeric(df[ret_col], errors="coerce").dropna()
    if r.empty:
        return {
            "trades": 0,
            "wins": 0,
            "win_rate": np.nan,
            "expectancy": np.nan,
            "median_return": np.nan,
            "payoff_ratio": np.nan,
        }

    wins = r[r > 0]
    losses = r[r < 0]
    avg_win = float(wins.mean()) if len(wins) else np.nan
    avg_loss_abs = abs(float(losses.mean())) if len(losses) else np.nan
    payoff = (avg_win / avg_loss_abs) if pd.notna(avg_win) and pd.notna(avg_loss_abs) and avg_loss_abs > 0 else np.nan

    return {
        "trades": int(len(r)),
        "wins": int((r > 0).sum()),
        "win_rate": float((r > 0).mean()),
        "expectancy": float(r.mean()),
        "median_return": float(r.median()),
        "payoff_ratio": payoff,
    }


def winner_mask(df_sid: pd.DataFrame, ret_col: str, mode: str, threshold: float, percentile: float, expectancy_cutoff: float) -> pd.Series:
    r = pd.to_numeric(df_sid[ret_col], errors="coerce")

    # Auto-scale user thresholds across percent-style (1.2) and decimal-style (0.012) datasets.
    abs_med = float(np.nanmedian(np.abs(r.to_numpy(dtype=float)))) if len(r) else np.nan

    def adapt(v: float) -> float:
        if pd.isna(abs_med):
            return v
        if abs_med <= 2.0 and v > 1.0:
            return v / 100.0
        if abs_med > 20.0 and 0.0 < v <= 1.0:
            return v * 100.0
        return v

    if mode == "threshold":
        return r >= adapt(threshold)
    if mode == "percentile":
        q = np.nanpercentile(r.to_numpy(dtype=float), percentile)
        return r >= q

    baseline = float(np.nanmean(r.to_numpy(dtype=float)))
    return r >= (baseline + adapt(expectancy_cutoff))


def adapt_return_thresholds(series: pd.Series, *thresholds: float) -> Tuple[float, ...]:
    """
    Adapt percentage-style thresholds such as 1.0 and 3.0 to decimal-style data
    such as 0.01 and 0.03 when needed.
    """
    r = pd.to_numeric(series, errors="coerce")
    abs_r = np.abs(r.dropna().to_numpy(dtype=float))
    abs_med = float(np.nanmedian(abs_r)) if len(abs_r) else np.nan
    abs_p95 = float(np.nanpercentile(abs_r, 95)) if len(abs_r) else np.nan
    share_gt_1 = float(np.mean(abs_r > 1.0)) if len(abs_r) else np.nan

    is_decimal_style = (
        pd.notna(abs_med)
        and pd.notna(abs_p95)
        and pd.notna(share_gt_1)
        and abs_med <= 0.5
        and abs_p95 <= 1.0
        and share_gt_1 <= 0.01
    )

    def adapt(v: float) -> float:
        if not len(abs_r):
            return v
        if is_decimal_style and v >= 1.0:
            return v / 100.0
        if not is_decimal_style and 0.0 < v < 1.0:
            return v * 100.0
        return v

    return tuple(adapt(v) for v in thresholds)


def is_baseline_winner(return_pct: float) -> bool:
    """
    A baseline winner is any trade with positive return.
    """
    return pd.notna(return_pct) and return_pct > 0.0


def is_discovery_winner(return_pct: float, adapted_threshold: float) -> bool:
    """
    A discovery winner is any trade meeting or exceeding the discovery threshold.
    The threshold must be already adapted to match the dataset's scale (% or decimal).
    """
    return pd.notna(return_pct) and return_pct >= adapted_threshold


def is_large_winner(return_pct: float, adapted_large_threshold: float) -> bool:
    """
    A large winner is any trade meeting or exceeding the large-winner threshold (default 3%).
    The threshold must be already adapted to match the dataset's scale (% or decimal).
    """
    return pd.notna(return_pct) and return_pct >= adapted_large_threshold


def calculate_discovery_diagnostics(
    df_sid: pd.DataFrame,
    ret_col: str,
    discovery_threshold: float,
    large_threshold: float = 3.0,
) -> Dict[str, Any]:
    """
    Validate discovery threshold behavior and report retention statistics.
    
    Returns:
        dict with keys:
            - total_baseline_winners: count of trades with return_pct > 0
            - total_discovery_winners: count of trades with return_pct >= discovery_threshold
            - total_large_winners: count of trades with return_pct >= large_threshold
            - retention_pct: discovery / baseline (%)
            - large_winner_retention_pct: large winners caught by discovery (%)
    """
    r = pd.to_numeric(df_sid[ret_col], errors="coerce")
    abs_r = np.abs(r.dropna().to_numpy(dtype=float))
    abs_med = float(np.nanmedian(abs_r)) if len(abs_r) else np.nan
    abs_p95 = float(np.nanpercentile(abs_r, 95)) if len(abs_r) else np.nan
    share_gt_1 = float(np.mean(abs_r > 1.0)) if len(abs_r) else np.nan

    is_decimal_style = (
        pd.notna(abs_med)
        and pd.notna(abs_p95)
        and pd.notna(share_gt_1)
        and abs_med <= 0.5
        and abs_p95 <= 1.0
        and share_gt_1 <= 0.01
    )

    def adapt(v: float) -> float:
        if not len(abs_r):
            return v
        if is_decimal_style and v >= 1.0:
            return v / 100.0
        if not is_decimal_style and 0.0 < v < 1.0:
            return v * 100.0
        return v

    adapted_discovery = adapt(discovery_threshold)
    adapted_large = adapt(large_threshold)

    baseline_mask = r.apply(lambda x: is_baseline_winner(x))
    discovery_mask = r.apply(lambda x: is_discovery_winner(x, adapted_discovery))
    large_mask = r.apply(lambda x: is_large_winner(x, adapted_large))

    total_baseline = int(baseline_mask.sum())
    total_discovery = int(discovery_mask.sum())
    total_large = int(large_mask.sum())

    # Large winners that are also caught by discovery
    large_retained = int((large_mask & discovery_mask).sum())

    retention_pct = (total_discovery / total_baseline * 100.0) if total_baseline > 0 else np.nan
    large_retention_pct = (large_retained / total_large * 100.0) if total_large > 0 else np.nan

    return {
        "total_baseline_winners": total_baseline,
        "total_discovery_winners": total_discovery,
        "total_large_winners": total_large,
        "retention_pct": retention_pct,
        "large_winner_retention_pct": large_retention_pct,
    }


def positive_winner_tier_counts(series: pd.Series) -> Dict[str, int]:
    r_all = pd.to_numeric(series, errors="coerce")
    r_wins = r_all[r_all > 0]
    small_max, medium_min, medium_max, large_min = adapt_return_thresholds(
        r_all,
        SMALL_WINNER_MAX,
        MEDIUM_WINNER_MIN,
        MEDIUM_WINNER_MAX,
        LARGE_WINNER_MIN,
    )

    return {
        "small": int(((r_wins > 0) & (r_wins < small_max)).sum()),
        "medium": int(((r_wins >= medium_min) & (r_wins < medium_max)).sum()),
        "large": int((r_wins >= large_min).sum()),
    }


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


def ranked_numeric_candidates(df_winners: pd.DataFrame, col: str, factor: str, min_winners: int) -> List[Dict[str, Any]]:
    s = pd.to_numeric(df_winners[col], errors="coerce").dropna()
    if len(s) < min_winners:
        return []

    qs = {q: float(s.quantile(q)) for q in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]}
    total = len(s)
    candidates: List[Dict[str, Any]] = []

    for q in [0.2, 0.3, 0.4, 0.5, 0.6]:
        v = qs[q]
        mask = s >= v
        captured = int(mask.sum())
        cond = Condition(factor=factor, kind="ge", value=round(v, 2))
        candidates.append(
            {
                "condition": cond,
                "winner_capture": captured,
                "winner_capture_pct": captured / total * 100.0,
                "rule_style": "threshold_ge",
            }
        )

    for q in [0.4, 0.5, 0.6, 0.7, 0.8]:
        v = qs[q]
        mask = s <= v
        captured = int(mask.sum())
        cond = Condition(factor=factor, kind="le", value=round(v, 2))
        candidates.append(
            {
                "condition": cond,
                "winner_capture": captured,
                "winner_capture_pct": captured / total * 100.0,
                "rule_style": "threshold_le",
            }
        )

    for lo_q, hi_q in [(0.2, 0.8), (0.25, 0.75), (0.3, 0.7), (0.35, 0.65), (0.4, 0.6)]:
        lo = float(s.quantile(lo_q))
        hi = float(s.quantile(hi_q))
        if lo >= hi:
            continue
        mask = (s >= lo) & (s <= hi)
        captured = int(mask.sum())
        cond = Condition(factor=factor, kind="range", value=(round(lo, 2), round(hi, 2)))
        candidates.append(
            {
                "condition": cond,
                "winner_capture": captured,
                "winner_capture_pct": captured / total * 100.0,
                "rule_style": "range",
            }
        )

    # Keep discovery candidates that are informative but not trivial all-capture bands.
    kept = [
        c for c in candidates
        if c["winner_capture"] >= min_winners and 45.0 <= c["winner_capture_pct"] <= 95.0
    ]

    kept.sort(
        key=lambda x: (
            x["winner_capture_pct"],
            1 if x["rule_style"].startswith("threshold") else 0,
        ),
        reverse=True,
    )

    dedup: Dict[str, Dict[str, Any]] = {}
    for c in kept:
        key = c["condition"].to_rule_text()
        dedup[key] = c
    return list(dedup.values())[:4]


def ranked_sqz_candidates(df_winners: pd.DataFrame, col: str, factor: str, min_winners: int) -> List[Dict[str, Any]]:
    s = pd.to_numeric(df_winners[col], errors="coerce").dropna()
    if len(s) < min_winners:
        return []

    total = len(s)
    # Stage 2 SQZ handling is binary-only and support-filter oriented.
    # Keep discovery centered on SQZ = 1 winner share.
    captured = int((s == 1).sum())
    cond = Condition(factor=factor, kind="eq", value=1)
    return [
        {
            "condition": cond,
            "winner_capture": captured,
            "winner_capture_pct": captured / total * 100.0,
            "rule_style": "binary",
        }
    ]


def evaluate_rule(
    df_sid: pd.DataFrame,
    ret_col: str,
    baseline: Dict[str, Any],
    conditions: List[Tuple[Condition, str]],
) -> Dict[str, Any]:
    if not conditions:
        return {}

    mask = pd.Series(True, index=df_sid.index)
    for cond, col in conditions:
        mask &= condition_mask(df_sid, col, cond).fillna(False)

    sub = df_sid[mask].copy()
    stats = basic_metrics(sub, ret_col)

    trades = int(stats["trades"])
    retention = (trades / baseline["trades"] * 100.0) if baseline["trades"] else np.nan
    expectancy = stats["expectancy"]
    lift = expectancy - baseline["expectancy"] if pd.notna(expectancy) and pd.notna(baseline["expectancy"]) else np.nan

    return {
        "rule": " AND ".join(c.to_rule_text() for c, _ in conditions),
        "trades": trades,
        "retention_pct": retention,
        "expectancy": expectancy,
        "win_rate": stats["win_rate"],
        "median_return": stats["median_return"],
        "lift": lift,
        "lift_pct": lift * 100.0 if pd.notna(lift) else np.nan,
    }


def enrichment_metrics(
    df_sid: pd.DataFrame,
    df_win: pd.DataFrame,
    col: str,
    cond: Condition,
) -> Dict[str, float]:
    all_mask = condition_mask(df_sid, col, cond).fillna(False)
    win_mask = condition_mask(df_win, col, cond).fillna(False)

    all_pct = float(all_mask.mean() * 100.0) if len(df_sid) else np.nan
    win_pct = float(win_mask.mean() * 100.0) if len(df_win) else np.nan
    enrichment = (win_pct / all_pct) if pd.notna(all_pct) and all_pct > 0 else np.nan

    return {
        "all_share_pct": all_pct,
        "winner_share_pct": win_pct,
        "winner_enrichment": enrichment,
    }


def winner_tier_counts(
    df_sid: pd.DataFrame,
    df_win: pd.DataFrame,
    ret_col: str,
    rule_mask: pd.Series,
) -> Dict[str, Any]:
    """
    Winner logic remains binary (return_pct > 0). These tiers are an analytical
    extension that shows whether a rule mostly captures small, medium, or large wins.
    """
    del df_win

    # Winner tiers are an analytical extension based on all positive-return trades,
    # independent of the stricter Stage 2 winner-discovery subset.
    overall_tiers = positive_winner_tier_counts(df_sid[ret_col])

    aligned_rule_mask = rule_mask.reindex(df_sid.index, fill_value=False)
    rule_tiers = positive_winner_tier_counts(df_sid.loc[aligned_rule_mask, ret_col])

    return {
        "Small_Winner_Trades_Captured": rule_tiers["small"],
        "Small_Winner_Trades_Total": overall_tiers["small"],
        "Small_Winner_Capture_Pct": (rule_tiers["small"] / overall_tiers["small"] * 100.0) if overall_tiers["small"] > 0 else np.nan,
        "Medium_Winner_Trades_Captured": rule_tiers["medium"],
        "Medium_Winner_Trades_Total": overall_tiers["medium"],
        "Medium_Winner_Capture_Pct": (rule_tiers["medium"] / overall_tiers["medium"] * 100.0) if overall_tiers["medium"] > 0 else np.nan,
        "Large_Winner_Trades_Captured": rule_tiers["large"],
        "Large_Winner_Trades_Total": overall_tiers["large"],
        "Large_Winner_Capture_Pct": (rule_tiers["large"] / overall_tiers["large"] * 100.0) if overall_tiers["large"] > 0 else np.nan,
    }


def baseline_winner_tier_totals(df_win: pd.DataFrame, ret_col: str) -> Dict[str, Any]:
    """
    Compute total counts of winners by tier for the baseline set.
    """
    totals = positive_winner_tier_counts(df_win[ret_col])
    
    return {
        "Small_Winner_Trades_Total": totals["small"],
        "Medium_Winner_Trades_Total": totals["medium"],
        "Large_Winner_Trades_Total": totals["large"],
    }


def plain_english_cluster_line(factor: str, candidate: Dict[str, Any]) -> str:
    cond = candidate["condition"]
    capture = candidate["winner_capture_pct"]
    if factor == "SQZ":
        return f"SQZ = {int(cond.value)} appears in {capture:.1f}% of winners"
    if cond.kind == "range":
        lo, hi = cond.value
        return f"{factor} {lo:.2f}-{hi:.2f} captures {capture:.1f}% of winners"
    if cond.kind == "ge":
        return f"{factor} >= {float(cond.value):.2f} captures {capture:.1f}% of winners"
    return f"{factor} <= {float(cond.value):.2f} captures {capture:.1f}% of winners"


def structural_priority_score(factor: str) -> int:
    if factor in STRUCTURAL_PRIORITY_FACTORS:
        return 2
    if factor in SECONDARY_PRIORITY_FACTORS:
        return 0
    return 1


def load_dataset(path: str, include_secondary: bool) -> Tuple[pd.DataFrame, str, str, Optional[str], Dict[str, str]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input file not found: {path}")

    df = pd.read_csv(path)

    sid_col = find_column(df, SID_ALIASES)
    ret_col = find_column(df, RETURN_ALIASES)
    date_col = find_column(df, DATE_ALIASES)

    if sid_col is None:
        raise ValueError(f"Missing SetupID column. Tried aliases: {SID_ALIASES}")
    if ret_col is None:
        raise ValueError(f"Missing return column. Tried aliases: {RETURN_ALIASES}")

    factor_map = factor_columns(df, include_secondary)
    missing_core = [f for f in CORE_FACTORS if f not in factor_map]
    if len(missing_core) == len(CORE_FACTORS):
        raise ValueError(f"None of the core factors found. Expected at least one of: {CORE_FACTORS}")

    df[sid_col] = pd.to_numeric(df[sid_col], errors="coerce")
    df[ret_col] = pd.to_numeric(df[ret_col], errors="coerce")

    if date_col is not None:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    for factor, col in factor_map.items():
        if factor in NUMERIC_FACTORS:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        elif factor in BINARY_FACTORS:
            df[col] = to_sqz_binary(df[col])

    needed = [sid_col, ret_col] + list(factor_map.values())
    df = df.dropna(subset=[sid_col, ret_col]).copy()
    df = df.dropna(subset=[c for c in needed if c in df.columns]).copy()

    df[sid_col] = df[sid_col].astype(int)

    return df, sid_col, ret_col, date_col, factor_map


def run_pipeline(args: argparse.Namespace) -> None:
    df, sid_col, ret_col, _date_col, factor_map = load_dataset(args.input, args.include_secondary_factors)
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    requested_sids = parse_setup_ids(args.setup_ids)
    available_sids = sorted(df[sid_col].dropna().unique().tolist())

    if requested_sids:
        chosen = [s for s in requested_sids if s in set(available_sids)]
    else:
        chosen = available_sids

    if not chosen:
        raise ValueError("No SetupIDs selected after filtering.")

    winner_dist_rows: List[Dict[str, Any]] = []
    single_rows: List[Dict[str, Any]] = []
    baseline_rows: List[Dict[str, Any]] = []
    relevance_rows: List[Dict[str, Any]] = []
    candidate_rows: List[Dict[str, Any]] = []
    promoted_rows: List[Dict[str, Any]] = []
    debug_rows: List[Dict[str, Any]] = []
    
    # Accumulate diagnostics across all SIDs
    all_diagnostics: List[Dict[str, Any]] = []

    print("\n" + "=" * 70)
    print("STAGE 2 - WINNER PROFILE DISCOVERY")
    print("=" * 70)
    print(f"Input rows: {len(df):,}")
    print(f"Selected SetupIDs: {chosen}")
    print(f"Core factors available: {[f for f in CORE_FACTORS if f in factor_map]}")
    print(f"Discovery Threshold: {args.discovery_threshold}%")
    if args.include_secondary_factors:
        print(f"Secondary factors available: {[f for f in SECONDARY_FACTORS if f in factor_map]}")

    for sid in chosen:
        df_sid = df[df[sid_col] == sid].copy()
        if len(df_sid) < args.min_trades_per_sid:
            print(f"SID {sid}: skipped, trades {len(df_sid)} < {args.min_trades_per_sid}")
            continue

        baseline = basic_metrics(df_sid, ret_col)
        
        # Calculate discovery diagnostics BEFORE filtering
        diagnostics = calculate_discovery_diagnostics(
            df_sid,
            ret_col,
            discovery_threshold=args.discovery_threshold,
            large_threshold=3.0,
        )
        all_diagnostics.append({"SID": sid, **diagnostics})
        
        # Use discovery threshold to identify candidates for rule mining
        wmask = winner_mask(
            df_sid=df_sid,
            ret_col=ret_col,
            mode="threshold",
            threshold=args.discovery_threshold,
            percentile=args.winner_percentile,
            expectancy_cutoff=args.winner_expectancy_cutoff,
        )
        df_win = df_sid[wmask.fillna(False)].copy()

        if len(df_win) < args.min_winner_trades:
            print(f"SID {sid}: skipped, winner trades {len(df_win)} < {args.min_winner_trades}")
            continue

        factors = [f for f in CORE_FACTORS if f in factor_map]
        if args.include_secondary_factors:
            factors += [f for f in SECONDARY_FACTORS if f in factor_map]

        per_factor_candidates: Dict[str, List[Dict[str, Any]]] = {}

        for factor in factors:
            col = factor_map[factor]
            if factor in NUMERIC_FACTORS:
                ranked = ranked_numeric_candidates(df_win, col, factor, args.min_winner_trades)
            else:
                ranked = ranked_sqz_candidates(df_win, col, factor, args.min_winner_trades)

            ranked_with_enrichment: List[Dict[str, Any]] = []
            for cand in ranked:
                em = enrichment_metrics(df_sid, df_win, col, cand["condition"])
                cand2 = dict(cand)
                cand2.update(em)
                ranked_with_enrichment.append(cand2)

            if factor in NUMERIC_FACTORS:
                ranked_with_enrichment = [
                    c for c in ranked_with_enrichment
                    if pd.notna(c["all_share_pct"]) and 5.0 <= c["all_share_pct"] <= 95.0
                ]

            ranked_with_enrichment.sort(
                key=lambda c: (
                    c.get("winner_enrichment", 0.0),
                    c.get("winner_capture_pct", 0.0),
                ),
                reverse=True,
            )

            per_factor_candidates[factor] = ranked_with_enrichment
            if ranked_with_enrichment:
                top = ranked_with_enrichment[0]
                winner_dist_rows.append(
                    {
                        "SID": sid,
                        "Factor": factor,
                        "Finding": (
                            f"SQZ = 1 appears in {top['winner_capture_pct']:.1f}% of winners"
                            if factor == "SQZ"
                            else plain_english_cluster_line(factor, top)
                        ),
                        "Rule": top["condition"].to_rule_text(),
                        "Winner_Capture_Pct": top["winner_capture_pct"],
                        "Winner_Trades_Captured": top["winner_capture"],
                        "Winner_Trades_Total": len(df_win),
                        "All_Trades_Share_Pct": top.get("all_share_pct"),
                        "Winner_Enrichment": top.get("winner_enrichment"),
                    }
                )

        single_validated: List[Dict[str, Any]] = []
        for factor in factors:
            for cand in per_factor_candidates.get(factor, [])[:2]:
                cond = cand["condition"]
                result = evaluate_rule(df_sid, ret_col, baseline, [(cond, factor_map[factor])])
                if not result:
                    continue
                filtered_trades = result["trades"]
                retention_pct = result["retention_pct"]
                lift_pct = result["lift_pct"]
                winner_enrichment = cand.get("winner_enrichment")
                
                # Tier 1: Strict standalone seeds
                tier_1_eligible = (
                    filtered_trades >= args.min_trades
                    and pd.notna(retention_pct) and retention_pct >= args.min_retention
                    and pd.notna(lift_pct) and lift_pct >= TIER_1_LIFT_MIN
                    and pd.notna(winner_enrichment) and winner_enrichment >= TIER_1_ENRICHMENT_MIN
                )
                
                # Tier 2: Controlled expansion for Stage 3 combos
                tier_2_retention_floor = max(args.min_retention, TIER_2_RETENTION_MIN)
                tier_2_trades_floor = max(args.min_trades, TIER_2_TRADES_MIN)
                tier_2_eligible = (
                    not tier_1_eligible  # Only if not already Tier 1
                    and filtered_trades >= tier_2_trades_floor
                    and pd.notna(retention_pct) and retention_pct >= tier_2_retention_floor
                    and pd.notna(lift_pct) and lift_pct >= TIER_2_LIFT_MIN
                    and pd.notna(winner_enrichment) and winner_enrichment >= TIER_2_ENRICHMENT_MIN
                )
                
                # Seed tier assignment
                if tier_1_eligible:
                    seed_tier = 1
                    combo_eligible = True
                    downstream_eligible = True
                    promoted = True
                elif tier_2_eligible:
                    seed_tier = 2
                    combo_eligible = True
                    downstream_eligible = False
                    promoted = False
                else:
                    seed_tier = 0
                    combo_eligible = False
                    downstream_eligible = False
                    promoted = False
                row = {
                    "SID": sid,
                    "Factor": factor,
                    "Rule": result["rule"],
                    "Winner_Capture_Pct": cand["winner_capture_pct"],
                    "Winner_Trades_Captured": cand["winner_capture"],
                    "Winner_Trades_Total": len(df_win),
                    "All_Trades_Share_Pct": cand.get("all_share_pct"),
                    "Winner_Enrichment": winner_enrichment,
                    "Filtered_Trades": filtered_trades,
                    "Retention_Pct": retention_pct,
                    "Filtered_Expectancy_Pct": result["expectancy"] * 100.0 if pd.notna(result["expectancy"]) else np.nan,
                    "Baseline_Expectancy_Pct": baseline["expectancy"] * 100.0 if pd.notna(baseline["expectancy"]) else np.nan,
                    "Expectancy_Lift_Pct": lift_pct,
                    "Filtered_Win_Rate": result["win_rate"],
                    "Filtered_Median_Return_Pct": result["median_return"] * 100.0 if pd.notna(result["median_return"]) else np.nan,
                    "Validation_Set": "all_trades_sid",
                    "Num_Factors": 1,
                    "Structural_Priority": structural_priority_score(factor),
                    "Seed_Tier": seed_tier,
                    "Combo_Eligible": combo_eligible,
                    "Downstream_Eligible": downstream_eligible,
                    "Promoted": promoted,
                    "Condition": cond,
                }
                
                # Compute winner tier capture statistics and merge into row
                rule_mask = condition_mask(df_sid, factor_map[factor], cond).fillna(False)
                tier_stats = winner_tier_counts(df_sid, df_win, ret_col, rule_mask)
                row.update(tier_stats)
                
                single_validated.append(row)
                single_rows.append({k: v for k, v in row.items() if k != "Condition"})

        single_validated.sort(
            key=lambda x: (
                x.get("Promoted", False),
                x.get("Combo_Eligible", False),
                x.get("Filtered_Expectancy_Pct", -9999.0),
                x.get("Expectancy_Lift_Pct", -9999.0),
                x.get("Winner_Enrichment", 0.0),
                x.get("Retention_Pct", -9999.0),
                x.get("Filtered_Trades", -9999.0),
                x.get("Structural_Priority", -1),
            ),
            reverse=True,
        )
        top_singles = single_validated[: args.top_k_single]

        normalized_candidates: List[Dict[str, Any]] = []
        for row in top_singles:
            normalized_candidates.append(
                {
                    "SID": sid,
                    "Rule": row["Rule"],
                    "Factor": row["Factor"],
                    "Num_Factors": 1,
                    "Winner_Capture_Pct": row["Winner_Capture_Pct"],
                    "Winner_Enrichment": row.get("Winner_Enrichment"),
                    "Val_Trades": row["Filtered_Trades"],
                    "Val_Expectancy_Pct": row["Filtered_Expectancy_Pct"],
                    "Val_Win_Rate": row["Filtered_Win_Rate"],
                    "Retention_Pct": row["Retention_Pct"],
                    "Expectancy_Uplift_Pct": row["Expectancy_Lift_Pct"],
                    "Median_Uplift_Pct": row["Filtered_Median_Return_Pct"] - (baseline["median_return"] * 100.0 if pd.notna(baseline["median_return"]) else np.nan),
                    "Structural_Priority": row.get("Structural_Priority"),
                    "Seed_Tier": row.get("Seed_Tier", 0),
                    "Combo_Eligible": row.get("Combo_Eligible", False),
                    "Downstream_Eligible": row.get("Downstream_Eligible", False),
                    "Promoted": row.get("Promoted", False),
                }
            )

        normalized_candidates.sort(
            key=lambda x: (
                x.get("Promoted", False),
                x.get("Combo_Eligible", False),
                x.get("Val_Expectancy_Pct", -9999.0),
                x.get("Expectancy_Uplift_Pct", -9999.0),
                x.get("Winner_Enrichment", -9999.0),
                x.get("Retention_Pct", -9999.0),
                x.get("Val_Trades", -9999.0),
                x.get("Structural_Priority", -1),
            ),
            reverse=True,
        )

        promoted_rules = [r for r in normalized_candidates if r["Promoted"]]

        baseline_rows.append(
            {
                "SID": sid,
                "Trades": baseline["trades"],
                "Wins": baseline["wins"],
                "Win_Rate": baseline["win_rate"],
                "Expectancy_Pct": baseline["expectancy"] * 100.0 if pd.notna(baseline["expectancy"]) else np.nan,
                "Median_Return_Pct": baseline["median_return"] * 100.0 if pd.notna(baseline["median_return"]) else np.nan,
                "Payoff_Ratio": baseline["payoff_ratio"],

            }
        )
        
        # Add winner tier totals to baseline
        if baseline["wins"] > 0:
            tier_totals = baseline_winner_tier_totals(df_sid, ret_col)
            baseline_rows[-1].update(tier_totals)
        for factor in factors:
            ranked = per_factor_candidates.get(factor, [])
            if not ranked:
                relevance_rows.append(
                    {
                        "SID": sid,
                        "Factor": factor,
                        "Relevant": "No",
                        "Label": "insufficient_winner_distribution",
                        "Effect_Size": np.nan,
                        "Relevance_Score": np.nan,
                        "Pattern": "NA",
                        "Reason": "no_viable_candidate",
                    }
                )
                continue

            top = ranked[0]
            relevance_rows.append(
                {
                    "SID": sid,
                    "Factor": factor,
                    "Relevant": "Yes" if top["winner_capture_pct"] >= 60.0 else "No",
                    "Label": "winner_cluster",
                    "Effect_Size": top["winner_capture_pct"] / 100.0,
                    "Relevance_Score": top["winner_capture_pct"],
                    "Pattern": top["condition"].to_rule_text(),
                    "Reason": "winner_capture",
                }
            )

        for i, row in enumerate(normalized_candidates, 1):
            # Keep legacy candidate schema for downstream compatibility with Stage 5 loader.
            candidate_rows.append(
                {
                    "SID": sid,
                    "Rule_Number": i,
                    "Rule": row["Rule"],
                    "Num_Factors": row["Num_Factors"],
                    "Winner_Capture_Pct": row["Winner_Capture_Pct"],
                    "Val_Trades": row["Val_Trades"],
                    "Val_Expectancy_Pct": row["Val_Expectancy_Pct"],
                    "Val_Win_Rate": row["Val_Win_Rate"],
                    "Retention_Pct": row["Retention_Pct"],
                    "Expectancy_Uplift_Pct": row["Expectancy_Uplift_Pct"],
                    "Median_Uplift_Pct": row["Median_Uplift_Pct"],
                    "Balance_Score": row["Expectancy_Uplift_Pct"],
                    "Composite_Score": row["Expectancy_Uplift_Pct"],
                    "Train_Val_Drift_Pct": 0.0,
                    "Promoted": row["Promoted"],
                }
            )
        for i, row in enumerate(promoted_rules, 1):
            # Keep legacy deployable filename/schema for downstream compatibility.
            # In Stage 2 semantics these are promoted single-factor rules only.
            promoted_rows.append(
                {
                    "SID": sid,
                    "Rule_Number": i,
                    "Rule": row["Rule"],
                    "Num_Factors": row["Num_Factors"],
                    "Winner_Capture_Pct": row["Winner_Capture_Pct"],
                    "Val_Trades": row["Val_Trades"],
                    "Val_Expectancy_Pct": row["Val_Expectancy_Pct"],
                    "Val_Win_Rate": row["Val_Win_Rate"],
                    "Retention_Pct": row["Retention_Pct"],
                    "Expectancy_Uplift_Pct": row["Expectancy_Uplift_Pct"],
                    "Median_Uplift_Pct": row["Median_Uplift_Pct"],
                    "Balance_Score": row["Expectancy_Uplift_Pct"],
                    "Composite_Score": row["Expectancy_Uplift_Pct"],
                    "Train_Val_Drift_Pct": 0.0,
                    "Promoted": row["Promoted"],
                }
            )

        debug_rows.append(
            {
                "SID": sid,
                "Winner_Trades": len(df_win),
                "Baseline_Trades": baseline["trades"],
                "Factors_Evaluated": len(factors),
                "Single_Candidates": len(single_validated),
                "Deployable": len(promoted_rules),
            }
        )

    # Write active Stage 2 outputs (clean default set).
    baseline_cols = [
        "SID", "Trades", "Wins", "Win_Rate", "Expectancy_Pct", "Median_Return_Pct", "Payoff_Ratio",
        "Small_Winner_Trades_Total", "Medium_Winner_Trades_Total", "Large_Winner_Trades_Total",
    ]
    relevance_cols = ["SID", "Factor", "Relevant", "Label", "Effect_Size", "Relevance_Score", "Pattern", "Reason"]
    candidate_cols = [
        "SID", "Rule_Number", "Rule", "Num_Factors", "Winner_Capture_Pct", "Val_Trades", "Val_Expectancy_Pct",
        "Val_Win_Rate", "Retention_Pct", "Expectancy_Uplift_Pct", "Median_Uplift_Pct", "Balance_Score", "Composite_Score",
        "Train_Val_Drift_Pct", "Downstream_Eligible", "Structural_Priority", "Promoted",
    ]
    debug_cols = ["SID", "Winner_Trades", "Baseline_Trades", "Factors_Evaluated", "Single_Candidates", "Deployable"]
    winner_dist_cols = [
        "SID", "Factor", "Finding", "Rule", "Winner_Capture_Pct", "Winner_Trades_Captured",
        "Winner_Trades_Total", "All_Trades_Share_Pct", "Winner_Enrichment",
    ]

    pd.DataFrame(baseline_rows, columns=baseline_cols).to_csv(output_dir / "baseline_by_sid.csv", index=False)
    pd.DataFrame(winner_dist_rows, columns=winner_dist_cols).to_csv(output_dir / "winner_distribution_by_sid.csv", index=False)
    single_cols = [
        "SID", "Factor", "Rule", "Winner_Capture_Pct", "Winner_Trades_Captured", "Winner_Trades_Total",
        "All_Trades_Share_Pct", "Winner_Enrichment",
        "Small_Winner_Trades_Captured", "Small_Winner_Trades_Total", "Small_Winner_Capture_Pct",
        "Medium_Winner_Trades_Captured", "Medium_Winner_Trades_Total", "Medium_Winner_Capture_Pct",
        "Large_Winner_Trades_Captured", "Large_Winner_Trades_Total", "Large_Winner_Capture_Pct",
        "Filtered_Trades", "Retention_Pct", "Filtered_Expectancy_Pct", "Baseline_Expectancy_Pct",
        "Expectancy_Lift_Pct", "Filtered_Win_Rate", "Filtered_Median_Return_Pct",
        "Validation_Set", "Num_Factors", "Seed_Tier", "Combo_Eligible", "Downstream_Eligible", "Structural_Priority", "Promoted",
    ]
    pd.DataFrame(single_rows, columns=single_cols).sort_values(
        [
            "SID", "Promoted", "Combo_Eligible", "Filtered_Expectancy_Pct", "Expectancy_Lift_Pct",
            "Winner_Enrichment", "Retention_Pct", "Filtered_Trades", "Structural_Priority",
        ],
        ascending=[True, False, False, False, False, False, False, False, False],
        na_position="last",
    ).to_csv(output_dir / "winner_single_rule_validation_by_sid.csv", index=False)

    if args.include_compatibility_outputs:
        # Compatibility outputs retained for legacy loaders and older workflows.
        pd.DataFrame(relevance_rows, columns=relevance_cols).to_csv(output_dir / "factor_relevance_by_sid.csv", index=False)
        pd.DataFrame([], columns=["SID", "Factor", "Range", "Range_Min", "Range_Max", "Category", "Expectancy_Pct", "Uplift_Pct", "Trades"]).to_csv(
            output_dir / "preferred_ranges_by_sid.csv", index=False
        )
        pd.DataFrame([], columns=["SID", "Factor", "Zone", "Zone_Min", "Zone_Max", "Category", "Expectancy_Pct", "Downside_Pct", "Trades"]).to_csv(
            output_dir / "avoid_zones_by_sid.csv", index=False
        )

        # "deployable" here means promoted single-factor rules passing Stage 2 gates.
        pd.DataFrame(candidate_rows, columns=["SID", "Rule_Number", *candidate_cols[2:]]).to_csv(output_dir / "candidate_rules_by_sid.csv", index=False)
        pd.DataFrame(promoted_rows, columns=["SID", "Rule_Number", *candidate_cols[2:]]).to_csv(output_dir / "deployable_rules_by_sid.csv", index=False)
        pd.DataFrame(debug_rows, columns=debug_cols).to_csv(output_dir / "debug_rule_pipeline_by_sid.csv", index=False)

        # Legacy compatibility placeholder retained only for workflows that still check this file.
        pd.DataFrame([], columns=[
            "SID", "Factors", "Rule", "Winner_Capture_Pct", "Winner_Trades_Captured", "Winner_Trades_Total",
            "Filtered_Trades", "Retention_Pct", "Filtered_Expectancy_Pct", "Baseline_Expectancy_Pct",
            "Expectancy_Lift_Pct", "Filtered_Win_Rate", "Filtered_Median_Return_Pct", "Validation_Set", "Num_Factors",
        ]).to_csv(output_dir / "winner_combo_rule_validation_by_sid.csv", index=False)
        compat_msg = "with compatibility CSVs"
    else:
        compat_msg = "without compatibility CSVs"

    # Print discovery diagnostics summary
    print("\n" + "=" * 70)
    print("DISCOVERY DIAGNOSTICS SUMMARY")
    print("=" * 70)
    
    if all_diagnostics:
        diag_df = pd.DataFrame(all_diagnostics)
        total_baseline = diag_df["total_baseline_winners"].sum()
        total_discovery = diag_df["total_discovery_winners"].sum()
        total_large = diag_df["total_large_winners"].sum()
        
        overall_retention = (total_discovery / total_baseline * 100.0) if total_baseline > 0 else np.nan
        overall_large_retention = (
            diag_df["total_large_winners"].sum() * diag_df["large_winner_retention_pct"].mean() / 100.0
        ) / total_large * 100.0 if total_large > 0 else np.nan
        
        print(f"Total Baseline Winners (return_pct > 0): {total_baseline:,}")
        print(f"Total Discovery Winners (return_pct >= {args.discovery_threshold}%): {total_discovery:,}")
        print(f"Total Large Winners (return_pct >= 3%): {total_large:,}")
        print(f"Overall Retention %: {overall_retention:.1f}%")
        print(f"Expected Range: 50-70%")
        print(f"Large Winner Retention %: {overall_large_retention:.1f}%")
        print(f"Expected: ≥ 80%")
        
        # Print per-SID diagnostics
        print("\nPer-SID Diagnostics:")
        for _, row in diag_df.iterrows():
            print(
                f"  SID {int(row['SID'])}: "
                f"baseline={int(row['total_baseline_winners'])}, "
                f"discovery={int(row['total_discovery_winners'])}, "
                f"retention={row['retention_pct']:.1f}% "
                f"(large_retention={row['large_winner_retention_pct']:.1f}%)"
            )

    print("\n" + "=" * 70)
    print("EXPORT COMPLETE")
    print("=" * 70)
    print(f"Outputs written to: {output_dir}")
    print(f"Remarks: Stage 2 generated active winner-discovery outputs {compat_msg}.")


def main() -> None:
    args = parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
