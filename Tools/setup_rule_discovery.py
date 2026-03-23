#!/usr/bin/env python3
"""
Setup Grader Validation-First Rule Miner
========================================

What this script does
---------------------
1) Splits each SetupID into TRAIN / VALIDATION
2) Mines single, AND, OR rules on TRAIN only
3) Scores each rule against SetupID baseline
4) Suppresses junk OR rules
5) Re-tests discovered rules on VALIDATION
6) Promotes only rules that survive validation
7) Builds 3-variable conditional heatmaps

This is designed for campaign-level trade data such as campaigns_clean.csv.
"""

from __future__ import annotations

import argparse
import itertools
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CONFIG
# ============================================================

RETURN_COL = "return_pct"
SID_COL = "sid"
ENTRY_DT_COL = "entry_dt"  # For chronological split

MIN_TRADES_RULE = 20
MIN_TRADES_SERIOUS = 40
MIN_TRADES_HEATMAP_CELL = 8

TRAIN_FRAC = 0.70
RANDOM_SEED = 42

# Promotion thresholds
MIN_TRAIN_LIFT = 0.40          # in percentage points, e.g. +0.40%
MIN_VALID_LIFT = 0.10          # looser because validation is smaller
MIN_RETAIN_PCT = 20.0          # rule must keep at least this % of baseline trades
MAX_EXPECTANCY_DROP_FROM_TRAIN = 1.25  # allowed deterioration in percentage points

# OR suppression
MIN_OR_REDUCTION_PCT = 5.0     # OR must reduce sample by at least 5%
MIN_OR_LIFT = 0.40             # or improve expectancy by at least 0.40%

# Balanced scoring weights
BALANCED_SCORE_WEIGHTS = {
    "expectancy": 0.45,
    "retention": 0.35,
    "stability": 0.20
}

THRESHOLDS = {
    "SS":  [60, 65, 70, 75, 80, 85],
    "TR":  [60, 65, 70, 75, 80, 85, 90],
    "RS":  [75, 80, 85, 90, 95],
    "CMP": [50, 60, 70, 80],
    "VQS": [50, 60, 70, 80],
    "DCR": [50, 60, 70, 80],
    "RMV": [10, 15, 20, 25, 30],
    "ADR": [2, 4, 6, 8],
    "ORC": [0.0, 0.25, 0.50, 0.75, 1.00],
    "DDV": [20_000_000, 50_000_000, 75_000_000, 100_000_000, 150_000_000, 200_000_000],
}

CATEGORIES = {
    "SQZ": [
        ["SqOn"],
        ["Fired"],
        ["Exp"],
        ["SqOn", "Fired"],
        ["Fired", "Exp"],
        ["SqOn", "Fired", "Exp"],
    ],
    "STATE": [
        ["Ready Now"],
        ["Near Term"],
        ["Ready Now", "Near Term"],
    ],
    "MODE": [
        ["Breakout"],
        ["Contrarian"],
    ],
    "GRP": [
        ["Strong"],
        ["Improving"],
        ["Strong", "Improving"],
        ["Weak"],
    ],
    "RUNNER": [
        [True],
        [False],
    ],
}

CATEGORY_ORDERS = {
    "SQZ": ["None", "SqOn", "Fired", "Exp"],
    "STATE": ["Not Imminent", "Near Term", "Ready Now"],
    "MODE": ["Contrarian", "Breakout"],
    "GRP": ["Weak", "Improving", "Strong"],
}

# 3-variable conditional heatmaps:
# (x, y, filter_name, filter_func)
CONDITIONAL_HEATMAPS = [
    ("RS", "TR",   "RMV_lte_25", lambda d: pd.to_numeric(d["RMV"], errors="coerce") <= 25),
    ("CMP", "DCR", "RS_gte_90",  lambda d: pd.to_numeric(d["RS"], errors="coerce") >= 90),
    ("CMP", "RMV", "TR_gte_85",  lambda d: pd.to_numeric(d["TR"], errors="coerce") >= 85),
    ("ORC", "GRP", "RS_gte_90",  lambda d: pd.to_numeric(d["RS"], errors="coerce") >= 90),
    ("CMP", "SQZ", "RMV_lte_25", lambda d: pd.to_numeric(d["RMV"], errors="coerce") <= 25),
]


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass(frozen=True)
class Rule:
    name: str
    func: Callable[[pd.DataFrame], pd.Series]
    rule_type: str  # single / AND / OR


# ============================================================
# LOADING / CLEANING
# ============================================================

def normalize_categories(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Normalize strings
    for col in ["SQZ", "STATE", "MODE", "GRP"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    # Fix common SQZ blanks
    if "SQZ" in df.columns:
        df["SQZ"] = df["SQZ"].replace({"nan": np.nan, "": np.nan}).fillna("None")

    if "RUNNER" in df.columns:
        df["RUNNER"] = (
            df["RUNNER"]
            .astype(str)
            .str.strip()
            .str.lower()
            .map({"true": True, "false": False, "1": True, "0": False})
        )

    return df


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = normalize_categories(df)

    numeric_cols = list(THRESHOLDS.keys()) + [RETURN_COL, "hold_days", "pnl_usd", SID_COL]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# ============================================================
# SPLITTING
# ============================================================

def train_validation_split(df_sid: pd.DataFrame, frac: float = TRAIN_FRAC, seed: int = RANDOM_SEED) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Chronological split per SetupID based on entry_dt.
    """
    if len(df_sid) < 2:
        return df_sid.copy(), df_sid.iloc[0:0].copy()

    # Sort by entry_dt if available
    if ENTRY_DT_COL in df_sid.columns:
        df_sid = df_sid.sort_values(ENTRY_DT_COL).reset_index(drop=True)

    cut = max(1, int(len(df_sid) * frac))
    return (
        df_sid.iloc[:cut].copy().reset_index(drop=True),
        df_sid.iloc[cut:].copy().reset_index(drop=True),
    )


# ============================================================
# METRICS
# ============================================================

def summarize_returns(rets: pd.Series) -> Dict[str, float]:
    rets = pd.to_numeric(rets, errors="coerce").dropna()
    if len(rets) == 0:
        return {
            "trades": 0,
            "expectancy": np.nan,
            "median_return": np.nan,
            "win_rate": np.nan,
            "stability": np.nan,
            "profit_factor": np.nan,
            "payoff_ratio": np.nan,
        }

    expc = float(rets.mean())
    n = int(len(rets))
    wins = rets[rets > 0]
    losses = rets[rets < 0]

    avg_win = float(wins.mean()) if len(wins) else np.nan
    avg_loss = float(losses.mean()) if len(losses) else np.nan
    pf = avg_win / abs(avg_loss) if len(wins) and len(losses) and avg_loss != 0 else np.nan
    payoff = avg_win / abs(avg_loss) if len(losses) and avg_loss != 0 else np.nan

    return {
        "trades": n,
        "expectancy": expc,
        "median_return": float(rets.median()),
        "win_rate": float((rets > 0).mean() * 100.0),
        "stability": float(expc * math.sqrt(n)),
        "profit_factor": pf,
        "payoff_ratio": payoff,
    }


def baseline_summary(df_sid: pd.DataFrame) -> Dict[str, float]:
    base = summarize_returns(df_sid[RETURN_COL])
    return base


def summarize_subset(df_sid: pd.DataFrame, mask: pd.Series, baseline: Dict[str, float]) -> Optional[Dict[str, float]]:
    sub = df_sid.loc[mask].copy()
    stats = summarize_returns(sub[RETURN_COL])

    if stats["trades"] < MIN_TRADES_RULE:
        return None

    lift = stats["expectancy"] - baseline["expectancy"]
    retain_pct = (stats["trades"] / baseline["trades"] * 100.0) if baseline["trades"] > 0 else np.nan

    stats.update({
        "lift_vs_baseline": lift,
        "retain_pct": retain_pct,
        "baseline_expectancy": baseline["expectancy"],
        "baseline_trades": baseline["trades"],
    })
    return stats


# ============================================================
# RULE GENERATION
# ============================================================

def build_single_rules(df_sid: pd.DataFrame) -> List[Rule]:
    rules: List[Rule] = []

    for col, vals in THRESHOLDS.items():
        if col not in df_sid.columns:
            continue

        for v in vals:
            rules.append(Rule(
                name=f"{col}>={v}",
                func=lambda d, c=col, vv=v: pd.to_numeric(d[c], errors="coerce") >= vv,
                rule_type="single",
            ))
            rules.append(Rule(
                name=f"{col}<={v}",
                func=lambda d, c=col, vv=v: pd.to_numeric(d[c], errors="coerce") <= vv,
                rule_type="single",
            ))

    for col, groups in CATEGORIES.items():
        if col not in df_sid.columns:
            continue

        for g in groups:
            label = ",".join(map(str, g))
            rules.append(Rule(
                name=f"{col} in {{{label}}}",
                func=lambda d, c=col, gg=g: d[c].isin(gg),
                rule_type="single",
            ))

    return rules


def combine_rules(rule_a: Rule, rule_b: Rule, op: str) -> Rule:
    if op == "AND":
        return Rule(
            name=f"({rule_a.name}) AND ({rule_b.name})",
            func=lambda d, ra=rule_a, rb=rule_b: ra.func(d) & rb.func(d),
            rule_type="AND",
        )
    elif op == "OR":
        return Rule(
            name=f"({rule_a.name}) OR ({rule_b.name})",
            func=lambda d, ra=rule_a, rb=rule_b: ra.func(d) | rb.func(d),
            rule_type="OR",
        )
    raise ValueError(op)


# ============================================================
# RULE MINING
# ============================================================

def should_keep_or_rule(stats: Dict[str, float]) -> bool:
    """
    Suppress broad OR rules that barely change the sample and add no real edge.
    """
    reduction_pct = 100.0 - stats["retain_pct"]
    return (reduction_pct >= MIN_OR_REDUCTION_PCT) or (stats["lift_vs_baseline"] >= MIN_OR_LIFT)


def mine_rules_train(df_train: pd.DataFrame, sid: int) -> pd.DataFrame:
    baseline = baseline_summary(df_train)
    single_rules = build_single_rules(df_train)

    # Limit to top single rules to prevent explosion
    single_rules = single_rules[:50]  # Limit to top 50 single rules

    rows = []

    # singles
    for rule in single_rules:
        mask = rule.func(df_train).fillna(False)
        stats = summarize_subset(df_train, mask, baseline)
        if stats is None:
            continue

        rows.append({
            "sid": sid,
            "rule_type": rule.rule_type,
            "rule": rule.name,
            **stats
        })

    # pairs (2-variable) - limit combinations
    pair_combinations = list(itertools.combinations(single_rules, 2))[:200]  # Limit to 200 pairs
    for ra, rb in pair_combinations:
        for op in ("AND", "OR"):
            rule = combine_rules(ra, rb, op)
            mask = rule.func(df_train).fillna(False)
            stats = summarize_subset(df_train, mask, baseline)
            if stats is None:
                continue

            if rule.rule_type == "OR" and not should_keep_or_rule(stats):
                continue

            rows.append({
                "sid": sid,
                "rule_type": rule.rule_type,
                "rule": rule.name,
                **stats
            })

    # triples (3-variable) - limit combinations
    triple_combinations = list(itertools.combinations(single_rules, 3))[:100]  # Limit to 100 triples
    for ra, rb, rc in triple_combinations:
        rule = combine_rules(combine_rules(ra, rb, "AND"), rc, "AND")
        mask = rule.func(df_train).fillna(False)
        stats = summarize_subset(df_train, mask, baseline)
        if stats is None:
            continue

        rows.append({
            "sid": sid,
            "rule_type": "AND",
            "rule": rule.name,
            **stats
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out = out.sort_values(
        ["lift_vs_baseline", "expectancy", "stability", "trades"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)

    return out


# ============================================================
# VALIDATION
# ============================================================

def rule_string_to_mask(df: pd.DataFrame, rule_str: str) -> pd.Series:
    """
    Rebuilds mask from rule string using the same grammar generated by the miner.
    This avoids serializing callables.
    """
    # Base parser for simple rule types only from our generator.
    rule_str = rule_str.strip()

    def parse_simple(simple: str) -> pd.Series:
        simple = simple.strip()

        if " in {" in simple:
            col, rest = simple.split(" in {", 1)
            values = rest.rstrip("}")
            vals = [v.strip() for v in values.split(",")]
            if col == "RUNNER":
                parsed = []
                for v in vals:
                    if v == "True":
                        parsed.append(True)
                    elif v == "False":
                        parsed.append(False)
                    else:
                        parsed.append(v)
                vals = parsed
            return df[col].isin(vals)

        if ">=" in simple:
            col, val = simple.split(">=", 1)
            return pd.to_numeric(df[col], errors="coerce") >= float(val)

        if "<=" in simple:
            col, val = simple.split("<=", 1)
            return pd.to_numeric(df[col], errors="coerce") <= float(val)

        raise ValueError(f"Cannot parse simple rule: {simple}")

    # compound
    if ") AND (" in rule_str:
        inner = rule_str[1:-1] if rule_str.startswith("(") and rule_str.endswith(")") else rule_str
        left, right = inner.split(") AND (", 1)
        return parse_simple(left.lstrip("(")) & parse_simple(right.rstrip(")"))

    if ") OR (" in rule_str:
        inner = rule_str[1:-1] if rule_str.startswith("(") and rule_str.endswith(")") else rule_str
        left, right = inner.split(") OR (", 1)
        return parse_simple(left.lstrip("(")) | parse_simple(right.rstrip(")"))

    return parse_simple(rule_str)


def validate_rules(df_valid: pd.DataFrame, sid: int, train_rules: pd.DataFrame) -> pd.DataFrame:
    if train_rules.empty or df_valid.empty:
        return pd.DataFrame()

    baseline = baseline_summary(df_valid)
    rows = []

    for _, r in train_rules.iterrows():
        try:
            mask = rule_string_to_mask(df_valid, r["rule"]).fillna(False)
        except Exception:
            continue

        stats = summarize_subset(df_valid, mask, baseline)
        if stats is None:
            continue

        rows.append({
            "sid": sid,
            "rule_type": r["rule_type"],
            "rule": r["rule"],
            **stats
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out = out.sort_values(
        ["lift_vs_baseline", "expectancy", "stability", "trades"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)

    return out


def promote_rules(train_df: pd.DataFrame, valid_df: pd.DataFrame) -> pd.DataFrame:
    if train_df.empty or valid_df.empty:
        return pd.DataFrame()

    merged = train_df.merge(
        valid_df,
        on=["sid", "rule_type", "rule"],
        suffixes=("_train", "_valid"),
        how="inner"
    )

    if merged.empty:
        return merged

    merged["expectancy_drop"] = merged["expectancy_train"] - merged["expectancy_valid"]

    promoted = merged[
        (merged["lift_vs_baseline_train"] >= MIN_TRAIN_LIFT) &
        (merged["lift_vs_baseline_valid"] >= MIN_VALID_LIFT) &
        (merged["retain_pct_train"] >= MIN_RETAIN_PCT) &
        (merged["retain_pct_valid"] >= MIN_RETAIN_PCT) &
        (merged["expectancy_valid"] > 0) &
        (merged["expectancy_drop"] <= MAX_EXPECTANCY_DROP_FROM_TRAIN)
    ].copy()

    if promoted.empty:
        return promoted

    promoted["promotion_score"] = (
        promoted["lift_vs_baseline_valid"] * 0.50 +
        promoted["lift_vs_baseline_train"] * 0.30 +
        (promoted["retain_pct_valid"] / 100.0) * 0.20
    )

    promoted = promoted.sort_values(
        ["promotion_score", "lift_vs_baseline_valid", "expectancy_valid", "trades_valid"],
        ascending=[False, False, False, False]
    ).reset_index(drop=True)

    return promoted


# ============================================================
# CANDIDATE GATE SUMMARY
# ============================================================

def candidate_gate_summary(promoted: pd.DataFrame) -> pd.DataFrame:
    if promoted.empty:
        return pd.DataFrame()

    cols = [
        "rule",
        "rule_type",
        "trades_train",
        "expectancy_train",
        "lift_vs_baseline_train",
        "retain_pct_train",
        "trades_valid",
        "expectancy_valid",
        "lift_vs_baseline_valid",
        "retain_pct_valid",
        "promotion_score",
        "balanced_score_1",
        "balanced_score_2",
    ]
    out = promoted[cols].copy()
    out = out.rename(columns={
        "expectancy_train": "train_exp",
        "lift_vs_baseline_train": "train_lift",
        "retain_pct_train": "train_retain_pct",
        "expectancy_valid": "valid_exp",
        "lift_vs_baseline_valid": "valid_lift",
        "retain_pct_valid": "valid_retain_pct",
    })
    return out


# ============================================================
# HEATMAPS
# ============================================================

def make_binned_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(index=df.index, dtype="object")

    if col in CATEGORY_ORDERS:
        s = df[col].astype(str).str.strip()
        unknown = sorted(set(s.dropna().unique()) - set(CATEGORY_ORDERS[col]))
        categories = CATEGORY_ORDERS[col] + unknown
        cat = pd.Categorical(s, categories=categories, ordered=True)
        return pd.Series(cat, index=df.index, name=col)

    if col not in THRESHOLDS:
        s = pd.to_numeric(df[col], errors="coerce")
        return pd.qcut(s, q=4, duplicates="drop").astype(str)

    bins = [-np.inf] + THRESHOLDS[col] + [np.inf]
    labels = []
    for i in range(len(bins) - 1):
        lo = bins[i]
        hi = bins[i + 1]
        if np.isneginf(lo):
            labels.append(f"<{hi}")
        elif np.isposinf(hi):
            labels.append(f">={lo}")
        else:
            labels.append(f"{lo}-{hi}")

    s = pd.to_numeric(df[col], errors="coerce")
    return pd.cut(s, bins=bins, labels=labels, include_lowest=True, right=False)


def build_conditional_heatmap_tables(df: pd.DataFrame, x_col: str, y_col: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    x = make_binned_series(df, x_col)
    y = make_binned_series(df, y_col)

    work = df.copy()
    work["_x"] = x
    work["_y"] = y
    work[RETURN_COL] = pd.to_numeric(work[RETURN_COL], errors="coerce")

    grouped = work.groupby(["_y", "_x"], observed=False)
    exp_tbl = grouped[RETURN_COL].mean().unstack("_x")
    n_tbl = grouped[RETURN_COL].count().unstack("_x")

    exp_tbl = exp_tbl.where(n_tbl >= MIN_TRADES_HEATMAP_CELL)

    return exp_tbl, n_tbl


def render_heatmap(table: pd.DataFrame, title: str, outpath: Path, fmt: str = ".3f", annotate_counts: Optional[pd.DataFrame] = None) -> None:
    if table.empty:
        return

    fig_w = max(7, 1.2 * max(1, table.shape[1]))
    fig_h = max(4, 0.9 * max(1, table.shape[0]))

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    values = table.values.astype(float)

    im = ax.imshow(values, aspect="auto", cmap="RdYlGn")

    ax.set_xticks(np.arange(table.shape[1]))
    ax.set_xticklabels([str(c) for c in table.columns], rotation=45, ha="right")
    ax.set_yticks(np.arange(table.shape[0]))
    ax.set_yticklabels([str(i) for i in table.index])

    ax.set_title(title)
    ax.set_xlabel(table.columns.name if table.columns.name else "")
    ax.set_ylabel(table.index.name if table.index.name else "")

    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            val = table.iat[i, j]
            if pd.isna(val):
                txt = ""
            else:
                txt = format(val, fmt)
                if annotate_counts is not None and i < annotate_counts.shape[0] and j < annotate_counts.shape[1]:
                    cnt = annotate_counts.iat[i, j]
                    if pd.notna(cnt):
                        txt = f"{format(val, fmt)}\n(n={int(cnt)})"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8)

    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Value", rotation=270, labelpad=15)

    fig.tight_layout()
    fig.savefig(outpath, dpi=180, bbox_inches="tight")
    plt.close(fig)


def export_conditional_heatmaps(df_sid: pd.DataFrame, outdir: Path, sid_label: str) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    for x_col, y_col, filter_name, filter_func in CONDITIONAL_HEATMAPS:
        missing = [c for c in [x_col, y_col] if c not in df_sid.columns]
        if missing:
            continue

        try:
            mask = filter_func(df_sid).fillna(False)
        except Exception:
            continue

        sub = df_sid.loc[mask].copy()
        if len(sub) < MIN_TRADES_RULE:
            continue

        exp_tbl, n_tbl = build_conditional_heatmap_tables(sub, x_col, y_col)
        if exp_tbl.empty or n_tbl.empty:
            continue

        name = f"{x_col}_x_{y_col}__{filter_name}"

        exp_tbl.to_csv(outdir / f"expectancy__{name}.csv")
        n_tbl.to_csv(outdir / f"trades__{name}.csv")

        render_heatmap(
            exp_tbl,
            title=f"{sid_label} Expectancy: {x_col} x {y_col} | {filter_name}",
            outpath=outdir / f"expectancy__{name}.png",
            fmt=".3f",
            annotate_counts=n_tbl,
        )

        render_heatmap(
            n_tbl,
            title=f"{sid_label} Trades: {x_col} x {y_col} | {filter_name}",
            outpath=outdir / f"trades__{name}.png",
            fmt=".0f",
        )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input campaigns_clean.csv")
    args = parser.parse_args()

    df = load_data(args.input)

    out_rule = Path("rule_validation")
    out_heat = Path("conditional_heatmaps")
    out_rule.mkdir(exist_ok=True)
    out_heat.mkdir(exist_ok=True)

    sid_values = sorted([int(x) for x in df[SID_COL].dropna().unique()])

    all_promoted = []

    for sid in sid_values:
        df_sid = df[df[SID_COL] == sid].copy().reset_index(drop=True)
        if len(df_sid) < MIN_TRADES_RULE:
            continue

        print(f"Analyzing SID {sid}")

        train_df, valid_df = train_validation_split(df_sid, frac=TRAIN_FRAC, seed=RANDOM_SEED + sid)

        train_base = pd.DataFrame([{"sid": sid, "dataset": "train", **baseline_summary(train_df)}])
        valid_base = pd.DataFrame([{"sid": sid, "dataset": "validation", **baseline_summary(valid_df)}])
        pd.concat([train_base, valid_base], ignore_index=True).to_csv(
            out_rule / f"sid_{sid}_baseline_summary.csv", index=False
        )

        train_rules = mine_rules_train(train_df, sid)
        train_rules.to_csv(out_rule / f"sid_{sid}_train_rules.csv", index=False)

        valid_rules = validate_rules(valid_df, sid, train_rules)
        valid_rules.to_csv(out_rule / f"sid_{sid}_valid_rules.csv", index=False)

        promoted = promote_rules(train_rules, valid_rules)
        promoted.to_csv(out_rule / f"sid_{sid}_promoted_rules.csv", index=False)

        candidate_gate_summary(promoted).to_csv(
            out_rule / f"sid_{sid}_candidate_gates.csv", index=False
        )

        if not promoted.empty:
            all_promoted.append(promoted)

        export_conditional_heatmaps(
            df_sid,
            out_heat / f"sid_{sid}",
            f"SID {sid}"
        )

    if all_promoted:
        pd.concat(all_promoted, ignore_index=True).to_csv(
            out_rule / "all_promoted_rules.csv", index=False
        )

    # Runner analysis
    print("Running runner analysis...")
    run_runner_analysis(df, out_rule, out_heat)

    print("Finished.")


# ============================================================
# RUNNER ANALYSIS
# ============================================================

def run_runner_analysis(df: pd.DataFrame, out_rule: Path, out_heat: Path) -> None:
    """Dedicated runner analysis for RUNNER=True trades."""
    if "RUNNER" not in df.columns:
        return

    runner_df = df[df["RUNNER"] == True].copy()
    if runner_df.empty:
        return

    runner_out_rule = out_rule / "runner"
    runner_out_heat = out_heat / "runner"
    runner_out_rule.mkdir(exist_ok=True)
    runner_out_heat.mkdir(exist_ok=True)

    sid_values = sorted([int(x) for x in runner_df[SID_COL].dropna().unique()])

    all_runner_promoted = []

    for sid in sid_values:
        df_sid = runner_df[runner_df[SID_COL] == sid].copy().reset_index(drop=True)
        if len(df_sid) < MIN_TRADES_RULE:
            continue

        print(f"Analyzing Runner SID {sid}")

        train_df, valid_df = train_validation_split(df_sid, frac=TRAIN_FRAC, seed=RANDOM_SEED + sid)

        train_base = pd.DataFrame([{"sid": sid, "dataset": "train", **baseline_summary(train_df)}])
        valid_base = pd.DataFrame([{"sid": sid, "dataset": "validation", **baseline_summary(valid_df)}])
        pd.concat([train_base, valid_base], ignore_index=True).to_csv(
            runner_out_rule / f"sid_{sid}_runner_baseline_summary.csv", index=False
        )

        train_rules = mine_rules_train(train_df, sid)
        train_rules.to_csv(runner_out_rule / f"sid_{sid}_runner_train_rules.csv", index=False)

        valid_rules = validate_rules(valid_df, sid, train_rules)
        valid_rules.to_csv(runner_out_rule / f"sid_{sid}_runner_valid_rules.csv", index=False)

        promoted = promote_rules(train_rules, valid_rules)
        promoted.to_csv(runner_out_rule / f"sid_{sid}_runner_promoted_rules.csv", index=False)

        candidate_gate_summary(promoted).to_csv(
            runner_out_rule / f"sid_{sid}_runner_candidate_gates.csv", index=False
        )

        if not promoted.empty:
            all_runner_promoted.append(promoted)

        export_conditional_heatmaps(
            df_sid,
            runner_out_heat / f"sid_{sid}",
            f"SID {sid} Runner"
        )

    if all_runner_promoted:
        pd.concat(all_runner_promoted, ignore_index=True).to_csv(
            runner_out_rule / "all_runner_promoted_rules.csv", index=False
        )


if __name__ == "__main__":
    main()