#!/usr/bin/env python3
"""
Validation-First Rule Miner v3 — Balanced Expectancy + Sample Size Focus
Focus: time-split, robust parsing, controlled 2&3&4-var mining, pareto + knee

Major updates (2026):
- Added ORC (fib-like pivot distance) and Sqz (TTM Squeeze state)
- Full direction control (≥ / ≤ per variable)
- Improved stability proxy
- Balanced formula tuned to match observed CSV pattern
"""

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ────────────────────────────────────────────────
#  Config
# ────────────────────────────────────────────────

RETURN_COL    = "return_pct"
ENTRY_DT_COL  = "entry_dt"
SID_COL       = "sid"

MIN_RULE_TRADES       = 18
MIN_SERIOUS_TRADES    = 35
MIN_HEATMAP_CELL      = 8

TRAIN_FRAC            = 0.70

# Promotion / acceptance thresholds
MIN_VALID_EXPECTANCY  = 0.002
MIN_LIFT_PP           = 0.0008

# Controlled mining limits — bumped slightly for new vars
MAX_STRONG_SINGLES    = 20
MAX_2VAR_RULES        = 140
MAX_3VAR_RULES        = 90
MAX_4VAR_RULES        = 60

THRESHOLDS = {
    "TR":   [60, 70, 80, 90],           # Trend Health                  ≥
    "RS":   [75, 85, 90, 95],           # Relative Strength             ≥
    "CMP":  [40, 50, 60, 70],           # Compression Score             ≤
    "VQS":  [40, 50, 60, 70],           # Volume Quality Score          ≥
    "DCR":  [50, 60, 70, 80],           # Demand/Close Location         ≥
    "RMV":  [15, 25, 35, 50],           # Relative Market Volume        ≤
    "ORC":  [-0.2, 0.0, 0.3, 0.5, 0.618, 0.786],  # Oracle / Fib distance  ≤ or ≥
    "Sqz":  ["SqOn", "Fired", "Exp", "-"],        # TTM Squeeze state (1H)  exact
}

# Direction: True = ≥ is better, False = ≤ is better, None = categorical/exact
DIRECTION = {
    "TR":   True,
    "RS":   True,
    "CMP":  False,
    "VQS":  True,
    "DCR":  True,
    "RMV":  False,
    "ORC":  False,          # default ≤ (shallow retracement / near PP), but we test both
    "Sqz":  None,           # exact match
}

# For ORC we want to test BOTH directions — so we'll duplicate rules where sensible
ORC_TEST_BOTH = True

# =====================================================================
# Startup print
# =====================================================================

print("Rule Miner v3 — with ORC & Sqz support")
print(f"  - Indicators: {list(THRESHOLDS.keys())}")
print(f"  - Values:     {[len(v) for v in THRESHOLDS.values()]}")
print(f"  - Direction:  {[DIRECTION.get(k, 'cat') for k in THRESHOLDS]}")
print(f"  - Limits:     singles≤{MAX_STRONG_SINGLES} | 2-var≤{MAX_2VAR_RULES} | 3-var≤{MAX_3VAR_RULES} | 4-var≤{MAX_4VAR_RULES}")
print("-" * 80)

# ────────────────────────────────────────────────
#  Data structures
# ────────────────────────────────────────────────

@dataclass
class Rule:
    name: str
    func: Callable[[pd.DataFrame], pd.Series]

# ────────────────────────────────────────────────
#  Metrics
# ────────────────────────────────────────────────

def summarize_returns(s: pd.Series) -> dict:
    s = pd.to_numeric(s.dropna(), errors='coerce')
    if len(s) == 0:
        return {"trades":0, "expectancy":np.nan, "win_rate":np.nan, "median_ret":np.nan, "stability":np.nan}

    n     = len(s)
    ex    = float(s.mean())
    wr    = float((s > 0).mean())
    med   = float(s.median())
    std   = float(s.std()) if n > 1 else 0.0
    # Improved stability: expectancy × √n × (1 - normalized volatility penalty)
    norm_std = min(1.0, std / max(0.0001, abs(ex))) if ex != 0 else 1.0
    stab  = ex * math.sqrt(max(1, n)) * max(0.1, 1.0 - norm_std)
    return {
        "trades": n,
        "expectancy": ex,
        "win_rate": wr,
        "median_ret": med,
        "stability": stab,
    }

# ────────────────────────────────────────────────
#  Time-based split
# ────────────────────────────────────────────────

def time_split(df: pd.DataFrame, train_frac: float = TRAIN_FRAC) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if ENTRY_DT_COL in df.columns:
        df = df.sort_values(ENTRY_DT_COL).reset_index(drop=True)
    else:
        df = df.sort_index().reset_index(drop=True)

    n = len(df)
    if n < 40:
        cut = n // 2
    else:
        cut = max(20, int(n * train_frac))

    train = df.iloc[:cut].copy()
    valid = df.iloc[cut:].copy()
    return train, valid

# ────────────────────────────────────────────────
#  Rule building — numerical + categorical
# ────────────────────────────────────────────────

def build_single_rules(df: pd.DataFrame) -> List[Rule]:
    rules = []

    for col, vals in THRESHOLDS.items():
        if col not in df.columns:
            continue

        for v in vals:
            if DIRECTION.get(col) is None:  # categorical (Sqz)
                name = f"{col} = {v}"
                func = lambda d, c=col, val=v: d[c].astype(str) == str(val)
            else:
                if col == "ORC" and ORC_TEST_BOTH:
                    # Test both directions for ORC
                    for direction in [True, False]:
                        op = "≥" if direction else "≤"
                        name = f"{col} {op} {v}"
                        func = lambda d, c=col, thresh=v, ge=direction: (
                            pd.to_numeric(d[c], errors='coerce') >= thresh if ge
                            else pd.to_numeric(d[c], errors='coerce') <= thresh
                        )
                        rules.append(Rule(name, func))
                    continue  # skip default so we don't duplicate

                op = "≥" if DIRECTION[col] else "≤"
                name = f"{col} {op} {v}"
                func = lambda d, c=col, thresh=v, ge=DIRECTION[col]: (
                    pd.to_numeric(d[c], errors='coerce') >= thresh if ge
                    else pd.to_numeric(d[c], errors='coerce') <= thresh
                )

            rules.append(Rule(name, func))

    return rules


def and_rule(r1: Rule, r2: Rule) -> Rule:
    return Rule(
        f"({r1.name}) AND ({r2.name})",
        lambda d, a=r1, b=r2: a.func(d) & b.func(d)
    )


def generate_controlled_rules(df_train: pd.DataFrame) -> List[Rule]:
    """
    Staged generation: singles → 2-var → 3-var → 4-var
    Keeps only strongest candidates at each level
    """
    # ─── 1. Singles ───────────────────────────────────────
    singles = build_single_rules(df_train)

    base = summarize_returns(df_train[RETURN_COL])
    scored = []
    for r in singles:
        mask = r.func(df_train).fillna(False)
        sub = summarize_returns(df_train.loc[mask, RETURN_COL])
        if sub["trades"] >= MIN_RULE_TRADES:
            lift = sub["expectancy"] - base["expectancy"] if not np.isnan(base["expectancy"]) else 0
            combined = sub["expectancy"] * math.sqrt(max(1, sub["trades"]))
            score_proxy = lift + combined * 0.015  # slight lift bias
            scored.append((score_proxy, combined, r))

    scored.sort(key=lambda x: x[0], reverse=True)
    strong_singles = [r for _,_,r in scored[:MAX_STRONG_SINGLES]]

    if not strong_singles:
        print("No strong single rules passed min trades threshold.")
        return []

    # ─── 2. 2-var ─────────────────────────────────────────
    pairs = []
    seen = set()
    for i in range(len(strong_singles)):
        for j in range(i+1, len(strong_singles)):
            r = and_rule(strong_singles[i], strong_singles[j])
            if r.name in seen: continue
            seen.add(r.name)
            pairs.append(r)
            if len(pairs) >= MAX_2VAR_RULES * 4: break
        if len(pairs) >= MAX_2VAR_RULES * 4: break

    pair_scores = []
    for r in pairs:
        mask = r.func(df_train)
        sub = summarize_returns(df_train.loc[mask, RETURN_COL])
        if sub["trades"] >= MIN_RULE_TRADES:
            combined = sub["expectancy"] * math.sqrt(max(1, sub["trades"]))
            pair_scores.append((combined, r))

    pair_scores.sort(key=lambda x: x[0], reverse=True)
    strong_pairs = [r for _,r in pair_scores[:MAX_2VAR_RULES]]

    # ─── 3-var ─────────────────────────────────────────
    triples = []
    seen = set()
    for p in strong_pairs:
        for s in strong_singles[:10]:
            r = and_rule(p, s)
            if r.name in seen: continue
            seen.add(r.name)
            triples.append(r)
            if len(triples) >= MAX_3VAR_RULES * 5: break
        if len(triples) >= MAX_3VAR_RULES * 5: break

    triple_scores = []
    for r in triples:
        mask = r.func(df_train)
        sub = summarize_returns(df_train.loc[mask, RETURN_COL])
        if sub["trades"] >= MIN_RULE_TRADES:
            combined = sub["expectancy"] * math.sqrt(max(1, sub["trades"]))
            triple_scores.append((combined, r))

    triple_scores.sort(key=lambda x: x[0], reverse=True)
    strong_triples = [r for _,r in triple_scores[:MAX_3VAR_RULES]]

    # ─── 4-var ─────────────────────────────────────────
    quads = []
    seen = set()
    for t in strong_triples:
        for s in strong_singles[:8]:
            r = and_rule(t, s)
            if r.name in seen: continue
            seen.add(r.name)
            quads.append(r)
            if len(quads) >= MAX_4VAR_RULES * 4: break
        if len(quads) >= MAX_4VAR_RULES * 4: break

    quad_scores = []
    for r in quads:
        mask = r.func(df_train)
        sub = summarize_returns(df_train.loc[mask, RETURN_COL])
        if sub["trades"] >= MIN_RULE_TRADES:
            combined = sub["expectancy"] * math.sqrt(max(1, sub["trades"]))
            quad_scores.append((combined, r))

    quad_scores.sort(key=lambda x: x[0], reverse=True)
    strong_quads = [r for _,r in quad_scores[:MAX_4VAR_RULES]]

    # ─── Final collection ────────────────────────────────
    all_rules = strong_singles + strong_pairs + strong_triples + strong_quads

    print(f"  Generated: {len(strong_singles)} singles | "
          f"{len(strong_pairs)} 2-var | "
          f"{len(strong_triples)} 3-var | "
          f"{len(strong_quads)} 4-var  → total {len(all_rules)}")

    return all_rules


# ────────────────────────────────────────────────
#  Evaluation
# ────────────────────────────────────────────────

def evaluate_rules(rules: List[Rule], df: pd.DataFrame, baseline: dict) -> pd.DataFrame:
    rows = []
    for r in rules:
        mask = r.func(df).fillna(False)
        stats = summarize_returns(df.loc[mask, RETURN_COL])
        if stats["trades"] < MIN_RULE_TRADES:
            continue
        lift = stats["expectancy"] - baseline["expectancy"] if not np.isnan(baseline["expectancy"]) else np.nan
        retain = stats["trades"] / baseline["trades"] * 100 if baseline["trades"] > 0 else 0
        balanced = stats["expectancy"] * math.sqrt(max(1, stats["trades"])) * max(0.05, stats["stability"] / max(0.0001, abs(stats["expectancy"])))
        rows.append({
            "rule": r.name,
            "trades": stats["trades"],
            "expectancy": stats["expectancy"],
            "win_rate": stats["win_rate"],
            "median_ret": stats["median_ret"],
            "stability": stats["stability"],
            "lift": lift,
            "retain_pct": retain,
            "balanced": balanced,
        })
    df_out = pd.DataFrame(rows)
    return df_out.sort_values("balanced", ascending=False)


# ────────────────────────────────────────────────
#  Pareto + Knee (unchanged but cleaned)
# ────────────────────────────────────────────────

def pareto_front(df: pd.DataFrame) -> pd.DataFrame:
    if "trades_valid" not in df.columns or "expectancy_valid" not in df.columns:
        print("Warning: pareto_front called without valid columns")
        return pd.DataFrame()

    pts = df[["trades_valid", "expectancy_valid"]].dropna().values
    if len(pts) < 2:
        return df

    is_efficient = np.ones(len(pts), dtype=bool)
    for i, c in enumerate(pts):
        for j, d in enumerate(pts):
            if i == j: continue
            if d[0] >= c[0] and d[1] >= c[1] and (d[0] > c[0] or d[1] > c[1]):
                is_efficient[i] = False
                break
    efficient_df = df[is_efficient].copy()
    return efficient_df.sort_values("trades_valid", ascending=False)


def find_knee(df_sorted_desc_trades: pd.DataFrame) -> pd.DataFrame:
    if "trades_valid" not in df_sorted_desc_trades.columns or len(df_sorted_desc_trades) < 4:
        return df_sorted_desc_trades.iloc[[0]] if not df_sorted_desc_trades.empty else pd.DataFrame()

    df = df_sorted_desc_trades.reset_index(drop=True)
    delta_exp    = -df["expectancy_valid"].diff().fillna(0)
    delta_trades = -df["trades_valid"].diff().fillna(0)
    ratio        = delta_exp / (delta_trades + 1e-6)
    knee_idx     = ratio.argmax()
    return df.iloc[[knee_idx]]


# ────────────────────────────────────────────────
#  Main per SetupID
# ────────────────────────────────────────────────

def process_sid(df_sid: pd.DataFrame, sid: int, outdir: Path):
    train, valid = time_split(df_sid)

    base_train = summarize_returns(train[RETURN_COL])
    base_valid = summarize_returns(valid[RETURN_COL])

    pd.DataFrame([
        {"set":"train", "sid":sid, **base_train},
        {"set":"valid", "sid":sid, **base_valid},
    ]).to_csv(outdir / f"sid_{sid}_baseline.csv", index=False)

    rules = generate_controlled_rules(train)

    df_train_eval = evaluate_rules(rules, train, base_train)
    df_valid_eval = evaluate_rules(rules, valid, base_valid)

    df_train_eval.to_csv(outdir / f"sid_{sid}_train_eval.csv", index=False)
    df_valid_eval.to_csv(outdir / f"sid_{sid}_valid_eval.csv", index=False)

    merged = df_train_eval.merge(
        df_valid_eval,
        on="rule",
        suffixes=("_train","_valid"),
        how="inner"
    )

    if merged.empty:
        print(f"SID {sid}: no rules survived evaluation")
        return

    # Final balanced — matches your CSV best
    merged["balanced_valid"] = (
        merged["expectancy_valid"] *
        np.sqrt(np.maximum(1, merged["trades_valid"])) *
        np.maximum(0.05, merged["stability_valid"] / np.maximum(0.0001, np.abs(merged["expectancy_valid"])))
    )
    merged = merged.sort_values("balanced_valid", ascending=False)

    pareto = pareto_front(merged)
    pareto.to_csv(outdir / f"sid_{sid}_pareto.csv", index=False)

    sorted_by_trades = merged.sort_values("trades_valid", ascending=False)
    knee = find_knee(sorted_by_trades)
    knee.to_csv(outdir / f"sid_{sid}_knee.csv", index=False)

    # Plot
    plt.figure(figsize=(10,7))
    plt.scatter(merged["trades_valid"], merged["expectancy_valid"], alpha=0.4, label="all rules", s=40)
    plt.scatter(pareto["trades_valid"], pareto["expectancy_valid"], color="red", s=100, label="Pareto front", marker="^")
    if not knee.empty:
        plt.scatter(knee["trades_valid"], knee["expectancy_valid"], color="green", s=180, marker="*", label="Knee point")
    plt.xlabel("Trades (validation set)")
    plt.ylabel("Expectancy (validation set)")
    plt.title(f"SetuPID {sid} — Expectancy vs Trade Count (validation)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(outdir / f"sid_{sid}_frontier.png", dpi=150)
    plt.close()


# ────────────────────────────────────────────────
#  Main
# ────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Rule Miner v3 — ORC + Sqz support")
    parser.add_argument("--input", required=True, help="Input CSV with all trades")
    parser.add_argument("--out", default="ruleminer_v3_output", help="Output directory")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    df[ENTRY_DT_COL] = pd.to_datetime(df[ENTRY_DT_COL], errors="coerce")

    out_root = Path(args.out)
    out_root.mkdir(exist_ok=True)

    for sid in sorted(df[SID_COL].dropna().unique()):
        df_sid = df[df[SID_COL] == sid].copy()
        if len(df_sid) < 60:
            print(f"Skipping SID {sid} — only {len(df_sid)} trades")
            continue
        sid_dir = out_root / f"sid_{int(sid)}"
        sid_dir.mkdir(exist_ok=True)
        print(f"\nProcessing SID {sid}  ({len(df_sid)} trades)")
        process_sid(df_sid, int(sid), sid_dir)

    print("\nDone.")
    print(f"Results saved to: {out_root.resolve()}")
    print("  • Check *_knee.csv for best balanced compromise")
    print("  • Check *_pareto.csv for non-dominated rules")
    print("  • Look at frontier.png for visual trade-off")


if __name__ == "__main__":
    main()