"""
reporting.py — Generate all output files.

Outputs produced in the configured output directory:
    1. setup_baseline_summary.csv
    2. factor_rule_results.csv
    3. branch_comparison_by_setup.csv
    4. recommendations.txt
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from branch_tester import BranchResult
from config import MIN_TRADES, SETUP_LABELS, SETUP_ID_COL, FACTOR_GRID
from factor_engine import sweep_single_factor, sweep_combinations
from metrics import baseline_by_setup
from utils import info, pct, fmt_float


def _ensure_output_dir(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. Setup baseline summary
# ---------------------------------------------------------------------------

def write_setup_baseline(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    baseline = baseline_by_setup(df)
    path = output_dir / "setup_baseline_summary.csv"
    baseline.to_csv(path, index=False)
    info(f"  Wrote {path}")
    return baseline


# ---------------------------------------------------------------------------
# 2. Factor rule results
# ---------------------------------------------------------------------------

def write_factor_rule_results(
    df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    output_dir: Path,
) -> pd.DataFrame:
    """
    Sweep single-factor conditions and 2-factor combinations per SetupID,
    writing results to factor_rule_results.csv.
    """
    rows = []
    baseline_lookup = baseline_df.set_index(SETUP_ID_COL).to_dict("index")

    for sid, df_setup in df.groupby(SETUP_ID_COL):
        sid = int(sid)
        base = baseline_lookup.get(sid, {})
        base_wr = base.get("win_rate", float("nan"))
        base_exp = base.get("expectancy", float("nan"))
        base_med = base.get("median_ret", float("nan"))
        base_n = base.get("trades", 0)

        # Single-factor sweep
        for factor, conditions in FACTOR_GRID.items():
            if factor not in df_setup.columns:
                continue
            for result in sweep_single_factor(df_setup, factor, conditions):
                n = result["trades"]
                row = {
                    "setup_id": sid,
                    "combo_type": "single",
                    "rule": result["rule"],
                    "trades": n,
                    "retention": n / base_n if base_n else float("nan"),
                    "win_rate": result["win_rate"],
                    "wr_lift": result["win_rate"] - base_wr if not (
                        np.isnan(result["win_rate"]) or np.isnan(base_wr)
                    ) else float("nan"),
                    "expectancy": result["expectancy"],
                    "exp_lift": result["expectancy"] - base_exp if not (
                        np.isnan(result["expectancy"]) or np.isnan(base_exp)
                    ) else float("nan"),
                    "median_ret": result["median_ret"],
                    "med_lift": result["median_ret"] - base_med if not (
                        np.isnan(result["median_ret"]) or np.isnan(base_med)
                    ) else float("nan"),
                    "notes": "LOW_N" if n < MIN_TRADES else "",
                }
                rows.append(row)

        # 2-factor combination sweep
        for result in sweep_combinations(df_setup, FACTOR_GRID, n_factors=2):
            n = result["trades"]
            row = {
                "setup_id": sid,
                "combo_type": "2-factor",
                "rule": result["rules"],
                "trades": n,
                "retention": n / base_n if base_n else float("nan"),
                "win_rate": result["win_rate"],
                "wr_lift": result["win_rate"] - base_wr if not (
                    np.isnan(result["win_rate"]) or np.isnan(base_wr)
                ) else float("nan"),
                "expectancy": result["expectancy"],
                "exp_lift": result["expectancy"] - base_exp if not (
                    np.isnan(result["expectancy"]) or np.isnan(base_exp)
                ) else float("nan"),
                "median_ret": result["median_ret"],
                "med_lift": result["median_ret"] - base_med if not (
                    np.isnan(result["median_ret"]) or np.isnan(base_med)
                ) else float("nan"),
                "notes": "LOW_N" if n < MIN_TRADES else "",
            }
            rows.append(row)

    factor_df = pd.DataFrame(rows)
    if not factor_df.empty:
        factor_df = factor_df.sort_values(
            ["setup_id", "exp_lift"], ascending=[True, False]
        ).reset_index(drop=True)

    path = output_dir / "factor_rule_results.csv"
    factor_df.to_csv(path, index=False)
    info(f"  Wrote {path}")
    return factor_df


# ---------------------------------------------------------------------------
# 3. Branch comparison
# ---------------------------------------------------------------------------

def write_branch_comparison(
    branch_results: list[BranchResult],
    output_dir: Path,
) -> pd.DataFrame:
    rows = [r.to_dict() for r in branch_results]
    branch_df = pd.DataFrame(rows)
    if not branch_df.empty:
        branch_df = branch_df.sort_values(
            ["setup_id", "branch_name"]
        ).reset_index(drop=True)

    path = output_dir / "branch_comparison_by_setup.csv"
    branch_df.to_csv(path, index=False)
    info(f"  Wrote {path}")
    return branch_df


# ---------------------------------------------------------------------------
# 4. Recommendations
# ---------------------------------------------------------------------------

def _interpret_branch(branch_results_for_sid: list[BranchResult]) -> str:
    """Produce a short paragraph interpreting the three branches for one SetupID."""
    by_name = {r.branch_name: r for r in branch_results_for_sid}
    a = by_name.get("A_CORE_ONLY")
    b = by_name.get("B_CORE_PLUS_PROFILE")
    c = by_name.get("C_PROFILE_THEN_CORE")

    lines = []

    if a:
        if a.trades < MIN_TRADES:
            lines.append(f"  CORE-only: sample too small ({a.trades} trades) — no reliable conclusion.")
        else:
            lines.append(
                f"  CORE-only: {a.trades} trades, WR={pct(a.win_rate)}, "
                f"Exp={fmt_float(a.expectancy, 4)}, "
                f"WR-lift={fmt_float(a.wr_lift, 4)}"
            )

    if b and a:
        if b.trades < MIN_TRADES:
            lines.append(f"  CORE+PROFILE: sample too small ({b.trades} trades).")
        else:
            wr_delta = b.win_rate - a.win_rate if not (
                np.isnan(b.win_rate) or np.isnan(a.win_rate)
            ) else float("nan")
            exp_delta = b.expectancy - a.expectancy if not (
                np.isnan(b.expectancy) or np.isnan(a.expectancy)
            ) else float("nan")
            additive = (not np.isnan(wr_delta)) and wr_delta > 0.02
            noise = (not np.isnan(wr_delta)) and abs(wr_delta) < 0.01
            lines.append(
                f"  CORE+PROFILE vs CORE-only: Δ WR={fmt_float(wr_delta, 4)}, "
                f"Δ Exp={fmt_float(exp_delta, 4)}, retention={pct(b.retention)} — "
                + (
                    "PROFILE appears ADDITIVE."
                    if additive
                    else ("PROFILE appears as NOISE (minimal delta)." if noise else "PROFILE is EXCLUSION filter.")
                )
            )

    if c:
        if c.trades < MIN_TRADES:
            lines.append(f"  PROFILE→CORE: sample too small ({c.trades} trades).")
        else:
            lines.append(
                f"  PROFILE→CORE: {c.trades} trades, WR={pct(c.win_rate)}, "
                f"Exp={fmt_float(c.expectancy, 4)}"
            )

    return "\n".join(lines) if lines else "  (no data)"


def write_recommendations(
    df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    factor_df: pd.DataFrame,
    branch_results: list[BranchResult],
    output_dir: Path,
) -> None:
    """Write the plain-English recommendations.txt file."""
    lines = []
    lines.append("=" * 70)
    lines.append("ISOLATED SETUP RESEARCH — RECOMMENDATIONS")
    lines.append("=" * 70)
    lines.append("")

    # Group branch results by setup
    from collections import defaultdict
    by_sid: dict[int, list[BranchResult]] = defaultdict(list)
    for r in branch_results:
        by_sid[r.setup_id].append(r)

    baseline_lookup = baseline_df.set_index(SETUP_ID_COL).to_dict("index") if not baseline_df.empty else {}

    for sid in sorted(by_sid.keys()):
        label = SETUP_LABELS.get(sid, f"Setup {sid}")
        base = baseline_lookup.get(sid, {})
        base_n = base.get("trades", 0)
        base_wr = base.get("win_rate", float("nan"))
        base_exp = base.get("expectancy", float("nan"))

        lines.append(f"SetupID {sid}: {label}")
        lines.append("-" * 60)
        lines.append(
            f"  Baseline: {base_n} trades | WR={pct(base_wr)} | Exp={fmt_float(base_exp, 4)}"
        )
        lines.append("")
        lines.append("  Branch comparison:")
        lines.append(_interpret_branch(by_sid[sid]))
        lines.append("")

        # Best single-factor rule by expectancy lift
        if not factor_df.empty:
            sid_factors = factor_df[
                (factor_df["setup_id"] == sid)
                & (factor_df["combo_type"] == "single")
                & (factor_df["notes"] != "LOW_N")
                & (~factor_df["exp_lift"].isna())
            ]
            if not sid_factors.empty:
                top3 = sid_factors.nlargest(3, "exp_lift")[["rule", "trades", "win_rate", "exp_lift"]]
                lines.append("  Top single-factor CORE candidates (by expectancy lift):")
                for _, row in top3.iterrows():
                    lines.append(
                        f"    {row['rule']:<35}  trades={int(row['trades']):<4}  "
                        f"WR={pct(row['win_rate']):<8}  Δ Exp={fmt_float(row['exp_lift'], 4)}"
                    )
                lines.append("")

            # Best 2-factor rule
            sid_2f = factor_df[
                (factor_df["setup_id"] == sid)
                & (factor_df["combo_type"] == "2-factor")
                & (factor_df["notes"] != "LOW_N")
                & (~factor_df["exp_lift"].isna())
            ]
            if not sid_2f.empty:
                best = sid_2f.nlargest(1, "exp_lift").iloc[0]
                lines.append(
                    f"  Best 2-factor CORE candidate: {best['rule']}"
                )
                lines.append(
                    f"    trades={int(best['trades'])}  WR={pct(best['win_rate'])}  "
                    f"Δ Exp={fmt_float(best['exp_lift'], 4)}"
                )
                lines.append("")

        # Assessment
        core_r = next((r for r in by_sid[sid] if r.branch_name == "A_CORE_ONLY"), None)
        profile_b = next((r for r in by_sid[sid] if r.branch_name == "B_CORE_PLUS_PROFILE"), None)

        assessment_lines = []

        if core_r and core_r.trades >= MIN_TRADES:
            if core_r.wr_lift > 0.05:
                assessment_lines.append(
                    f"  ✓ SetupID {sid} has strong CORE-only edge "
                    f"(WR lift = {pct(core_r.wr_lift)} vs baseline)."
                )
            elif core_r.wr_lift > 0:
                assessment_lines.append(
                    f"  ~ SetupID {sid} shows modest CORE-only edge "
                    f"(WR lift = {pct(core_r.wr_lift)})."
                )
            else:
                assessment_lines.append(
                    f"  ✗ SetupID {sid} CORE rules do not improve WR vs baseline."
                )
        else:
            assessment_lines.append(
                f"  ? SetupID {sid}: insufficient trades after CORE filter — review rules."
            )

        if profile_b and core_r and profile_b.trades >= MIN_TRADES and core_r.trades >= MIN_TRADES:
            wr_add = profile_b.win_rate - core_r.win_rate
            if wr_add > 0.03:
                assessment_lines.append(
                    f"  ✓ PROFILE adds material value for SetupID {sid} "
                    f"(additional WR lift = {pct(wr_add)})."
                )
            elif wr_add < -0.03:
                assessment_lines.append(
                    f"  ✗ PROFILE hurts edge for SetupID {sid} — consider dropping PROFILE filter."
                )
            else:
                assessment_lines.append(
                    f"  ~ PROFILE has minimal incremental effect for SetupID {sid}."
                )

        lines.extend(assessment_lines)
        lines.append("")

    # Pine OR logic suggestions
    lines.append("=" * 70)
    lines.append("CANDIDATE PINE OR LOGIC BRANCHES")
    lines.append("=" * 70)
    lines.append("")
    lines.append(
        "The following SetupID / branch combinations showed positive expectancy "
        "lift and adequate sample size. These are candidates for later OR-combination "
        "in Pine:"
    )
    lines.append("")

    for sid in sorted(by_sid.keys()):
        core_r = next((r for r in by_sid[sid] if r.branch_name == "A_CORE_ONLY"), None)
        if core_r and core_r.trades >= MIN_TRADES and not np.isnan(core_r.exp_lift) and core_r.exp_lift > 0:
            lines.append(
                f"  Setup {sid} CORE-only: {core_r.rules_used}"
            )
            lines.append(
                f"    trades={core_r.trades}  WR={pct(core_r.win_rate)}  "
                f"Exp={fmt_float(core_r.expectancy, 4)}  Δ Exp={fmt_float(core_r.exp_lift, 4)}"
            )
            lines.append("")

    lines.append("")
    lines.append("NOTES")
    lines.append("-" * 60)
    lines.append(f"  Minimum trades threshold used: {MIN_TRADES}")
    lines.append(
        "  Rows flagged LOW_N were excluded from recommendations "
        "but appear in factor_rule_results.csv for inspection."
    )
    lines.append(
        "  All thresholds are configurable in config.py — "
        "edit FACTOR_GRID, DEFAULT_CORE_RULES, DEFAULT_PROFILE_RULES."
    )

    path = output_dir / "recommendations.txt"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    info(f"  Wrote {path}")


# ---------------------------------------------------------------------------
# Master reporting entry point
# ---------------------------------------------------------------------------

def run_reporting(
    df: pd.DataFrame,
    branch_results: list[BranchResult],
    output_dir: Path,
) -> None:
    """Run all report writers in sequence."""
    _ensure_output_dir(output_dir)
    info("Generating reports...")

    baseline_df = write_setup_baseline(df, output_dir)
    factor_df = write_factor_rule_results(df, baseline_df, output_dir)
    write_branch_comparison(branch_results, output_dir)
    write_recommendations(df, baseline_df, factor_df, branch_results, output_dir)

    info(f"All outputs written to: {output_dir}")
