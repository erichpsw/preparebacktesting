"""
branch_tester.py — Compare research branches per SetupID.

Three framing strategies are compared for each SetupID:

  A. CORE-only
     Apply only CORE rules. Establishes the pure setup-specific edge.

  B. CORE + PROFILE exclusion
     Start with CORE, then apply PROFILE rules as additional AND filters.
     Goal: see if PROFILE narrows the set to higher-quality trades.

  C. PROFILE → CORE
     Apply PROFILE first, then CORE within that subset.
     Goal: test whether the environment filter (PROFILE) changes the CORE's
     efficacy. Mathematically equivalent to B when all rules are AND'd, but
     produces a separate labelled output because the *framing* matters for
     Pine logic design.

Each strategy returns a BranchResult dataclass containing:
    setup_id, branch_name, rules_used, metrics, lift_vs_baseline, notes
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from config import (
    DEFAULT_CORE_RULES,
    DEFAULT_PROFILE_RULES,
    SETUP_ID_COL,
    MIN_TRADES,
)
from factor_engine import evaluate_ruleset
from metrics import compute_metrics, lift, flag_small_sample
from utils import branch_label


@dataclass
class BranchResult:
    setup_id: int
    branch_name: str
    rules_used: str
    trades: int
    win_rate: float
    expectancy: float
    median_ret: float
    avg_win: float
    avg_loss: float
    retention: float
    wr_lift: float
    exp_lift: float
    med_lift: float
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "setup_id": self.setup_id,
            "branch_name": self.branch_name,
            "rules_used": self.rules_used,
            "trades": self.trades,
            "win_rate": self.win_rate,
            "expectancy": self.expectancy,
            "median_ret": self.median_ret,
            "avg_win": self.avg_win,
            "avg_loss": self.avg_loss,
            "retention": self.retention,
            "wr_lift": self.wr_lift,
            "exp_lift": self.exp_lift,
            "med_lift": self.med_lift,
            "notes": self.notes,
        }


def _make_branch(
    setup_id: int,
    branch_name: str,
    df_setup: pd.DataFrame,
    rules: list[tuple],
    baseline_metrics: dict,
) -> BranchResult:
    """Evaluate rules on df_setup and assemble a BranchResult."""
    filtered, label = evaluate_ruleset(df_setup, rules)
    m = compute_metrics(filtered)
    lft = lift(m, baseline_metrics)
    note = flag_small_sample(m, MIN_TRADES)

    return BranchResult(
        setup_id=setup_id,
        branch_name=branch_name,
        rules_used=label,
        trades=m["trades"],
        win_rate=m["win_rate"],
        expectancy=m["expectancy"],
        median_ret=m["median_ret"],
        avg_win=m["avg_win"],
        avg_loss=m["avg_loss"],
        retention=lft["retention"],
        wr_lift=lft["wr_lift"],
        exp_lift=lft["exp_lift"],
        med_lift=lft["med_lift"],
        notes=note,
    )


def run_branch_comparison(
    df: pd.DataFrame,
    core_rules_override: dict[int, list[tuple]] | None = None,
    profile_rules_override: dict[int, list[tuple]] | None = None,
) -> list[BranchResult]:
    """
    For every SetupID present in df, run three branches and return results.

    Parameters
    ----------
    df:
        Full cleaned dataframe (all SetupIDs).
    core_rules_override:
        Optional dict of SetupID -> rules to replace DEFAULT_CORE_RULES.
    profile_rules_override:
        Optional dict of SetupID -> rules to replace DEFAULT_PROFILE_RULES.

    Returns
    -------
    list[BranchResult]
        One result per (SetupID, branch_name) combination.
    """
    core_map = core_rules_override if core_rules_override is not None else DEFAULT_CORE_RULES
    profile_map = profile_rules_override if profile_rules_override is not None else DEFAULT_PROFILE_RULES

    results: list[BranchResult] = []

    for sid, df_setup in df.groupby(SETUP_ID_COL):
        sid = int(sid)
        baseline_metrics = compute_metrics(df_setup)

        core_rules = core_map.get(sid, [])
        profile_rules = profile_map.get(sid, [])
        combined_rules = core_rules + profile_rules

        # A — CORE only
        results.append(
            _make_branch(sid, "A_CORE_ONLY", df_setup, core_rules, baseline_metrics)
        )

        # B — CORE + PROFILE exclusion (CORE AND PROFILE)
        results.append(
            _make_branch(sid, "B_CORE_PLUS_PROFILE", df_setup, combined_rules, baseline_metrics)
        )

        # C — PROFILE first → CORE (AND combination but framed differently)
        #     We first filter to PROFILE rows, then apply CORE within that subset.
        profile_filtered, _ = evaluate_ruleset(df_setup, profile_rules)
        if len(profile_filtered) > 0:
            core_within_profile_filtered, core_label = evaluate_ruleset(profile_filtered, core_rules)
            m = compute_metrics(core_within_profile_filtered)
            lft = lift(m, baseline_metrics)
            note = flag_small_sample(m, MIN_TRADES)

            profile_label = branch_label(profile_rules) if profile_rules else "(all)"
            full_label = f"PROFILE({profile_label}) → CORE({core_label})"

            results.append(
                BranchResult(
                    setup_id=sid,
                    branch_name="C_PROFILE_THEN_CORE",
                    rules_used=full_label,
                    trades=m["trades"],
                    win_rate=m["win_rate"],
                    expectancy=m["expectancy"],
                    median_ret=m["median_ret"],
                    avg_win=m["avg_win"],
                    avg_loss=m["avg_loss"],
                    retention=lft["retention"],
                    wr_lift=lft["wr_lift"],
                    exp_lift=lft["exp_lift"],
                    med_lift=lft["med_lift"],
                    notes=note,
                )
            )
        else:
            # No rows pass the profile filter — record empty result
            results.append(
                BranchResult(
                    setup_id=sid,
                    branch_name="C_PROFILE_THEN_CORE",
                    rules_used="PROFILE filter removed all rows",
                    trades=0,
                    win_rate=float("nan"),
                    expectancy=float("nan"),
                    median_ret=float("nan"),
                    avg_win=float("nan"),
                    avg_loss=float("nan"),
                    retention=0.0,
                    wr_lift=float("nan"),
                    exp_lift=float("nan"),
                    med_lift=float("nan"),
                    notes="LOW_N",
                )
            )

    return results
