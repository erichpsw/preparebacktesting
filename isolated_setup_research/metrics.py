"""
metrics.py — Core statistical calculations.

Functions return plain dicts or DataFrames; no side-effects.

Metric definitions:
    trades      — number of rows in the subset
    win_rate    — fraction of rows where is_win == True
    expectancy  — average return_pct across all rows (wins AND losses)
    median_ret  — median return_pct
    avg_win     — average return_pct on winning rows only
    avg_loss    — average return_pct on losing rows only (typically negative)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import RETURN_COL, SETUP_ID_COL, WIN_COL, SETUP_LABELS, MIN_TRADES
from utils import safe_div


def compute_metrics(subset: pd.DataFrame) -> dict:
    """
    Compute summary statistics for an arbitrary subset of the data.

    Returns a dict with keys:
        trades, win_rate, expectancy, median_ret, avg_win, avg_loss
    """
    n = len(subset)
    if n == 0:
        return {
            "trades": 0,
            "win_rate": float("nan"),
            "expectancy": float("nan"),
            "median_ret": float("nan"),
            "avg_win": float("nan"),
            "avg_loss": float("nan"),
        }

    wins = subset[subset[WIN_COL].astype(bool)]
    losses = subset[~subset[WIN_COL].astype(bool)]

    return {
        "trades": n,
        "win_rate": safe_div(len(wins), n),
        "expectancy": subset[RETURN_COL].mean(),
        "median_ret": subset[RETURN_COL].median(),
        "avg_win": wins[RETURN_COL].mean() if len(wins) else float("nan"),
        "avg_loss": losses[RETURN_COL].mean() if len(losses) else float("nan"),
    }


def baseline_by_setup(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-SetupID baseline metrics across the whole population.

    Returns a DataFrame indexed by SetupID with columns:
        setup_label, trades, win_rate, expectancy, median_ret, avg_win, avg_loss
    """
    rows = []
    for sid, group in df.groupby(SETUP_ID_COL):
        m = compute_metrics(group)
        m[SETUP_ID_COL] = int(sid)
        m["setup_label"] = SETUP_LABELS.get(int(sid), f"Setup {sid}")
        rows.append(m)

    if not rows:
        return pd.DataFrame()

    cols = [SETUP_ID_COL, "setup_label", "trades", "win_rate",
            "expectancy", "median_ret", "avg_win", "avg_loss"]
    return pd.DataFrame(rows)[cols].sort_values(SETUP_ID_COL).reset_index(drop=True)


def lift(subset_metrics: dict, baseline_metrics: dict) -> dict:
    """
    Compute lift of a filtered subset vs. its setup baseline.

    Returns dict with:
        retention     — fraction of baseline trades retained
        wr_lift       — absolute win-rate improvement (subset - baseline)
        exp_lift      — absolute expectancy improvement
        med_lift      — absolute median-return improvement
    """
    base_n = baseline_metrics.get("trades", 0)
    sub_n = subset_metrics.get("trades", 0)

    def _diff(key: str) -> float:
        sv = subset_metrics.get(key, float("nan"))
        bv = baseline_metrics.get(key, float("nan"))
        if np.isnan(sv) or np.isnan(bv):
            return float("nan")
        return sv - bv

    return {
        "retention": safe_div(sub_n, base_n) if base_n else float("nan"),
        "wr_lift": _diff("win_rate"),
        "exp_lift": _diff("expectancy"),
        "med_lift": _diff("median_ret"),
    }


def flag_small_sample(metrics: dict, min_trades: int = MIN_TRADES) -> str:
    """Return 'LOW_N' if trades < min_trades, else ''."""
    return "LOW_N" if metrics.get("trades", 0) < min_trades else ""
