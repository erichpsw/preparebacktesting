"""
factor_engine.py — Rule DSL and evaluation.

A *rule* is a 3-tuple: (factor_column, operator, threshold)
    ("RS",  "gte",     90)
    ("ORC", "between", (0.25, 0.75))
    ("RMV", "rising",  None)
    ("ADR", "lte",     6.0)

A *ruleset* is a list of rules; all rules are AND-combined.

Public API:
    apply_rule(df, factor, operator, threshold)  -> boolean Series
    apply_ruleset(df, rules)                     -> boolean Series
    evaluate_ruleset(df, rules)                  -> (filtered_df, label_str)
    sweep_single_factor(df, factor, grid)        -> list[dict]  (one per condition)
    sweep_combinations(df, factors, grid, n)     -> list[dict]  (n-factor combos)
"""

from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd

from config import NUMERIC_FACTOR_COLS
from metrics import compute_metrics
from utils import branch_label, rule_label, warn


def apply_rule(
    df: pd.DataFrame,
    factor: str,
    operator: str,
    threshold: Any,
) -> pd.Series:
    """
    Return a boolean Series indicating which rows satisfy one rule condition.

    Missing values in the factor column are treated as False (rule not satisfied).
    """
    if factor not in df.columns:
        warn(f"Factor column '{factor}' not found — rule will match zero rows.")
        return pd.Series(False, index=df.index)

    col = df[factor]

    if operator == "gte":
        mask = col >= threshold
    elif operator == "lte":
        mask = col <= threshold
    elif operator == "between":
        lo, hi = threshold
        mask = (col >= lo) & (col <= hi)
    elif operator == "rising":
        # True where the value increased vs. the previous row.
        # Useful only when rows are time-ordered; otherwise treat as >=0 sentinel.
        mask = col.diff() > 0
    else:
        warn(f"Unknown operator '{operator}' — rule will match zero rows.")
        mask = pd.Series(False, index=df.index)

    # NaN in the factor column → False (rule not met)
    return mask.fillna(False)


def apply_ruleset(df: pd.DataFrame, rules: list[tuple]) -> pd.Series:
    """
    AND-combine all rules in the list.

    An empty ruleset returns all-True (no filtering).
    """
    if not rules:
        return pd.Series(True, index=df.index)

    combined = pd.Series(True, index=df.index)
    for factor, operator, threshold in rules:
        combined &= apply_rule(df, factor, operator, threshold)
    return combined


def evaluate_ruleset(
    df: pd.DataFrame,
    rules: list[tuple],
) -> tuple[pd.DataFrame, str]:
    """
    Apply a ruleset and return (filtered_dataframe, label_string).
    """
    mask = apply_ruleset(df, rules)
    filtered = df[mask].copy()
    label = branch_label(rules) if rules else "(all)"
    return filtered, label


# ---------------------------------------------------------------------------
# Single-factor sweep
# ---------------------------------------------------------------------------

def sweep_single_factor(
    df: pd.DataFrame,
    factor: str,
    conditions: list[tuple],
) -> list[dict]:
    """
    Test each (operator, threshold) condition for one factor column.

    Parameters
    ----------
    df:
        Data subset for a specific SetupID.
    factor:
        Column name.
    conditions:
        List of (operator, threshold) tuples from the factor grid.

    Returns
    -------
    list[dict]
        One dict per condition with metrics + rule metadata.
    """
    results = []
    for operator, threshold in conditions:
        mask = apply_rule(df, factor, operator, threshold)
        filtered = df[mask]
        m = compute_metrics(filtered)
        m["factor"] = factor
        m["operator"] = operator
        m["threshold"] = str(threshold)
        m["rule"] = rule_label(factor, operator, threshold)
        results.append(m)
    return results


# ---------------------------------------------------------------------------
# Multi-factor combination sweep
# ---------------------------------------------------------------------------

def sweep_combinations(
    df: pd.DataFrame,
    factor_grid: dict[str, list[tuple]],
    n_factors: int,
) -> list[dict]:
    """
    Test all n_factors combinations of (factor, operator, threshold) drawn
    from factor_grid and return metric dicts for each combination.

    Parameters
    ----------
    df:
        Data subset (already filtered to a single SetupID or branch).
    factor_grid:
        Dict mapping factor -> list[(operator, threshold)].
    n_factors:
        Number of factors to combine (2 or 3 recommended).

    Returns
    -------
    list[dict]
        One dict per tested combination with metrics + label.
    """
    # Flatten grid: one entry per (factor, operator, threshold)
    candidates: list[tuple] = []
    for factor, conditions in factor_grid.items():
        if factor not in df.columns:
            continue
        for operator, threshold in conditions:
            candidates.append((factor, operator, threshold))

    results = []
    for combo in combinations(candidates, n_factors):
        # Skip duplicates involving the same factor twice
        factors_used = [c[0] for c in combo]
        if len(set(factors_used)) < n_factors:
            continue

        mask = apply_ruleset(df, list(combo))
        filtered = df[mask]
        m = compute_metrics(filtered)
        m["rules"] = branch_label(list(combo))
        results.append(m)

    return results
