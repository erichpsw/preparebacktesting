"""
utils.py — Lightweight shared helpers.
"""

from __future__ import annotations

import sys
import textwrap
from typing import Any


def die(message: str, code: int = 1) -> None:
    """Print an error message to stderr and exit."""
    print(f"[ERROR] {message}", file=sys.stderr)
    sys.exit(code)


def warn(message: str) -> None:
    """Print a warning to stderr without exiting."""
    print(f"[WARN]  {message}", file=sys.stderr)


def info(message: str) -> None:
    """Print an informational message to stdout."""
    print(f"[INFO]  {message}")


def pct(value: float) -> str:
    """Format a fraction as a percentage string, e.g. 0.643 -> '64.3%'."""
    return f"{value * 100:.1f}%"


def fmt_float(value: float, decimals: int = 4) -> str:
    return f"{value:.{decimals}f}"


def indent(text: str, spaces: int = 4) -> str:
    return textwrap.indent(text, " " * spaces)


def safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Return numerator / denominator, or default if denominator is zero."""
    if denominator == 0:
        return default
    return numerator / denominator


def rule_label(factor: str, operator: str, threshold: Any) -> str:
    """
    Return a compact human-readable string for a single rule condition.

    Examples:
        rule_label("RS", "gte", 90)          -> "RS>=90"
        rule_label("ORC", "between", (0.25, 0.75)) -> "ORC[0.25,0.75]"
        rule_label("RMV", "rising", None)    -> "RMV.rising"
        rule_label("ADR", "lte", 6.0)        -> "ADR<=6.0"
    """
    if operator == "gte":
        return f"{factor}>={threshold}"
    if operator == "lte":
        return f"{factor}<={threshold}"
    if operator == "between":
        lo, hi = threshold
        return f"{factor}[{lo},{hi}]"
    if operator == "rising":
        return f"{factor}.rising"
    return f"{factor}({operator},{threshold})"


def branch_label(rules: list[tuple]) -> str:
    """Join individual rule labels into a compact branch description."""
    return " & ".join(rule_label(f, op, th) for f, op, th in rules)
