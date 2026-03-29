"""
config.py — Central configuration for the isolated setup research framework.

Edit this file to adjust:
- Input data path
- Required / numeric columns
- Factor candidate grids
- CORE and PROFILE rule assignments per SetupID
- Minimum sample-size guard
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# I/O paths
# ---------------------------------------------------------------------------

# Input CSV — can be campaigns_clean.csv or any similarly structured file.
# Override via CLI: run_research.py --input <path>
DEFAULT_INPUT_CSV: Path = Path(__file__).parent.parent / "campaigns_clean.csv"

OUTPUT_DIR: Path = Path(__file__).parent / "output"

# ---------------------------------------------------------------------------
# Column names  (adjust if your CSV uses different headers)
# ---------------------------------------------------------------------------

SETUP_ID_COL: str = "sid"          # SetupID column in the CSV
RETURN_COL: str = "return_pct"     # Primary outcome: fractional return per trade
WIN_COL: str = "is_win"            # Boolean win flag

# Minimum trades required to produce a meaningful comparison
MIN_TRADES: int = 5

# ---------------------------------------------------------------------------
# Numeric factor columns to coerce and use in rule evaluation
# ---------------------------------------------------------------------------

NUMERIC_FACTOR_COLS: list[str] = [
    "RS",
    "CMP",
    "TR",
    "VQS",
    "ADR",
    "DCR",
    "RMV",
    "ORC",
    "SS",
    "DDV",
    "GRP",
]

# ---------------------------------------------------------------------------
# Setup labels (for readable output)
# ---------------------------------------------------------------------------

SETUP_LABELS: dict[int, str] = {
    1: "Reclaim AVWAP (early base)",
    2: "Inside base / lower zone, above AVWAP",
    3: "Inside base / middle zone",
    4: "Inside base / upper zone",
    5: "Post-breakout tight continuation",
    6: "Post-breakout structured ACP continuation",
}

# ---------------------------------------------------------------------------
# Factor candidate grids
#
# Each key is a factor column name.
# Value is a list of (operator, threshold) tuples used for single-factor sweep.
#   operator: "gte" | "lte" | "between" | "rising"
#   For "between", threshold is a (low, high) tuple.
#   For "gte" / "lte", threshold is a scalar.
#   For "rising", threshold is ignored (uses prev-row comparison).
# ---------------------------------------------------------------------------

FACTOR_GRID: dict[str, list[tuple]] = {
    "RS": [
        ("gte", 85),
        ("gte", 90),
        ("gte", 95),
    ],
    "CMP": [
        ("between", (50, 70)),
        ("between", (65, 85)),
        ("between", (70, 90)),
        ("gte", 75),
    ],
    "TR": [
        ("gte", 85),
        ("gte", 90),
        ("gte", 95),
    ],
    "VQS": [
        ("gte", 50),
        ("gte", 60),
        ("gte", 70),
    ],
    "ADR": [
        ("between", (2.0, 5.0)),
        ("between", (2.5, 6.0)),
        ("lte", 6.0),
    ],
    "DCR": [
        ("gte", 45),
        ("gte", 60),
        ("gte", 75),
    ],
    "RMV": [
        ("gte", 0),    # placeholder: positive RMV treated as rising
        ("gte", 3),
        ("gte", 8),
    ],
    "ORC": [
        ("between", (0.25, 0.50)),
        ("between", (0.30, 0.60)),
        ("between", (0.25, 0.75)),
    ],
    "GRP": [
        ("gte", 1),    # Improving or Strong groups
        ("gte", 2),    # Strong group only
    ],
}

# ---------------------------------------------------------------------------
# CORE and PROFILE rule assignments per SetupID
#
# Keys are SetupID integers.
# "core" and "profile" each list (factor, operator, threshold) triples.
# These are defaults; you can override per-run via custom branch definitions.
# ---------------------------------------------------------------------------

DEFAULT_CORE_RULES: dict[int, list[tuple]] = {
    1: [("RS", "gte", 85), ("TR", "gte", 85), ("CMP", "between", (65, 85))],
    2: [("RS", "gte", 85), ("TR", "gte", 85), ("CMP", "between", (50, 70))],
    3: [("RS", "gte", 85), ("TR", "gte", 85), ("CMP", "between", (50, 75))],
    4: [("RS", "gte", 85), ("TR", "gte", 85), ("CMP", "between", (65, 85))],
    5: [("RS", "gte", 90), ("TR", "gte", 90), ("VQS", "gte", 60)],
    6: [("RS", "gte", 90), ("TR", "gte", 85), ("VQS", "gte", 55)],
}

DEFAULT_PROFILE_RULES: dict[int, list[tuple]] = {
    1: [("ORC", "between", (0.25, 0.75)), ("GRP", "gte", 1)],
    2: [("ORC", "between", (0.25, 0.75)), ("DCR", "gte", 45)],
    3: [("ORC", "between", (0.25, 0.75)), ("DCR", "gte", 45)],
    4: [("ORC", "between", (0.25, 0.75)), ("DCR", "gte", 60)],
    5: [("ADR", "between", (2.0, 6.0)), ("DCR", "gte", 60)],
    6: [("ADR", "between", (2.0, 6.0)), ("DCR", "gte", 45)],
}
