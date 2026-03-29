# Isolated Setup Research Framework

A clean, self-contained Python research framework for testing hypothesis-driven
trading setup rule structures using real executed trade data.

---

## What This Does

This framework answers four research questions:

1. **Does SetupID already define the regime?**  
   Baseline metrics per SetupID reveal whether each setup has a distinct
   statistical profile before any factor filtering is applied.

2. **What are the likely CORE conditions by setup?**  
   Single-factor and 2-factor sweeps identify which factor conditions best
   improve win rate and expectancy within each setup's population.

3. **Does PROFILE materially help?**  
   Three branch strategies are compared per SetupID:
   - **A — CORE-only**: minimum setup-specific conditions
   - **B — CORE + PROFILE**: CORE rules AND-combined with contextual PROFILE filters
   - **C — PROFILE → CORE**: PROFILE filter applied first, then CORE within that subset

4. **Which edges could later be combined with OR logic in Pine?**  
   The recommendations file lists which setup/branch combinations showed
   positive expectancy lift with adequate sample size.

---

## Expected Input Format

A CSV file with at least these columns:

| Column       | Type    | Description                          |
|--------------|---------|--------------------------------------|
| `sid`        | int     | Setup identifier (1–6)               |
| `return_pct` | float   | Fractional return per trade, e.g. 0.045 |
| `is_win`     | bool    | True / False win flag                |
| `RS`         | float   | Relative Strength score              |
| `CMP`        | float   | CMP percentile score                 |
| `TR`         | float   | Trend score                          |
| `VQS`        | float   | Volume Quality Score                 |
| `ADR`        | float   | Average Daily Range                  |
| `DCR`        | float   | DCR percentile                       |
| `RMV`        | float   | Relative Market Volume               |
| `ORC`        | float   | ORC ratio                            |
| `GRP`        | int     | Group classification (-1, 1, 2)      |

Extra columns are ignored. Missing factor values are coerced to NaN and
excluded from rule evaluation without dropping the entire row.

The default input path is `../campaigns_clean.csv` (relative to this folder).
Override it with `--input`.

---

## How to Run

```bash
# From inside the isolated_setup_research/ directory:
cd isolated_setup_research

# Install dependencies (if not already installed)
pip install pandas numpy

# Run with default input
python run_research.py

# Run with a custom input file
python run_research.py --input ../campaigns_clean.csv

# Run with custom output directory
python run_research.py --input ../campaigns_clean.csv --output output/run1
```

---

## Output Files

All outputs are written to `output/` (or your `--output` directory).

### `setup_baseline_summary.csv`
Per-SetupID baseline metrics computed over the full population (no filters).
Columns: `sid`, `setup_label`, `trades`, `win_rate`, `expectancy`,
`median_ret`, `avg_win`, `avg_loss`.

### `factor_rule_results.csv`
Results of the single-factor and 2-factor condition sweeps for each SetupID.
Includes `trades`, `retention`, `win_rate`, `wr_lift`, `expectancy`,
`exp_lift`, `median_ret`, `med_lift`, and a `notes` column (`LOW_N` when
fewer than `MIN_TRADES` trades matched).

Use this file to:
- Identify which individual factor conditions best improve expectancy.
- Compare 2-factor combinations across setups.

### `branch_comparison_by_setup.csv`
Three-branch comparison per SetupID (A, B, C) with all metrics and lift
columns relative to the setup baseline.

### `recommendations.txt`
Human-readable plain-English analysis covering:
- Which SetupIDs have strong CORE-only edge
- Whether PROFILE is additive, exclusion, or noise
- Which SetupIDs have insufficient sample size
- Candidate rule branches for Pine OR-logic

---

## Where to Edit Thresholds and Rules

Everything is configured in `config.py`:

| Config Variable        | Purpose                                          |
|------------------------|--------------------------------------------------|
| `DEFAULT_INPUT_CSV`    | Default input file path                          |
| `OUTPUT_DIR`           | Default output directory                         |
| `SETUP_ID_COL`         | Name of the SetupID column in your CSV           |
| `RETURN_COL`           | Name of the return column in your CSV            |
| `WIN_COL`              | Name of the win/loss flag column                 |
| `MIN_TRADES`           | Minimum trades to report a result as reliable    |
| `NUMERIC_FACTOR_COLS`  | Columns to coerce to float during loading        |
| `SETUP_LABELS`         | Human-readable setup descriptions                |
| `FACTOR_GRID`          | Factor conditions swept in the single/combo scan |
| `DEFAULT_CORE_RULES`   | CORE rule set per SetupID (list of 3-tuples)     |
| `DEFAULT_PROFILE_RULES`| PROFILE rule set per SetupID (list of 3-tuples)  |

### Rule format
A rule is a 3-tuple: `(factor_column, operator, threshold)`

```python
("RS",  "gte",     90)           # RS >= 90
("ORC", "between", (0.25, 0.75)) # 0.25 <= ORC <= 0.75
("ADR", "lte",     6.0)          # ADR <= 6.0
("RMV", "rising",  None)         # RMV increased vs previous row
```

To add a new factor condition to the sweep, add an entry to `FACTOR_GRID`.
To update which conditions define CORE or PROFILE for a setup, edit
`DEFAULT_CORE_RULES` or `DEFAULT_PROFILE_RULES`.

---

## Project Structure

```
isolated_setup_research/
├── config.py          # All thresholds, grids, rule assignments
├── run_research.py    # CLI entry point
├── data_loader.py     # CSV loading, normalisation, validation
├── metrics.py         # win_rate, expectancy, median_ret, lift
├── factor_engine.py   # Rule DSL and single/combo sweeps
├── branch_tester.py   # CORE-only / CORE+PROFILE / PROFILE→CORE
├── reporting.py       # All CSV and text output generation
├── utils.py           # Shared formatting and helper functions
├── README.md          # This file
└── output/            # Generated output files (git-ignored)
```

---

## Dependencies

- Python 3.10+
- `pandas`
- `numpy`

No other packages required.

---

## Design Notes

- **Isolated**: no imports from or references to other scripts in this repo.
- **Research-first**: all logic is explicit; no black-box optimisation.
- **Extensible**: add factors to `FACTOR_GRID`, adjust thresholds in `config.py`,
  or define entirely custom branch logic in `branch_tester.py`.
- **Small sample guard**: any result with fewer than `MIN_TRADES` trades is
  flagged `LOW_N` in outputs and excluded from recommendations.
