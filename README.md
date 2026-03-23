# PrepareBacktesting Pipeline

## Overview
This repository contains a staged research-to-deployment pipeline for rule discovery and translation.

Pipeline intent by stage:
1. Stage 1: Clean raw campaign/trade exports into a consistent research table.
2. Stage 2: Discover winner-profile single-factor seeds.
3. Stage 3: Refine and evaluate factor combinations.
4. Stage 4: Validate robustness/stability.
5. Stage 5: Package approved outputs for deployment.
6. Stage 6 (optional): Translate Stage 5 packaging into Pine-ready artifacts.

Stage 6 is a translation layer, not a research stage.

## Environment
- OS: Linux (dev container)
- Python: 3.10+

Install core dependencies:

```bash
pip install pandas numpy
```

## Expected Input
Primary dataset used through research stages:

- `campaigns_clean.csv`

python stage_1_campaign_cleaner.py --input merged.csv --output campaigns_clean.csv

Expected fields include SetupID/SID, `return_pct`, and factor columns (for example `RS`, `CMP`, `TR`, `VQS`, `SQZ`; optional `RMV`, `DCR`, `ORC`).

## Stage Scripts
- `stage_1_campaign_cleaner.py`
- `stage_2_winner_profile_engine.py`
- `stage_3_factor_rule_engine.py`
- `stage_4_rule_stability_engine.py`
- `stage_5_profile_exporter.py`
- `stage_6_pine_integration_engine.py` (optional)

## Quick Start

### 1) Stage 2 Winner Profiling
```bash
python stage_2_winner_profile_engine.py \
  --input campaigns_clean.csv \
  --output-dir Stage_2_Report \
  --discovery-threshold 5.0
```

Primary outputs:
- `Stage_2_Report/baseline_by_sid.csv`
- `Stage_2_Report/winner_distribution_by_sid.csv`
- `Stage_2_Report/winner_single_rule_validation_by_sid.csv`

### 2) Stage 3 Factor Rule Refinement
```bash
python stage_3_factor_rule_engine.py \
  --input-dir Stage_2_Report \
  --campaign-input campaigns_clean.csv \
  --output-dir Stage_3_Report
```

Primary outputs:
- `Stage_3_Report/factor_combo_validation_by_sid.csv`
- `Stage_3_Report/best_factor_combos_by_sid.csv`
- `Stage_3_Report/stage_3_refinement_summary.txt`

### 3) Stage 4 Rule Stability
```bash
python stage_4_rule_stability_engine.py \
  --input-dir Stage_3_Report \
  --campaign-input campaigns_clean.csv \
  --output-dir Stage_4_Report
```

Primary outputs:
- `Stage_4_Report/rule_stability_report.csv`
- `Stage_4_Report/stable_factor_combos_by_sid.csv`
- `Stage_4_Report/stage_4_stability_summary.txt`

### 4) Stage 5 Profile Export (Packaging)
```bash
python stage_5_profile_exporter.py \
  --profiles-dir Stage_2_Report \
  --refinement-dir Stage_3_Report \
  --stability-dir Stage_4_Report \
  --output-dir Stage_5_Report
```

Primary outputs:
- `Stage_5_Report/final_profile_package.csv`
- `Stage_5_Report/final_rule_package.csv`
- `Stage_5_Report/setup_summary_export.csv`
- `Stage_5_Report/deployment_baseline_profiles.csv`

### 5) Stage 6 Pine Integration (Optional)
Default lean mode:

```bash
python stage_6_pine_integration_engine.py \
  --deployment-dir Stage_5_Report \
  --output-dir Stage_6_Report
```

Default outputs:
- `Stage_6_Report/pine_profile_rules.txt`
- `Stage_6_Report/pine_factor_rules.csv`
- `Stage_6_Report/pine_integration_summary.txt`

Compatibility mode (extra files):

```bash
python stage_6_pine_integration_engine.py \
  --deployment-dir Stage_5_Report \
  --output-dir Stage_6_Report \
  --include-compatibility-outputs
```

## Recommended Workflows

### Research Only
Stage 2 -> Stage 3 -> Stage 4

### Deployment Packaging
Stage 2 -> Stage 3 -> Stage 4 -> Stage 5

### Pine Translation
Stage 2 -> Stage 3 -> Stage 4 -> Stage 5 -> Stage 6

## Notes
- Stage 4 is the primary validation gate for robustness.
- Stage 5 and Stage 6 are deployment-oriented layers.
- Stage 6 does not change thresholds or discover new logic.
- Output quality depends on data quality and setup filtering.

## Typical Output Directories
- `Stage_2_Report/`
- `Stage_3_Report/`
- `Stage_4_Report/`
- `Stage_5_Report/`
- `Stage_6_Report/`

## Scope and Non-Goals
This repo provides a reproducible analysis and packaging workflow.

It does not:
- execute trades,
- manage orders/positions,
- guarantee profitability.