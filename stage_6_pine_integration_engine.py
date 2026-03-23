#!/usr/bin/env python3
"""
stage_6_pine_integration_engine.py

STAGE 6 — PINE INTEGRATION

Purpose
-------
Read Stage 5 deployment outputs and translate approved profiles/rules into
Pine-ready integration artifacts for Setup Grader workflows.

This stage is translation/integration only. It does not re-rank rules,
re-select combos, discover new logic, or perform robustness testing.

Inputs (from --deployment-dir)
------------------------------
Required:
- deployment_baseline_profiles.csv

Optional:
- deployment_archetype_adjustments.csv
- final_rule_package.csv
- final_profile_package.csv
- setup_summary_export.csv

Outputs
-------
Default Stage 6 outputs (lean mode):
- pine_factor_rules.csv
- pine_profile_rules.txt
- pine_integration_summary.txt

Optional compatibility outputs (--include-compatibility-outputs):
- pine_runner_gate.txt
- pine_archetype_adjustments.txt
- pine_rule_trace.txt
- pine_archetype_score.txt         (compat alias of archetype adjustment text)
- pine_stage7_summary.txt          (compat alias of integration summary)
- pine_environment_score.txt       (compat placeholder)
- pine_runner_score.txt            (compat alias of runner gate text)
- pine_setup_weights.txt           (compat placeholder)

Run Example
-----------
python stage_6_pine_integration_engine.py \\
  --deployment-dir results_export \\
  --output-dir pine_stage6

This stage does NOT own:
- winner discovery
- combo generation
- robustness testing
- packaging/selection decisions
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import pandas as pd

FACTOR_ORDER = ["rs", "cmp", "tr", "vqs", "rmv", "dcr", "orc", "sqz"]


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_int(x: Any) -> Optional[int]:
    try:
        if pd.isna(x):
            return None
        return int(x)
    except Exception:
        return None


def safe_float(x: Any) -> Optional[float]:
    try:
        if pd.isna(x):
            return None
        return float(x)
    except Exception:
        return None


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def split_pipe_values(x: Any) -> List[str]:
    if x is None or pd.isna(x):
        return []
    return [v.strip() for v in str(x).split("|") if v.strip()]


def format_float(x: Any, decimals: int = 4) -> Optional[str]:
    val = safe_float(x)
    if val is None:
        return None
    txt = f"{val:.{decimals}f}"
    return txt.rstrip("0").rstrip(".") if "." in txt else txt


def parse_setup_ids(raw: Optional[str]) -> Optional[Set[int]]:
    if not raw:
        return None
    out: Set[int] = set()
    for part in raw.split(","):
        p = part.strip()
        if not p:
            continue
        if p.isdigit():
            out.add(int(p))
    return out if out else None


# ---------------------------------------------------------------------------
# CLI and loading
# ---------------------------------------------------------------------------

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 6 — Pine Integration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Reads Stage 5 deployment outputs and translates them into Pine-ready "
            "rule/config artifacts without re-running research or selection logic."
        ),
    )
    parser.add_argument(
        "--deployment-dir",
        required=True,
        help="Directory containing Stage 5 outputs",
    )
    parser.add_argument(
        "--output-dir",
        default="pine_stage6",
        help="Output directory for Stage 6 Pine artifacts",
    )
    parser.add_argument(
        "--setup-ids",
        default=None,
        help="Comma-separated SetupIDs to emit (default: all available)",
    )
    parser.add_argument(
        "--include-compatibility-outputs",
        action="store_true",
        help="Also write legacy Stage-7-era filenames as compatibility aliases/placeholders",
    )
    return parser.parse_args()


def load_stage5_inputs(deployment_dir: Path) -> Optional[Dict[str, pd.DataFrame]]:
    baseline = load_csv(deployment_dir / "deployment_baseline_profiles.csv")
    if baseline.empty or "SID" not in baseline.columns:
        print(
            "ERROR: Required Stage 5 file deployment_baseline_profiles.csv is missing "
            "or invalid (needs SID column)."
        )
        return None

    inputs = {
        "deployment_baseline_profiles": baseline,
        "deployment_archetype_adjustments": load_csv(
            deployment_dir / "deployment_archetype_adjustments.csv"
        ),
        "final_rule_package": load_csv(deployment_dir / "final_rule_package.csv"),
        "final_profile_package": load_csv(deployment_dir / "final_profile_package.csv"),
        "setup_summary_export": load_csv(deployment_dir / "setup_summary_export.csv"),
    }
    return inputs


def filter_by_setup_ids(df: pd.DataFrame, setup_ids: Optional[Set[int]], sid_col: str = "SID") -> pd.DataFrame:
    if df is None or df.empty or setup_ids is None:
        return df
    if sid_col not in df.columns:
        return df
    sid_series = pd.to_numeric(df[sid_col], errors="coerce")
    return df[sid_series.isin(setup_ids)].copy()


# ---------------------------------------------------------------------------
# Translation builders
# ---------------------------------------------------------------------------

def build_pine_factor_rules_csv(baseline_profiles: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    baseline_sorted = baseline_profiles.copy()
    baseline_sorted["_sid"] = pd.to_numeric(baseline_sorted["SID"], errors="coerce")
    baseline_sorted = baseline_sorted.dropna(subset=["_sid"]).copy()
    baseline_sorted["_sid"] = baseline_sorted["_sid"].astype(int)
    baseline_sorted = baseline_sorted.sort_values("_sid")

    for _, row in baseline_sorted.iterrows():
        sid = int(row["_sid"])
        for factor in FACTOR_ORDER:
            mn = safe_float(row.get(f"{factor}_min"))
            mx = safe_float(row.get(f"{factor}_max"))
            allowed = row.get(f"{factor}_allowed")
            avoided = row.get(f"{factor}_avoided")

            allowed_txt = "|".join(split_pipe_values(allowed)) or None
            avoided_txt = "|".join(split_pipe_values(avoided)) or None
            has_constraint = (
                mn is not None
                or mx is not None
                or allowed_txt is not None
                or avoided_txt is not None
            )

            rows.append(
                {
                    "SID": sid,
                    "Factor": factor,
                    "Min": mn,
                    "Max": mx,
                    "Allowed_Values": allowed_txt,
                    "Avoided_Values": avoided_txt,
                    "Has_Constraint": bool(has_constraint),
                    "Source": "deployment_baseline_profiles.csv",
                }
            )

    return pd.DataFrame(rows)


def _append_numeric_constraints(lines: List[str], var: str, min_v: Any, max_v: Any) -> None:
    mn = format_float(min_v, 4)
    mx = format_float(max_v, 4)
    if mn is not None:
        lines.append(f"    ok := ok and {var} >= {mn}")
    if mx is not None:
        lines.append(f"    ok := ok and {var} <= {mx}")


def _append_categorical_constraints(lines: List[str], var: str, allowed: Any, avoided: Any) -> None:
    allowed_vals = split_pipe_values(allowed)
    avoided_vals = split_pipe_values(avoided)

    if allowed_vals:
        ors = " or ".join([f'{var} == "{v}"' for v in allowed_vals])
        lines.append(f"    ok := ok and ({ors})")
    if avoided_vals:
        ands = " and ".join([f'{var} != "{v}"' for v in avoided_vals])
        lines.append(f"    ok := ok and ({ands})")


def build_pine_profile_rules_text(baseline_profiles: pd.DataFrame) -> str:
    lines: List[str] = [
        "//==========================================================",
        "// STAGE 6 — PINE PROFILE RULES",
        "//==========================================================",
        "// Source: Stage 5 deployment_baseline_profiles.csv",
        "// Translation only: this file emits already-packaged constraints.",
        "",
    ]

    if baseline_profiles.empty or "SID" not in baseline_profiles.columns:
        lines += [
            "f_profile_ok(_setupId, _rs, _cmp, _tr, _vqs, _rmv, _dcr, _orc, _sqzState, _sqzQualified) =>",
            "    true",
        ]
        return "\n".join(lines)

    sid_series = pd.to_numeric(baseline_profiles["SID"], errors="coerce")
    sid_list = sorted(sid_series.dropna().astype(int).unique().tolist())
    by_sid: Dict[int, pd.Series] = {}
    for sid in sid_list:
        match = baseline_profiles[sid_series == sid]
        if not match.empty:
            by_sid[sid] = match.iloc[0]

    for sid in sid_list:
        row = by_sid[sid]
        lines.append(
            f"f_sid{sid}_profile_ok(_rs, _cmp, _tr, _vqs, _rmv, _dcr, _orc, _sqzState, _sqzQualified) =>"
        )
        lines.append("    bool ok = true")

        _append_numeric_constraints(lines, "_rs", row.get("rs_min"), row.get("rs_max"))
        _append_numeric_constraints(lines, "_cmp", row.get("cmp_min"), row.get("cmp_max"))
        _append_numeric_constraints(lines, "_tr", row.get("tr_min"), row.get("tr_max"))
        _append_numeric_constraints(lines, "_vqs", row.get("vqs_min"), row.get("vqs_max"))
        _append_numeric_constraints(lines, "_rmv", row.get("rmv_min"), row.get("rmv_max"))
        _append_numeric_constraints(lines, "_dcr", row.get("dcr_min"), row.get("dcr_max"))
        _append_numeric_constraints(lines, "_orc", row.get("orc_min"), row.get("orc_max"))

        _append_categorical_constraints(
            lines,
            "_sqzState",
            row.get("sqz_allowed"),
            row.get("sqz_avoided"),
        )

        lines.append("    ok")
        lines.append("")

    lines.append(
        "f_profile_ok(_setupId, _rs, _cmp, _tr, _vqs, _rmv, _dcr, _orc, _sqzState, _sqzQualified) =>"
    )
    lines.append("    bool ok = true")
    for i, sid in enumerate(sid_list):
        if i == 0:
            lines.append(f"    if _setupId == {sid}")
        else:
            lines.append(f"    else if _setupId == {sid}")
        lines.append(
            f"        ok := f_sid{sid}_profile_ok(_rs, _cmp, _tr, _vqs, _rmv, _dcr, _orc, _sqzState, _sqzQualified)"
        )
    lines.append("    ok")

    return "\n".join(lines)


def build_pine_archetype_adjustments_text(archetype_adjustments: pd.DataFrame) -> str:
    lines: List[str] = [
        "//==========================================================",
        "// STAGE 6 — PINE ARCHETYPE ADJUSTMENTS",
        "//==========================================================",
        "// Source: Stage 5 deployment_archetype_adjustments.csv",
        "// Translation only: no re-scoring or optimization.",
        "",
        "f_archetype_adjustment_note(_setupId, _archetype) =>",
        "    // Placeholder function for compatibility; adjustments are represented below as comments.",
        "    0",
        "",
    ]

    if archetype_adjustments is None or archetype_adjustments.empty:
        lines.append("// No archetype adjustments provided by Stage 5 (expected in baseline-only workflows).")
        return "\n".join(lines)

    df = archetype_adjustments.copy()
    if "sid" in df.columns:
        df["_sid"] = pd.to_numeric(df["sid"], errors="coerce")
        df = df.dropna(subset=["_sid"]).copy()
        df["_sid"] = df["_sid"].astype(int)
    else:
        lines.append("// Archetype adjustment file present but missing sid column.")
        return "\n".join(lines)

    if "baseline_unchanged" in df.columns:
        unchanged_mask = df["baseline_unchanged"].astype(str).str.lower().isin(["true", "1", "yes"])
        df = df[~unchanged_mask].copy()

    if df.empty:
        lines.append("// Archetype rows exist but all are baseline_unchanged=true.")
        return "\n".join(lines)

    df = df.sort_values(["_sid", "archetype" if "archetype" in df.columns else "_sid"])

    for _, row in df.iterrows():
        sid = int(row["_sid"])
        arch = str(row.get("archetype", "unknown"))
        lines.append(f"// SID {sid} | Archetype: {arch}")
        for factor in FACTOR_ORDER:
            mn = format_float(row.get(f"{factor}_min_override"), 4)
            mx = format_float(row.get(f"{factor}_max_override"), 4)
            allowed = "|".join(split_pipe_values(row.get(f"{factor}_allowed_override")))
            avoided = "|".join(split_pipe_values(row.get(f"{factor}_avoided_override")))
            if mn is None and mx is None and not allowed and not avoided:
                continue
            lines.append(
                f"//   {factor.upper()} min_override={mn or 'NA'} max_override={mx or 'NA'} "
                f"allowed={allowed or 'NA'} avoided={avoided or 'NA'}"
            )
        lines.append("")

    return "\n".join(lines)


def _boolish(x: Any) -> Optional[bool]:
    if x is None or pd.isna(x):
        return None
    if isinstance(x, bool):
        return x
    txt = str(x).strip().lower()
    if txt in ("1", "true", "yes", "on", "y"):
        return True
    if txt in ("0", "false", "no", "off", "n"):
        return False
    return None


def build_pine_runner_gate_text(baseline_profiles: pd.DataFrame) -> str:
    lines: List[str] = [
        "//==========================================================",
        "// STAGE 6 — RUNNER GATE",
        "//==========================================================",
        "// Source: Stage 5 deployment_baseline_profiles.csv",
        "// Translation only: no runner scoring inference is performed here.",
        "",
    ]

    required_cols = {"runner_gate_enabled", "runner_score_min", "runner_profile_label"}
    if baseline_profiles is None or baseline_profiles.empty or not required_cols.issubset(set(baseline_profiles.columns)):
        lines += [
            "// Runner gate is not part of the active exported deployment package.",
            "f_runner_gate_ok(_setupId, _runnerScore) =>",
            "    true",
        ]
        return "\n".join(lines)

    df = baseline_profiles.copy()
    if "SID" not in df.columns:
        lines += [
            "// deployment_baseline_profiles.csv missing SID; no per-SID runner gate emitted.",
            "f_runner_gate_ok(_setupId, _runnerScore) =>",
            "    true",
        ]
        return "\n".join(lines)

    df["_sid"] = pd.to_numeric(df["SID"], errors="coerce")
    df = df.dropna(subset=["_sid"]).copy()
    df["_sid"] = df["_sid"].astype(int)
    df = df.sort_values("_sid")

    lines.append("f_runner_gate_ok(_setupId, _runnerScore) =>")
    lines.append("    bool ok = true")

    emitted = 0
    for _, row in df.iterrows():
        sid = int(row["_sid"])
        enabled = _boolish(row.get("runner_gate_enabled"))
        min_val = format_float(row.get("runner_score_min"), 4)
        label_raw = row.get("runner_profile_label")
        label = "" if label_raw is None or pd.isna(label_raw) else str(label_raw)
        if not enabled or min_val is None:
            continue

        lines.append(f"    // SID {sid} runner gate | label={label or 'NA'} | min={min_val}")
        lines.append(f"    if _setupId == {sid}")
        lines.append(f"        ok := ok and _runnerScore >= {min_val}")
        emitted += 1

    if emitted == 0:
        lines.append("    // Runner gate fields exist but no enabled per-SID gate was exported.")

    lines.append("    ok")
    return "\n".join(lines)


def build_pine_rule_trace_text(
    final_rule_package: pd.DataFrame,
    final_profile_package: pd.DataFrame,
) -> str:
    lines: List[str] = [
        "//==========================================================",
        "// STAGE 6 — RULE TRACE",
        "//==========================================================",
        "// Traceability comments sourced from Stage 5 packaged outputs.",
        "// These comments do not change Pine execution logic.",
        "",
    ]

    if final_rule_package is None or final_rule_package.empty:
        lines.append("// final_rule_package.csv not available or empty.")
    else:
        df = final_rule_package.copy()
        if "SID" in df.columns:
            df["_sid"] = pd.to_numeric(df["SID"], errors="coerce")
            df = df.dropna(subset=["_sid"]).copy()
            df["_sid"] = df["_sid"].astype(int)
            df = df.sort_values(["_sid"]) 
        for sid, grp in df.groupby("_sid"):
            lines.append(f"// SID {sid} rules:")
            for _, row in grp.iterrows():
                rule = str(row.get("Rule", ""))
                label = row.get("Stability_Label", "")
                promoted = row.get("Promoted_Stable", "")
                lift = row.get("Expectancy_Lift_Pct", "")
                lines.append(
                    f"//   rule={rule} | label={label} | promoted={promoted} | lift_pct={lift}"
                )
            lines.append("")

    if final_profile_package is not None and not final_profile_package.empty:
        lines.append("// Best-rule per SID from final_profile_package.csv")
        dfp = final_profile_package.copy()
        if "SID" in dfp.columns:
            dfp["_sid"] = pd.to_numeric(dfp["SID"], errors="coerce")
            dfp = dfp.dropna(subset=["_sid"]).copy()
            dfp["_sid"] = dfp["_sid"].astype(int)
            dfp = dfp.sort_values("_sid")
            for _, row in dfp.iterrows():
                sid = int(row["_sid"])
                best_rule = row.get("Best_Rule")
                st_label = row.get("Stability_Label")
                lines.append(
                    f"//   SID {sid}: best_rule={best_rule} | stability_label={st_label}"
                )

    return "\n".join(lines)


def build_integration_summary_text(inputs: Dict[str, pd.DataFrame], emitted_sids: List[int]) -> str:
    baseline_df = inputs["deployment_baseline_profiles"]
    archetype_df = inputs["deployment_archetype_adjustments"]
    rule_df = inputs["final_rule_package"]

    runner_gate_supported = {"runner_gate_enabled", "runner_score_min", "runner_profile_label"}.issubset(
        set(baseline_df.columns)
    )
    runner_enabled_count = 0
    if runner_gate_supported and not baseline_df.empty:
        runner_enabled_count = int(
            baseline_df["runner_gate_enabled"].astype(str).str.lower().isin(["true", "1", "yes", "on"]).sum()
        )

    lines: List[str] = [
        "STAGE 6 — PINE INTEGRATION SUMMARY",
        "==================================",
        "",
        "Role:",
        "- Optional, deployment-oriented translation layer.",
        "- Translates Stage 5 packaged outputs into Pine-ready artifacts.",
        "- Deterministic and mechanical only (no re-ranking/re-selection).",
        "",
        "Input stats:",
        f"- deployment_baseline_profiles.csv rows: {len(baseline_df)}",
        f"- deployment_archetype_adjustments.csv rows: {len(archetype_df)}",
        f"- final_rule_package.csv rows: {len(rule_df)}",
        f"- SetupIDs emitted: {', '.join([str(x) for x in emitted_sids]) if emitted_sids else 'none'}",
        f"- Runner gate fields present: {runner_gate_supported}",
        f"- Runner-gated SetupIDs: {runner_enabled_count}",
        "",
        "Default files written:",
        "- pine_factor_rules.csv",
        "- pine_profile_rules.txt",
        "- pine_integration_summary.txt",
        "",
        "Compatibility files (with --include-compatibility-outputs):",
        "- pine_runner_gate.txt",
        "- pine_archetype_adjustments.txt",
        "- pine_rule_trace.txt",
        "",
        "Notes:",
        "- Stage 6 does not generate new thresholds or scores.",
        "- Stage 6 uses Stage 5 deployment outputs as the source of truth.",
        "- Default mode emits only essential Pine artifacts.",
        "- Compatibility files are emitted only when explicitly requested.",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Compatibility outputs
# ---------------------------------------------------------------------------

def write_compatibility_outputs(
    output_dir: Path,
    archetype_text: str,
    summary_text: str,
    runner_gate_text: str,
) -> None:
    # Compatibility alias retained because some existing integration workflows
    # still look for the older filename.
    write_text(output_dir / "pine_archetype_score.txt", archetype_text)

    # Compatibility alias retained for older docs/scripts that refer to Stage 7 summary.
    write_text(output_dir / "pine_stage7_summary.txt", summary_text)

    placeholder = "\n".join(
        [
            "// Compatibility placeholder",
            "// Stage 6 no longer derives this layer from analyzer outputs.",
            "// Translation source of truth is Stage 5 deployment packaging.",
        ]
    )
    write_text(output_dir / "pine_environment_score.txt", placeholder)
    # Compatibility alias retained for older filenames expecting runner score text.
    write_text(output_dir / "pine_runner_score.txt", runner_gate_text)
    write_text(output_dir / "pine_setup_weights.txt", placeholder)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_arguments()

    deployment_dir = Path(args.deployment_dir)
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    inputs = load_stage5_inputs(deployment_dir)
    if inputs is None:
        raise SystemExit(1)

    setup_ids = parse_setup_ids(args.setup_ids)

    baseline_profiles = filter_by_setup_ids(
        inputs["deployment_baseline_profiles"], setup_ids, sid_col="SID"
    )

    archetype_adjustments = inputs["deployment_archetype_adjustments"]
    if not archetype_adjustments.empty and "sid" in archetype_adjustments.columns:
        archetype_adjustments = filter_by_setup_ids(
            archetype_adjustments, setup_ids, sid_col="sid"
        )

    final_rule_package = filter_by_setup_ids(
        inputs["final_rule_package"], setup_ids, sid_col="SID"
    )
    final_profile_package = filter_by_setup_ids(
        inputs["final_profile_package"], setup_ids, sid_col="SID"
    )

    # Build translation artifacts
    pine_factor_rules_df = build_pine_factor_rules_csv(baseline_profiles)
    profile_rules_text = build_pine_profile_rules_text(baseline_profiles)
    runner_gate_text = build_pine_runner_gate_text(baseline_profiles)
    archetype_text = build_pine_archetype_adjustments_text(archetype_adjustments)
    rule_trace_text = build_pine_rule_trace_text(final_rule_package, final_profile_package)

    sid_series = pd.to_numeric(baseline_profiles.get("SID"), errors="coerce")
    emitted_sids = sorted(sid_series.dropna().astype(int).unique().tolist()) if sid_series is not None else []
    summary_text = build_integration_summary_text(inputs, emitted_sids)

    # Write lean default Stage 6 outputs
    pine_factor_rules_df.to_csv(output_dir / "pine_factor_rules.csv", index=False)
    write_text(output_dir / "pine_profile_rules.txt", profile_rules_text)
    write_text(output_dir / "pine_integration_summary.txt", summary_text)

    if args.include_compatibility_outputs:
        # Backward-compatible outputs retained for legacy integration paths.
        write_text(output_dir / "pine_runner_gate.txt", runner_gate_text)
        write_text(output_dir / "pine_archetype_adjustments.txt", archetype_text)
        write_text(output_dir / "pine_rule_trace.txt", rule_trace_text)
        write_compatibility_outputs(output_dir, archetype_text, summary_text, runner_gate_text)

    print("\n" + "=" * 70)
    print("STAGE 6 — PINE INTEGRATION COMPLETE")
    print("=" * 70)
    print(f"Output directory: {output_dir}")
    print("\nDefault outputs:")
    print("- pine_factor_rules.csv")
    print("- pine_profile_rules.txt")
    print("- pine_integration_summary.txt")
    if args.include_compatibility_outputs:
        print("\nCompatibility outputs enabled:")
        print("- pine_runner_gate.txt")
        print("- pine_archetype_adjustments.txt")
        print("- pine_rule_trace.txt")
        print("- pine_archetype_score.txt")
        print("- pine_stage7_summary.txt")
        print("- pine_environment_score.txt")
        print("- pine_runner_score.txt")
        print("- pine_setup_weights.txt")


if __name__ == "__main__":
    main()
