#!/usr/bin/env python3
"""
Analyze campaigns_clean.csv by SetupID and factor buckets.

Purpose:
- Break down performance by SID and factor bucket
- Evaluate RS, CMP, TR, VQS, RMV, DCR, ORC, SQZ/SqzQualified
- Produce a text report plus a CSV of bucket-level stats

Example:
python analyze_setupid_factor_buckets.py --input campaigns_clean.csv --output-report Stage_2_Bucket_Report/setupid_factor_bucket_report.txt --output-csv Stage_2_Bucket_Report/setupid_factor_bucket_stats.csv --min-trades 20

"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd


FACTOR_SPECS = {
    "RS": {
        "value_aliases": ["RS", "rs"],
        "bucket_aliases": ["RS_bucket", "rs_bucket"],
    },
    "CMP": {
        "value_aliases": ["CMP", "cmp"],
        "bucket_aliases": ["CMP_bucket", "cmp_bucket"],
    },
    "TR": {
        "value_aliases": ["TR", "tr"],
        "bucket_aliases": ["TR_bucket", "tr_bucket"],
    },
    "VQS": {
        "value_aliases": ["VQS", "vqs"],
        "bucket_aliases": ["VQS_bucket", "vqs_bucket"],
    },
    "RMV": {
        "value_aliases": ["RMV", "rmv"],
        "bucket_aliases": ["RMV_bucket", "rmv_bucket"],
    },
    "DCR": {
        "value_aliases": ["DCR", "dcr"],
        "bucket_aliases": ["DCR_bucket", "dcr_bucket"],
    },
    "ORC": {
        "value_aliases": ["ORC", "Orc", "orc"],
        "bucket_aliases": ["ORC_bucket", "orc_bucket"],
    },
    "GRP": {
        "value_aliases": ["GRP", "grp"],
        "bucket_aliases": ["GRP_bucket", "GRP_LABEL", "grp_bucket", "grp_label"],
    },
    "SQZ": {
        "value_aliases": ["SQZ", "SqzQualified", "sqzqualified", "sqzQualified"],
        "bucket_aliases": [],
    },
}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Analyze SetupID factor buckets from campaigns_clean.csv")
    p.add_argument("--input", required=True, help="Input campaigns_clean.csv")
    p.add_argument("--output-report", default="setupid_factor_bucket_report.txt", help="Text report path")
    p.add_argument("--output-csv", default="setupid_factor_bucket_stats.csv", help="CSV output path")
    p.add_argument("--min-trades", type=int, default=20, help="Minimum trades per bucket to include")
    p.add_argument("--setup-ids", default="", help="Optional comma-separated SIDs to include, e.g. 1,2,3")
    return p


def find_column(df: pd.DataFrame, candidates: Sequence[str], required: bool = True) -> Optional[str]:
    lowered = {str(c).strip().lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lowered:
            return lowered[cand.lower()]
    if required:
        raise KeyError(f"Could not find required column. Tried: {candidates}")
    return None


def first_existing_column(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    lowered = {str(c).strip().lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lowered:
            return lowered[cand.lower()]
    return None


def safe_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def fmt_pct(x: float | int | None) -> str:
    if x is None or pd.isna(x):
        return "n/a"
    return f"{x:.2f}%"


def fmt_num(x: float | int | None, digits: int = 2) -> str:
    if x is None or pd.isna(x):
        return "n/a"
    return f"{x:.{digits}f}"


def fmt_money(x: float | int | None) -> str:
    if x is None or pd.isna(x):
        return "n/a"
    return f"${x:,.2f}"


def parse_setup_ids(raw: str) -> Optional[list[int]]:
    if not raw.strip():
        return None
    out: list[int] = []
    for token in raw.split(","):
        tok = token.strip()
        if tok:
            out.append(int(tok))
    return sorted(set(out))


def normalize_sqz_bucket(series: pd.Series) -> pd.Series:
    def _norm(x: object) -> str:
        if pd.isna(x):
            return "Unknown"
        s = str(x).strip().lower()
        if s in {"1", "true", "on", "yes", "sqon", "qualified"}:
            return "Qualified"
        if s in {"0", "false", "off", "no", "none"}:
            return "Not Qualified"
        return str(x).strip() or "Unknown"
    return series.apply(_norm)


def summarize_bucket(df: pd.DataFrame, ret_col: str, pnl_col: Optional[str]) -> dict[str, float]:
    r = pd.to_numeric(df[ret_col], errors="coerce").dropna()
    if r.empty:
        return {
            "trades": 0,
            "wins": 0,
            "win_rate": float("nan"),
            "expectancy_pct": float("nan"),
            "median_return_pct": float("nan"),
            "avg_win_pct": float("nan"),
            "avg_loss_pct": float("nan"),
            "net_pnl_usd": float("nan"),
        }

    wins = r[r > 0]
    losses = r[r < 0]

    net_pnl = float("nan")
    if pnl_col and pnl_col in df.columns:
        pnl = pd.to_numeric(df[pnl_col], errors="coerce")
        net_pnl = float(pnl.sum()) if not pnl.dropna().empty else float("nan")

    return {
        "trades": int(len(r)),
        "wins": int((r > 0).sum()),
        "win_rate": float((r > 0).mean() * 100.0),
        "expectancy_pct": float(r.mean() * 100.0),
        "median_return_pct": float(r.median() * 100.0),
        "avg_win_pct": float(wins.mean() * 100.0) if not wins.empty else float("nan"),
        "avg_loss_pct": float(losses.mean() * 100.0) if not losses.empty else float("nan"),
        "net_pnl_usd": net_pnl,
    }


def main() -> int:
    args = build_parser().parse_args()

    input_path = Path(args.input)
    report_path = Path(args.output_report)
    csv_path = Path(args.output_csv)

    try:
        df = pd.read_csv(input_path)
        df.columns = [str(c).strip() for c in df.columns]

        sid_col = find_column(df, ["sid", "SID", "setupid", "setup_id"])
        ret_col = find_column(df, ["return_pct", "ReturnPct", "return"])
        pnl_col = first_existing_column(df, ["pnl_usd", "PnlUsd", "net_pnl_usd"])
        entry_col = first_existing_column(df, ["entry_dt", "EntryDate", "entry_date"])
        symbol_col = first_existing_column(df, ["symbol", "Symbol", "ticker", "Ticker"])

        df[sid_col] = safe_num(df[sid_col]).astype("Int64")
        df[ret_col] = safe_num(df[ret_col])
        if pnl_col:
            df[pnl_col] = safe_num(df[pnl_col])

        df = df.dropna(subset=[sid_col, ret_col]).copy()
        df[sid_col] = df[sid_col].astype(int)

        selected_sids = parse_setup_ids(args.setup_ids)
        if selected_sids:
            df = df[df[sid_col].isin(selected_sids)].copy()

        rows: list[dict[str, object]] = []
        report_lines: list[str] = []

        report_lines.append("SETUPID FACTOR BUCKET REPORT")
        report_lines.append("============================")
        report_lines.append("")
        report_lines.append(f"Input file: {input_path}")
        report_lines.append(f"Rows analyzed: {len(df):,}")
        report_lines.append(f"Minimum bucket trades: {args.min_trades}")
        report_lines.append("")

        for sid in sorted(df[sid_col].unique().tolist()):
            sid_df = df[df[sid_col] == sid].copy()
            sid_base = summarize_bucket(sid_df, ret_col, pnl_col)

            report_lines.append(f"SID {sid}")
            report_lines.append("-" * 72)
            report_lines.append(
                f"Baseline: trades={sid_base['trades']}, "
                f"win_rate={fmt_pct(sid_base['win_rate'])}, "
                f"expectancy={fmt_pct(sid_base['expectancy_pct'])}, "
                f"median={fmt_pct(sid_base['median_return_pct'])}, "
                f"net_pnl={fmt_money(sid_base['net_pnl_usd'])}"
            )

            for factor, spec in FACTOR_SPECS.items():
                value_col = first_existing_column(sid_df, spec["value_aliases"])
                bucket_col = first_existing_column(sid_df, spec["bucket_aliases"],)

                if factor == "SQZ":
                    if value_col is None:
                        continue
                    sid_df["_sqz_bucket_temp"] = normalize_sqz_bucket(sid_df[value_col])
                    working_bucket_col = "_sqz_bucket_temp"
                else:
                    if bucket_col is None:
                        continue
                    working_bucket_col = bucket_col

                factor_rows: list[dict[str, object]] = []
                grouped = sid_df.groupby(working_bucket_col, dropna=False)

                for bucket_name, bucket_df in grouped:
                    stats = summarize_bucket(bucket_df, ret_col, pnl_col)
                    if stats["trades"] < args.min_trades:
                        continue

                    row = {
                        "SID": sid,
                        "Factor": factor,
                        "Bucket": str(bucket_name),
                        "Trades": stats["trades"],
                        "Wins": stats["wins"],
                        "Win_Rate_Pct": stats["win_rate"],
                        "Expectancy_Pct": stats["expectancy_pct"],
                        "Median_Return_Pct": stats["median_return_pct"],
                        "Avg_Win_Pct": stats["avg_win_pct"],
                        "Avg_Loss_Pct": stats["avg_loss_pct"],
                        "Net_PnL_USD": stats["net_pnl_usd"],
                        "Baseline_Expectancy_Pct": sid_base["expectancy_pct"],
                        "Expectancy_Lift_Pct": (
                            stats["expectancy_pct"] - sid_base["expectancy_pct"]
                            if not pd.isna(stats["expectancy_pct"]) and not pd.isna(sid_base["expectancy_pct"])
                            else float("nan")
                        ),
                    }
                    rows.append(row)
                    factor_rows.append(row)

                if factor_rows:
                    factor_rows = sorted(
                        factor_rows,
                        key=lambda x: (
                            float("-inf") if pd.isna(x["Expectancy_Lift_Pct"]) else x["Expectancy_Lift_Pct"],
                            x["Trades"],
                        ),
                        reverse=True,
                    )

                    report_lines.append("")
                    report_lines.append(f"  {factor}")
                    for r in factor_rows:
                        report_lines.append(
                            f"    {r['Bucket']}: "
                            f"trades={r['Trades']}, "
                            f"win_rate={fmt_pct(r['Win_Rate_Pct'])}, "
                            f"exp={fmt_pct(r['Expectancy_Pct'])}, "
                            f"lift={fmt_pct(r['Expectancy_Lift_Pct'])}, "
                            f"net_pnl={fmt_money(r['Net_PnL_USD'])}"
                        )

            report_lines.append("")

        result_df = pd.DataFrame(rows)
        if not result_df.empty:
            result_df = result_df.sort_values(
                ["SID", "Factor", "Expectancy_Lift_Pct", "Trades"],
                ascending=[True, True, False, False],
                na_position="last",
            )

        report_path.parent.mkdir(parents=True, exist_ok=True)
        csv_path.parent.mkdir(parents=True, exist_ok=True)

        result_df.to_csv(csv_path, index=False, float_format="%.4f")
        report_path.write_text("\n".join(report_lines), encoding="utf-8")

        print({
            "input_rows": int(len(df)),
            "sids_analyzed": sorted(df[sid_col].unique().tolist()),
            "output_csv_rows": int(len(result_df)),
            "report_file": str(report_path),
            "csv_file": str(csv_path),
        })
        return 0

    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())