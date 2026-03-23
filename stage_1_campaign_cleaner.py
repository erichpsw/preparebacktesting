#!/usr/bin/env python3
"""
TradingView "List of Trades" CSV -> Campaign-level research table.
Stage 1

Implements:
1) Fix malformed rows (two trades jammed together) by splitting on record starts:
   "<integer>,Entry long," or "<integer>,Exit long,"
2) Parse corrected CSV
3) Reconstruct campaigns (entry + 1-3 partial exits)
4) Extract ENTRY payload features ONLY (never from EXIT rows)
5) Standardize fields + exact buckets
6) Validate + report anomalies
7) Output clean research table CSV (Excel-ready, UTF-8 BOM)

NEW:
- Detect runner flag from payload State field when HQS is present
- Keep STATE normalized as Ready Now / Near Term / Not Imminent
- Add RUNNER output column
- Add ORC (Oracle Weekly Pivot Point Distance) output column + bucket
- Add GRP (Group Strength state) output column + bucket
- Add DDV (Average Daily Dollar Volume, $M) output column + bucket
- Write output/rejects with UTF-8 BOM for Excel friendliness

python stage_1_campaign_cleaner.py --input merged.csv --output campaigns_clean.csv


Examples:
  State=READYNOW-HQS  -> STATE="Ready Now", RUNNER=True
  State=NEAR TERM     -> STATE="Near Term", RUNNER=False

  GRP=2   -> Strong Group
  GRP=1   -> Improving / Strengthening
  GRP=0   -> Neutral / Mixed
  GRP=-1  -> Weakening Group
  GRP=-2  -> Weak Group

  DDV=87.4 -> $87.4M average daily dollar volume
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

try:
    from dateutil import parser as dtparser  # type: ignore
except Exception:
    dtparser = None


# -------------------------
# Helpers
# -------------------------

RE_RECORD_START = re.compile(r'(?=(\d+,(Entry long|Exit long),))')


def parse_dt(s: str) -> Optional[datetime]:
    s = (s or "").strip()
    if not s:
        return None
    if dtparser:
        try:
            return dtparser.parse(s)
        except Exception:
            return None
    try:
        return datetime.fromisoformat(s.replace("Z", "").replace("T", " "))
    except Exception:
        return None


def to_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    s = str(x).strip()
    if not s:
        return None
    s = s.replace("$", "").replace(",", "").replace("%", "")
    try:
        return float(s)
    except Exception:
        return None


def r2(x: Optional[float]) -> Optional[float]:
    return round(x, 2) if x is not None else None


def r4(x: Optional[float]) -> Optional[float]:
    return round(x, 4) if x is not None else None


def parse_runner_flag(raw: Optional[str]) -> bool:
    """
    HQS in the raw State payload means the setup was flagged as a runner.
    Examples:
      READYNOW-HQS
      READY NOW HQS
      NEARTERM_HQS
    """
    if raw is None:
        return False
    s = raw.strip().upper()
    if not s:
        return False
    return "HQS" in s


def norm_state(raw: Optional[str]) -> Optional[str]:
    """
    Normalize payload state into clean research buckets while stripping HQS runner suffix/tag.
    """
    if raw is None:
        return None
    s = raw.strip().lower()
    if not s:
        return None

    s = s.replace("_hqs", "").replace("-hqs", "").replace(" hqs", "").strip()

    if s in {"readynow", "ready now", "ready_now", "ready"}:
        return "Ready Now"
    if s in {"near", "near term", "near_term", "nearterm"}:
        return "Near Term"
    if s in {"not", "not imminent", "not_imminent", "notimminent", "intermittent"}:
        return "Not Imminent"

    if "near" in s:
        return "Near Term"
    if "ready" in s:
        return "Ready Now"
    if "not" in s or "immi" in s or "inter" in s:
        return "Not Imminent"

    return None


def norm_sqz(raw: Optional[str]) -> Optional[str]:
    if raw is None:
        return None
    s = raw.strip()
    if not s or s == "-":
        return None

    s_up = s.upper()
    if s_up == "ON":
        return "On"
    if s_up == "OFF":
        return "Off"

    return None


def norm_mode(raw: Optional[str]) -> str:
    if raw is None:
        return "Unknown"
    s = raw.strip().upper()
    if s == "C":
        return "Contrarian"
    if s == "B":
        return "Breakout"
    return "Unknown"


def norm_grp(raw: Optional[str]) -> Optional[int]:
    """
    Group strength state from payload:
      2   = Strong Group
      1   = Improving / Strengthening
      0   = Neutral / Mixed
     -1   = Weakening
     -2   = Weak
    """
    if raw is None:
        return None
    s = raw.strip()
    if not s:
        return None

    try:
        value = int(float(s))
    except Exception:
        return None

    if value in {2, 1, 0, -1, -2}:
        return value
    return None


def grp_label(val: Optional[int]) -> Optional[str]:
    if val is None:
        return None
    mapping = {
        2: "Strong",
        1: "Improving/Strengthening",
        0: "Neutral",
        -1: "Weakening",
        -2: "Weak",
    }
    return mapping.get(val)


def bucket_value(val: Optional[float], edges: List[float], labels: List[str]) -> Optional[str]:
    """
    edges: increasing list of cut points, length = len(labels)-1
    labels correspond to:
      (-inf, edges[0]) -> labels[0]
      [edges[0], edges[1]) -> labels[1]
      ...
      [edges[-1], +inf) -> labels[-1]
    """
    if val is None:
        return None
    for i, cut in enumerate(edges):
        if val < cut:
            return labels[i]
    return labels[-1]


def bucket_grp(val: Optional[int]) -> Optional[str]:
    if val is None:
        return None
    mapping = {
        2: "Strong",
        1: "Improving",
        0: "Neutral",
        -1: "Weakening",
        -2: "Weak",
    }
    return mapping.get(val)


def detect_targets(comment: str) -> List[int]:
    s = (comment or "").lower()
    hits = []
    if "target 1" in s or "t1" in s:
        hits.append(1)
    if "target 2" in s or "t2" in s:
        hits.append(2)
    if "target 3" in s or "t3" in s:
        hits.append(3)
    return sorted(set(hits))


# -------------------------
# Payload parsing
# -------------------------

def parse_payload(comment: str) -> Tuple[Optional[str], Dict[str, str]]:
    """
    ENTRY comment format:
      "SYMBOL|SID=4|SS=84|TR=98|RS=98|CMP=80|VQS=72|ADR=2.8|DCR=94|RMV=63|State=NEAR TERM|Sqz=-|Orc=0.42|GRP=1|DDV=87.4|MODE=B"

    Returns: (symbol, kv dict)
    """
    if not comment:
        return None, {}
    parts = comment.split("|")
    if not parts:
        return None, {}
    symbol = parts[0].strip() or None
    kv: Dict[str, str] = {}
    for p in parts[1:]:
        if "=" in p:
            k, v = p.split("=", 1)
            kv[k.strip()] = v.strip()
    return symbol, kv


def get_campaign_id(kv: Dict[str, str], entry_dt_str: str) -> str:
    for key in ("LastPivotHighTime", "LPHTime", "CampaignID", "CID"):
        if key in kv and kv[key].strip():
            return kv[key].strip()
    return entry_dt_str


# -------------------------
# Core data structures
# -------------------------

@dataclass
class TVRow:
    raw_line_no: int
    trade_no: Optional[int]
    typ: str
    dt: Optional[datetime]
    dt_str: str
    price: Optional[float]
    qty: Optional[float]
    net_pnl_usd: Optional[float]
    comment: str
    raw: Dict[str, str] = field(default_factory=dict)


@dataclass
class Campaign:
    trade_no: int
    symbol: str
    sid: int
    entry_dt: datetime
    entry_dt_str: str
    campaign_id: str
    entry_line_no: int

    entry_px: float
    entry_payload: Dict[str, str]

    exits: List[TVRow] = field(default_factory=list)

    def add_exit(self, r: TVRow) -> None:
        self.exits.append(r)


# -------------------------
# STEP 1 — Fix malformed CSV
# -------------------------

def split_jammed_lines(text: str) -> List[str]:
    """
    If a line contains two records jammed together, split whenever a new record start appears.
    Record starts: "<integer>,Entry long," OR "<integer>,Exit long,"
    """
    out_lines: List[str] = []
    for original_line in text.splitlines():
        line = original_line.lstrip("\ufeff")
        if not line.strip():
            continue

        matches = [m.start() for m in RE_RECORD_START.finditer(line)]
        if len(matches) <= 1:
            out_lines.append(line)
            continue

        for i, start in enumerate(matches):
            end = matches[i + 1] if i + 1 < len(matches) else len(line)
            chunk = line[start:end].strip()
            if chunk:
                out_lines.append(chunk)
    return out_lines


# -------------------------
# CSV parsing (robust)
# -------------------------

def find_col(header: List[str], candidates: List[str]) -> Optional[str]:
    lowered = {h.lower(): h for h in header}
    for c in candidates:
        if c.lower() in lowered:
            return lowered[c.lower()]
    for h in header:
        hl = h.lower()
        for c in candidates:
            if c.lower() in hl:
                return h
    return None


def parse_tradingview_csv(corrected_lines: List[str]) -> Tuple[List[TVRow], Dict[str, Any]]:
    """
    Parses TradingView List of Trades CSV after correction.
    Attempts to map needed fields by header name.
    """
    stats: Dict[str, Any] = {
        "raw_lines_after_correction": len(corrected_lines),
        "parsed_rows": 0,
        "skipped_blank_rows": 0,
        "skipped_bad_column_rows": 0,
        "trimmed_wide_rows": 0,
        "skipped_header_duplicates": 0,
        "missing_required_cols": [],
    }

    if not corrected_lines:
        return [], stats

    reader = csv.reader(corrected_lines)
    try:
        header = next(reader)
    except StopIteration:
        return [], stats

    header[0] = header[0].lstrip("\ufeff")

    col_type = find_col(header, ["Type"])
    col_dt = find_col(header, ["Date and time", "Date/Time", "Date", "Time"])
    col_price = find_col(header, ["Price", "Price USD", "Price (USD)"])
    col_qty = find_col(header, [
        "Position size",
        "Position size (qty)",
        "Qty",
        "Quantity"
    ])
    col_pnl = find_col(header, [
        "Net P&L",
        "Net P/L",
        "Net P&L USD",
        "Net P/L USD",
        "P&L USD",
        "Profit USD"
    ])
    col_trade_no = find_col(header, ["Trade #", "Trade", "Trade No"])
    col_comment = find_col(header, ["Comment", "Comments", "Signal", "Order comment", "Strategy comment"])

    print({
        "col_trade_no": col_trade_no,
        "col_type": col_type,
        "col_dt": col_dt,
        "col_price": col_price,
        "col_qty": col_qty,
        "col_pnl": col_pnl,
        "col_comment": col_comment,
    })

    required = [("Type", col_type), ("Date and time", col_dt), ("Comment/Signal", col_comment)]
    missing = [name for name, col in required if col is None]
    if missing:
        stats["missing_required_cols"] = missing

    rows: List[TVRow] = []
    for i, rec in enumerate(reader, start=2):
        if rec == header:
            stats["skipped_header_duplicates"] += 1
            continue

        if not any(rec):
            stats["skipped_blank_rows"] += 1
            continue

        # Newer TradingView exports can include extra trailing metrics beyond header width.
        # We only need core columns, so trim wider rows and only reject rows that are too short.
        if len(rec) < len(header):
            stats["skipped_bad_column_rows"] += 1
            continue
        if len(rec) > len(header):
            stats["trimmed_wide_rows"] += 1
            rec = rec[:len(header)]

        d = dict(zip(header, rec))
        typ = (d.get(col_type, "") if col_type else "").strip()
        dt_str = (d.get(col_dt, "") if col_dt else "").strip()
        comment = (d.get(col_comment, "") if col_comment else "").strip()

        trade_no = None
        if col_trade_no:
            try:
                trade_no = int(str(d.get(col_trade_no, "")).strip())
            except Exception:
                trade_no = None

        dt = parse_dt(dt_str)
        price = to_float(d.get(col_price, "")) if col_price else None
        qty = to_float(d.get(col_qty, "")) if col_qty else None
        pnl = to_float(d.get(col_pnl, "")) if col_pnl else None

        rows.append(TVRow(
            raw_line_no=i,
            trade_no=trade_no,
            typ=typ,
            dt=dt,
            dt_str=dt_str,
            price=price,
            qty=qty,
            net_pnl_usd=pnl,
            comment=comment,
            raw=d
        ))
        stats["parsed_rows"] += 1

    return rows, stats


# -------------------------
# STEP 2 — Reconstruct campaigns
# -------------------------

def build_campaigns(rows: List[TVRow]) -> Tuple[Dict[Tuple[str, int, str, str, int], Campaign], Dict[str, Any], List[List[Any]]]:
    """
    Campaign grouping keys:
      - entry timestamp (string)
      - symbol (from ENTRY payload)
      - SetupID (SID from ENTRY payload)
      - Campaign ID (LastPivotHighTime if present; else entry timestamp)
    """
    stats = {
        "entries": 0,
        "exits": 0,
        "campaigns": 0,
        "entry_payload_missing": 0,
        "exit_without_campaign": 0,
        "exit_symbol_mismatch": 0,
        "entry_missing_trade_no": 0,
        "exit_missing_trade_no": 0,
        "campaigns_without_exits_after_linkage": 0,
        "missing_payload_keys": 0,
        "unknown_state": 0,
        "unknown_sqz": 0,
        "unknown_mode": 0,
        "unknown_grp": 0,
        "unknown_ddv": 0,
    }
    rejects: List[List[Any]] = [["issue", "line_no", "type", "dt", "comment"]]

    campaigns: Dict[Tuple[str, int, str, str, int], Campaign] = {}
    campaigns_by_trade_no: Dict[int, List[Campaign]] = {}
    campaign_closed: Dict[Tuple[str, int, str, str, int], bool] = {}

    for r in rows:
        t = (r.typ or "").strip()
        if t == "Entry long":
            stats["entries"] += 1
            if r.trade_no is None:
                stats["entry_missing_trade_no"] += 1
                rejects.append(["entry_missing_trade_no", r.raw_line_no, r.typ, r.dt_str, r.comment])
                continue

            sym, kv = parse_payload(r.comment)
            if not sym or "SID" not in kv:
                stats["entry_payload_missing"] += 1
                rejects.append(["missing_entry_payload_or_symbol_or_SID", r.raw_line_no, r.typ, r.dt_str, r.comment])
                continue

            sid = None
            try:
                sid = int(float(kv.get("SID", "").strip()))
            except Exception:
                rejects.append(["bad_SID", r.raw_line_no, r.typ, r.dt_str, r.comment])
                continue

            if r.dt is None:
                rejects.append(["bad_entry_datetime", r.raw_line_no, r.typ, r.dt_str, r.comment])
                continue

            if r.price is None:
                rejects.append(["missing_entry_price", r.raw_line_no, r.typ, r.dt_str, r.comment])
                continue

            cid = get_campaign_id(kv, r.dt_str)
            key = (r.dt_str, sym, str(sid), cid, r.raw_line_no)

            campaign = Campaign(
                trade_no=r.trade_no,
                symbol=sym,
                sid=sid,
                entry_dt=r.dt,
                entry_dt_str=r.dt_str,
                campaign_id=cid,
                entry_line_no=r.raw_line_no,
                entry_px=r.price,
                entry_payload=kv
            )
            campaigns[key] = campaign
            campaigns_by_trade_no.setdefault(r.trade_no, []).append(campaign)
            campaign_closed[key] = False

        elif t == "Exit long":
            stats["exits"] += 1
        else:
            continue

    for trade_no in campaigns_by_trade_no:
        campaigns_by_trade_no[trade_no].sort(key=lambda x: x.entry_dt)

    for r in rows:
        if (r.typ or "").strip() != "Exit long":
            continue
        if r.trade_no is None:
            stats["exit_missing_trade_no"] += 1
            stats["exit_without_campaign"] += 1
            rejects.append(["exit_missing_trade_no", r.raw_line_no, r.typ, r.dt_str, r.comment])
            continue

        candidates = campaigns_by_trade_no.get(r.trade_no, [])
        if not candidates or r.dt is None:
            c = None
        else:
            exit_sym, _ = parse_payload(r.comment)
            c = None
            symbol_matched_candidate_exists = False

            for cand in reversed(candidates):
                if cand.entry_dt > r.dt:
                    continue
                ckey = (cand.entry_dt_str, cand.symbol, str(cand.sid), cand.campaign_id, cand.entry_line_no)
                if campaign_closed.get(ckey, False):
                    continue
                if exit_sym:
                    if cand.symbol == exit_sym:
                        symbol_matched_candidate_exists = True
                        c = cand
                        break
                    symbol_matched_candidate_exists = True
                    continue
                c = cand
                break

            if c is None and exit_sym and symbol_matched_candidate_exists:
                stats["exit_symbol_mismatch"] += 1
                stats["exit_without_campaign"] += 1
                rejects.append([
                    "exit_symbol_mismatch_within_trade_no",
                    r.raw_line_no,
                    r.typ,
                    r.dt_str,
                    r.comment
                ])
                continue

        if c is None:
            stats["exit_without_campaign"] += 1
            rejects.append(["exit_without_trade_no_campaign_match", r.raw_line_no, r.typ, r.dt_str, r.comment])
            continue
        c.add_exit(r)
        ckey = (c.entry_dt_str, c.symbol, str(c.sid), c.campaign_id, c.entry_line_no)
        campaign_closed[ckey] = True

    for c in campaigns.values():
        if not c.exits:
            stats["campaigns_without_exits_after_linkage"] += 1
            rejects.append([
                "entry_has_no_exit_after_trade_no_linkage",
                c.entry_line_no,
                "Entry long",
                c.entry_dt_str,
                f"trade_no={c.trade_no}|symbol={c.symbol}|sid={c.sid}|campaign_id={c.campaign_id}"
            ])

    stats["campaigns"] = len(campaigns)
    return campaigns, stats, rejects

# -------------------------
# STEP 3-7 — Build research table
# -------------------------

OUTPUT_COLS = [
    "symbol", "sid", "entry_dt", "hold_days", "pnl_usd", "return_pct", "is_win", "campaign_result", "exit_type",
    "SS", "TR", "RS", "CMP", "VQS", "ADR", "DCR", "RMV", "ORC", "GRP", "GRP_LABEL", "DDV",
    "PROK", "RUNOK",
    "STATE", "RUNNER", "SQZ", "MODE",
    "SS_bucket", "TR_bucket", "RS_bucket", "CMP_bucket", "VQS_bucket", "RMV_bucket",
    "ADR_bucket", "DCR_bucket", "ORC_bucket", "GRP_bucket", "DDV_bucket"
]


def build_research_rows(campaigns: Dict[Tuple[str, int, str, str, int], Campaign]) -> Tuple[List[Dict[str, Any]], List[List[Any]], Dict[str, Any]]:
    rejects: List[List[Any]] = [["issue", "symbol", "sid", "entry_dt", "details"]]
    stats = {
        "duplicate_campaign_keys": 0,
        "missing_feature_fields": 0,
        "unknown_state": 0,
        "unknown_sqz": 0,
        "unknown_mode": 0,
        "unknown_grp": 0,
        "unknown_ddv": 0,
        "unknown_prok": 0,
        "unknown_runok": 0,
        "pnl_return_sign_mismatch": 0,
        "campaigns_with_no_exits": 0,
        "dropped_incomplete_campaigns": 0,
    }

    out: List[Dict[str, Any]] = []

    for key, c in campaigns.items():
        kv = c.entry_payload

        def need(k: str) -> Optional[str]:
            v = kv.get(k)
            return v.strip() if isinstance(v, str) and v.strip() != "" else None

        sid = c.sid
        SS = to_float(need("SS"))
        TR = to_float(need("TR"))
        RS = to_float(need("RS"))
        CMP = to_float(need("CMP"))
        VQS = to_float(need("VQS"))
        ADR = to_float(need("ADR"))
        DCR = to_float(need("DCR"))
        RMV = to_float(need("RMV"))
        ORC = to_float(need("Orc") or need("ORC"))
        GRP = norm_grp(need("GRP"))
        GRP_LABEL = grp_label(GRP)
        DDV = to_float(need("DDV"))

        PROK = to_float(need("PROK"))
        RUNOK = to_float(need("RUNOK"))

        raw_state = need("State") or need("STATE")
        STATE = norm_state(raw_state)

        if RUNOK is not None:
            RUNNER = int(RUNOK) == 1
        else:
            RUNNER = parse_runner_flag(raw_state)

        raw_sqz = need("Sqz") or need("SQZ")
        SQZ = norm_sqz(raw_sqz)
        MODE = norm_mode(need("MODE"))

        required_vals = {
            "SS": SS,
            "TR": TR,
            "RS": RS,
            "CMP": CMP,
            "VQS": VQS,
            "ADR": ADR,
            "DCR": DCR,
            "RMV": RMV,
            "ORC": ORC,
            "GRP": GRP,
            "DDV": DDV,
            "STATE": STATE,
            "MODE": MODE,
            "SQZ": SQZ,
            "PROK": PROK,
            "RUNOK": RUNOK,
        }
        missing = [k for k, v in required_vals.items() if v is None]
        if missing:
            stats["missing_feature_fields"] += 1
            rejects.append([
                "missing_entry_feature_fields",
                c.symbol,
                sid,
                c.entry_dt_str,
                ",".join(missing)
            ])
            continue

        if STATE is None:
            stats["unknown_state"] += 1
            rejects.append(["unknown_STATE", c.symbol, sid, c.entry_dt_str, kv.get("State") or kv.get("STATE")])
            continue

        if GRP is None:
            stats["unknown_grp"] += 1
            rejects.append(["unknown_GRP", c.symbol, sid, c.entry_dt_str, kv.get("GRP")])
            continue

        if DDV is None:
            stats["unknown_ddv"] += 1
            rejects.append(["unknown_DDV", c.symbol, sid, c.entry_dt_str, kv.get("DDV")])
            continue

        if raw_sqz and raw_sqz.strip() != "-" and SQZ is None:
            stats["unknown_sqz"] += 1
            rejects.append(["unknown_SQZ", c.symbol, sid, c.entry_dt_str, raw_sqz])
            continue

        raw_mode = need("MODE")
        if raw_mode and MODE == "Unknown" and raw_mode.strip().upper() not in {"B", "C"}:
            stats["unknown_mode"] += 1
            rejects.append(["unknown_MODE", c.symbol, sid, c.entry_dt_str, raw_mode])
            continue

        if not c.exits:
            stats["campaigns_with_no_exits"] += 1
            stats["dropped_incomplete_campaigns"] += 1
            rejects.append(["campaign_has_no_exits", c.symbol, sid, c.entry_dt_str, "missing hold_days/return_pct"])
            continue

        exits_sorted = sorted([e for e in c.exits if e.dt is not None], key=lambda x: x.dt)  # type: ignore
        last_exit_dt = exits_sorted[-1].dt if exits_sorted else None
        last_exit_label = exits_sorted[-1].comment if exits_sorted else ""

        hold_days = None
        if last_exit_dt:
            hold_days = (last_exit_dt - c.entry_dt).days

        pnl_usd = None
        wavg_exit_px = None
        w_sum = 0.0
        px_qty = 0.0

        targets_hit = set()
        for e in exits_sorted:
            if e.price is not None and e.qty is not None and e.qty != 0:
                w_sum += abs(e.qty)
                px_qty += e.price * abs(e.qty)
            for t in detect_targets(e.comment):
                targets_hit.add(t)

        if w_sum > 0:
            wavg_exit_px = px_qty / w_sum

        return_pct = None
        if wavg_exit_px is not None and c.entry_px:
            return_pct = (wavg_exit_px - c.entry_px) / c.entry_px

        if wavg_exit_px is not None and w_sum > 0:
            pnl_usd = (wavg_exit_px - c.entry_px) * w_sum

        pnl_usd = r2(pnl_usd)
        return_pct = r4(return_pct)

        is_win = return_pct is not None and return_pct > 0

        if pnl_usd is not None and return_pct is not None:
            if (pnl_usd > 0 and return_pct < 0) or (pnl_usd < 0 and return_pct > 0):
                stats["pnl_return_sign_mismatch"] += 1
                stats["dropped_incomplete_campaigns"] += 1
                rejects.append([
                    "pnl_return_sign_mismatch",
                    c.symbol,
                    sid,
                    c.entry_dt_str,
                    f"pnl_usd={pnl_usd};return_pct={return_pct}"
                ])
                continue

        if hold_days is None or return_pct is None:
            stats["dropped_incomplete_campaigns"] += 1
            missing_stage = []
            if hold_days is None:
                missing_stage.append("hold_days")
            if return_pct is None:
                missing_stage.append("return_pct")
            rejects.append([
                "missing_stage_fields",
                c.symbol,
                sid,
                c.entry_dt_str,
                ",".join(missing_stage)
            ])
            continue

        campaign_result = "Exit"
        if exits_sorted:
            lc = (last_exit_label or "").lower()
            if 3 in targets_hit:
                campaign_result = "Full Target Completion"
            elif 1 in targets_hit or 2 in targets_hit:
                campaign_result = "Partial Win"
            elif "expired" in lc or "time" in lc:
                campaign_result = "Time Exit"
            elif "stop" in lc:
                campaign_result = "Full Stop"

        exit_type = "Other"
        lc = (last_exit_label or "").lower()
        if 3 in targets_hit:
            exit_type = "T3"
        elif 2 in targets_hit:
            exit_type = "T2"
        elif 1 in targets_hit:
            exit_type = "T1"
        elif "expired" in lc or "time" in lc:
            exit_type = "TimeExit"
        elif "stop" in lc:
            exit_type = "Stop"

        ADR_bucket = bucket_value(ADR, [2, 4, 6, 8, 10], ["<2", "2–4", "4–6", "6–8", "8–10", ">10"])
        DCR_bucket = bucket_value(DCR, [30, 45, 60, 75], ["<30", "30–45", "45–60", "60–75", ">75"])
        CMP_bucket = bucket_value(CMP, [50, 65, 80, 90], ["<50", "50–65", "65–80", "80–90", ">90"])
        TR_bucket = bucket_value(TR, [55, 65, 75, 85], ["<55", "55–65", "65–75", "75–85", ">85"])
        RS_bucket = bucket_value(RS, [70, 80, 90], ["<70", "70–80", "80–90", ">90"])
        SS_bucket = bucket_value(SS, [55, 65, 75, 85], ["<55", "55–65", "65–75", "75–85", ">85"])
        VQS_bucket = bucket_value(VQS, [40, 60, 75, 90], ["<40", "40–60", "60–75", "75–90", ">90"])
        RMV_bucket = bucket_value(RMV, [40, 60, 75, 90], ["<40", "40–60", "60–75", "75–90", ">90"])

        ORC_bucket = bucket_value(
            ORC,
            [0.0, 0.25, 0.50, 0.75, 1.00],
            ["<0", "0–0.25", "0.25–0.50", "0.50–0.75", "0.75–1.00", ">1.00"]
        )

        GRP_bucket = bucket_grp(GRP)

        DDV_bucket = bucket_value(
            DDV,
            [20, 50, 75, 100, 200],
            ["<20", "20–50", "50–75", "75–100", "100–200", ">200"]
        )

        out.append({
            "symbol": c.symbol,
            "sid": sid,
            "entry_dt": c.entry_dt_str,
            "hold_days": hold_days,
            "pnl_usd": f"{pnl_usd:.2f}" if pnl_usd is not None else "",
            "return_pct": f"{return_pct:.4f}" if return_pct is not None else "",
            "is_win": is_win,
            "campaign_result": campaign_result,
            "exit_type": exit_type,
            "SS": SS,
            "TR": TR,
            "RS": RS,
            "CMP": CMP,
            "VQS": VQS,
            "ADR": ADR,
            "DCR": DCR,
            "RMV": RMV,
            "ORC": ORC,
            "GRP": GRP,
            "GRP_LABEL": GRP_LABEL,
            "DDV": DDV,
            "PROK": int(PROK) if PROK is not None else "",
            "RUNOK": int(RUNOK) if RUNOK is not None else "",
            "STATE": STATE,
            "RUNNER": RUNNER,
            "SQZ": SQZ,
            "MODE": MODE,
            "SS_bucket": SS_bucket,
            "TR_bucket": TR_bucket,
            "RS_bucket": RS_bucket,
            "CMP_bucket": CMP_bucket,
            "VQS_bucket": VQS_bucket,
            "RMV_bucket": RMV_bucket,
            "ADR_bucket": ADR_bucket,
            "DCR_bucket": DCR_bucket,
            "ORC_bucket": ORC_bucket,
            "GRP_bucket": GRP_bucket,
            "DDV_bucket": DDV_bucket,
        })

    return out, rejects, stats


def write_csv(path: str, rows: List[Dict[str, Any]], cols: List[str]) -> None:
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            w.writerow([r.get(c) for c in cols])


# -------------------------
# Main
# -------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="RAW TradingView List of Trades CSV (possibly malformed / merged).")
    ap.add_argument("--output", required=True, help="Output campaign-level research table CSV.")
    ap.add_argument("--rejects", default="rejects.csv", help="Rejects/anomalies CSV (default: rejects.csv).")
    args = ap.parse_args()

    with open(args.input, "r", encoding="utf-8-sig", errors="replace", newline="") as f:
        raw_text = f.read()

    corrected_lines = split_jammed_lines(raw_text)

    if corrected_lines:
        header_line = corrected_lines[0].lstrip("\ufeff")
        corrected_filtered = [header_line]
        for ln in corrected_lines[1:]:
            l = ln.lstrip("\ufeff")
            if l == header_line:
                continue
            corrected_filtered.append(l)
        corrected_lines = corrected_filtered

    tv_rows, parse_stats = parse_tradingview_csv(corrected_lines)
    campaigns, camp_stats, camp_rejects = build_campaigns(tv_rows)
    research_rows, feature_rejects, feature_stats = build_research_rows(campaigns)

    write_csv(args.output, research_rows, OUTPUT_COLS)

    with open(args.rejects, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        for sheet in (camp_rejects, [["----"]], feature_rejects):
            for r in sheet:
                w.writerow(r)

    report = {
        "raw_rows_after_correction_lines": parse_stats.get("raw_lines_after_correction"),
        "parsed_rows": parse_stats.get("parsed_rows"),
        "skipped_blank_rows": parse_stats.get("skipped_blank_rows"),
        "skipped_bad_column_rows": parse_stats.get("skipped_bad_column_rows"),
        "trimmed_wide_rows": parse_stats.get("trimmed_wide_rows"),
        "skipped_header_duplicates": parse_stats.get("skipped_header_duplicates"),
        "missing_required_cols": parse_stats.get("missing_required_cols"),
        "entries": camp_stats.get("entries"),
        "exits": camp_stats.get("exits"),
        "campaigns": camp_stats.get("campaigns"),
        "entry_payload_missing": camp_stats.get("entry_payload_missing"),
        "entry_missing_trade_no": camp_stats.get("entry_missing_trade_no"),
        "exit_missing_trade_no": camp_stats.get("exit_missing_trade_no"),
        "exit_symbol_mismatch": camp_stats.get("exit_symbol_mismatch"),
        "exit_without_campaign": camp_stats.get("exit_without_campaign"),
        "campaigns_without_exits_after_linkage": camp_stats.get("campaigns_without_exits_after_linkage"),
        "campaigns_with_no_exits": feature_stats.get("campaigns_with_no_exits"),
        "dropped_incomplete_campaigns": feature_stats.get("dropped_incomplete_campaigns"),
        "pnl_return_sign_mismatch": feature_stats.get("pnl_return_sign_mismatch"),
        "missing_feature_fields_campaigns": feature_stats.get("missing_feature_fields"),
        "unknown_state_campaigns": feature_stats.get("unknown_state"),
        "unknown_sqz_campaigns": feature_stats.get("unknown_sqz"),
        "unknown_mode_campaigns": feature_stats.get("unknown_mode"),
        "unknown_grp_campaigns": feature_stats.get("unknown_grp"),
        "unknown_ddv_campaigns": feature_stats.get("unknown_ddv"),
        "unknown_prok_campaigns": feature_stats.get("unknown_prok"),
        "unknown_runok_campaigns": feature_stats.get("unknown_runok"),
        "output_rows": len(research_rows),
        "output_file": args.output,
        "rejects_file": args.rejects,
    }
    print(report)


if __name__ == "__main__":
    main()