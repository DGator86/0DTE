"""
learning/settlement.py
======================
Unified counterfactual settlement and label hooks (handoff §16.6, §17).

Unknown outcomes must remain unresolved. They must NEVER be converted into
zero-return training examples.

NOT financial advice.
"""
from __future__ import annotations

import datetime as dt
from typing import Any, Optional


LABEL_VERSION = "v1.0.0-unified"


class SettlementIncomplete(RuntimeError):
    """Settlement could not complete; outcomes must not enter learning."""


def settle_session_counterfactuals(
    *,
    session_date: str,
    journal_rows: list | None = None,
    candidate_evaluations: list | None = None,
    fill_records: list | None = None,
    settlement_fn=None,
    label_version: str = LABEL_VERSION,
    allow_incomplete: bool = False,
) -> dict:
    """
    Settle trades, no-trades, nonselected candidates, and unfilled attempts.

    If settlement_fn is not provided, returns complete=False unless the caller
    explicitly opts into allow_incomplete for named test fixtures.
    Missing P&L is NEVER coerced to 0.0.
    """
    journal_rows = list(journal_rows or [])
    candidate_evaluations = list(candidate_evaluations or [])
    fill_records = list(fill_records or [])

    if settlement_fn is None:
        if not allow_incomplete:
            unfilled = [
                dict(f) if isinstance(f, dict) else {"raw": repr(f)}
                for f in fill_records
                if (f.get("fill_status") if isinstance(f, dict)
                    else getattr(f, "fill_status", None)) in (
                    None, "unfilled", "rejected", "expired")
            ]
            return {
                "session_date": session_date,
                "settled_journal": [],
                "candidate_outcomes": [],
                "unfilled_attempts": unfilled,
                "label_version": label_version,
                "complete": False,
                "reason": "settlement_fn_required",
            }
        # Explicit test-fixture path only — still does NOT invent PnL=0.
        settled = []
        for row in journal_rows:
            r = dict(row) if isinstance(row, dict) else {"raw": repr(row)}
            if "net_pnl" not in r or r.get("net_pnl") is None:
                r["settlement_status"] = "unresolved"
            else:
                r["settled_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
                r["settlement_status"] = "settled"
            r["label_version"] = label_version
            settled.append(r)
        complete = all(
            s.get("settlement_status") == "settled" for s in settled
        ) if settled else False
    else:
        settled = list(settlement_fn(session_date, journal_rows) or [])
        # Require every settled row to carry an explicit net_pnl (may be 0.0
        # only when the settlement engine computed it).
        unresolved = [
            r for r in settled
            if not isinstance(r, dict) or r.get("net_pnl") is None
        ]
        complete = len(unresolved) == 0
        if not complete and not allow_incomplete:
            raise SettlementIncomplete(
                f"session {session_date}: {len(unresolved)} rows lack net_pnl")

    candidate_outcomes = []
    for ev in candidate_evaluations:
        d = ev.to_dict() if hasattr(ev, "to_dict") else dict(ev)
        pnl = d.get("realized_net_pnl")
        candidate_outcomes.append({
            "snapshot_id": d.get("snapshot_id"),
            "candidate_id": d.get("candidate_id"),
            "session_date": session_date,
            "entry_assumption": "natural_not_midpoint",
            "fill_status": d.get("fill_status", "unfilled_counterfactual"),
            "fill_price": d.get("expected_fill_price"),
            "exit_price": d.get("exit_price"),
            "fees": d.get("fees"),
            "net_pnl": pnl,  # may be None — unresolved
            "max_adverse_excursion": d.get("max_adverse_excursion"),
            "max_favorable_excursion": d.get("max_favorable_excursion"),
            "target_first": d.get("target_first"),
            "stop_first": d.get("stop_first"),
            "settled_at": (
                dt.datetime.now(dt.timezone.utc).isoformat()
                if pnl is not None else None),
            "label_version": label_version,
            "settlement_status": (
                "settled" if pnl is not None else "unresolved"),
        })

    # Candidate outcomes with unresolved PnL keep complete=False for learning.
    if any(o.get("net_pnl") is None for o in candidate_outcomes):
        if settlement_fn is not None and not allow_incomplete:
            complete = False

    unfilled = [
        dict(f) if isinstance(f, dict) else {"raw": repr(f)}
        for f in fill_records
        if (f.get("fill_status") if isinstance(f, dict)
            else getattr(f, "fill_status", None)) in (
            None, "unfilled", "rejected", "expired")
    ]

    return {
        "session_date": session_date,
        "settled_journal": settled,
        "candidate_outcomes": candidate_outcomes,
        "unfilled_attempts": unfilled,
        "label_version": label_version,
        "complete": bool(complete),
    }


def build_market_labels(
    *,
    observations: list,
    label_version: str = LABEL_VERSION,
) -> list:
    """Versioned market forecast labels (post-settlement only)."""
    out = []
    for obs in observations:
        d = dict(obs) if isinstance(obs, dict) else {}
        d["label_version"] = label_version
        out.append(d)
    return out
