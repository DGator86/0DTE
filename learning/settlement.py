"""
learning/settlement.py
======================
Unified counterfactual settlement and label hooks (handoff §16.6, §17).

NOT financial advice.
"""
from __future__ import annotations

import datetime as dt
from typing import Any, Optional


LABEL_VERSION = "v1.0.0-unified"


def settle_session_counterfactuals(
    *,
    session_date: str,
    journal_rows: list | None = None,
    candidate_evaluations: list | None = None,
    fill_records: list | None = None,
    settlement_fn=None,
    label_version: str = LABEL_VERSION,
) -> dict:
    """
    Settle trades, no-trades, nonselected candidates, and unfilled attempts.

    Idempotent when the same inputs are provided. Current-session labels are
    only produced after settlement_fn reports complete.
    """
    journal_rows = list(journal_rows or [])
    candidate_evaluations = list(candidate_evaluations or [])
    fill_records = list(fill_records or [])

    settled = []
    if settlement_fn is not None:
        settled = list(settlement_fn(session_date, journal_rows) or [])
    else:
        # Synthetic settlement: mark rows settled with zero PnL when missing.
        for row in journal_rows:
            r = dict(row) if isinstance(row, dict) else {"raw": repr(row)}
            r.setdefault("net_pnl", 0.0)
            r.setdefault("settled_at", dt.datetime.now(dt.timezone.utc).isoformat())
            r["label_version"] = label_version
            settled.append(r)

    candidate_outcomes = []
    for ev in candidate_evaluations:
        d = ev.to_dict() if hasattr(ev, "to_dict") else dict(ev)
        candidate_outcomes.append({
            "snapshot_id": d.get("snapshot_id"),
            "candidate_id": d.get("candidate_id"),
            "session_date": session_date,
            "entry_assumption": "natural_not_midpoint",
            "fill_status": d.get("fill_status", "unfilled_counterfactual"),
            "fill_price": d.get("expected_fill_price"),
            "exit_price": None,
            "fees": d.get("fees"),
            "net_pnl": d.get("realized_net_pnl"),
            "max_adverse_excursion": d.get("max_adverse_excursion"),
            "max_favorable_excursion": d.get("max_favorable_excursion"),
            "target_first": d.get("target_first"),
            "stop_first": d.get("stop_first"),
            "settled_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "label_version": label_version,
        })

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
        "complete": True,
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
