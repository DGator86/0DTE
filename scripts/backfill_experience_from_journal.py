#!/usr/bin/env python3
"""Backfill SPY-DER experience packets from the 0DTE shadow journal.

The live shadow loop publishes MarketPackets into
``/var/lib/zerodte/spyder_experience`` going forward, but older sessions only
exist in ``shadow.db``. Dojo reads the SPY-DER inbox and sees 0 sessions until
those rows are converted.

Usage (on the VPS)::

    sudo -u zerodte /opt/zerodte/venv/bin/python \\
        /opt/zerodte/scripts/backfill_experience_from_journal.py \\
        --db /var/lib/zerodte/shadow.db \\
        --root /var/lib/zerodte/spyder_experience

    sudo bash /opt/zerodte/deploy/ops/sync-experience-to-spyder.sh
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import date
from pathlib import Path
from typing import Any, Optional

# Allow `python scripts/...` from repo root without installing the package.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from integrations.spy_der.contracts import (  # noqa: E402
    build_market_packet,
    build_outcome_packet,
    candidate_view_to_dict,
)
from integrations.spy_der.market_publisher import (  # noqa: E402
    experience_root,
    publish_market_packet,
)
from integrations.spy_der.outcome_publisher import publish_outcome_packet  # noqa: E402


def _parse_json(value: Any) -> dict[str, Any]:
    if value is None or value == "":
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            raw = json.loads(value)
        except (TypeError, ValueError):
            return {}
        return raw if isinstance(raw, dict) else {}
    return {}


def _candidate_from_row(row: dict[str, Any]) -> Optional[dict[str, Any]]:
    signals = _parse_json(row.get("signals_json"))
    family = (
        row.get("selected_family")
        or signals.get("v2_top_family")
        or signals.get("legacy_top_family")
        or signals.get("policy_structure")
        or "unknown"
    )
    cid = (
        signals.get("legacy_top_candidate_id")
        or signals.get("v2_top_candidate_id")
        or row.get("snapshot_id")
    )
    if not cid and not row.get("candidate_present"):
        return None
    cid = str(cid or f"journal-{row.get('id', 'x')}")
    direction = str(
        row.get("regime_direction")
        or signals.get("policy_direction")
        or "both"
    )
    return candidate_view_to_dict(
        candidate_id=cid,
        family=str(family),
        direction=direction,
        maximum_loss=row.get("max_loss") or 1,
        mid_price=row.get("credit"),
        utility=row.get("ev_per_risk") if row.get("ev_per_risk") is not None else row.get("ev"),
        session_date=row.get("session_date"),
    )


def _hard_vetoes(row: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for key in ("gate_failed", "veto_reasons"):
        raw = row.get(key)
        if isinstance(raw, list):
            out.extend(str(x) for x in raw if x)
        elif isinstance(raw, str) and raw.strip():
            try:
                parsed = json.loads(raw)
            except (TypeError, ValueError):
                out.append(raw)
            else:
                if isinstance(parsed, list):
                    out.extend(str(x) for x in parsed if x)
                elif parsed:
                    out.append(str(parsed))
    # Deduplicate, preserve order.
    seen: set[str] = set()
    uniq: list[str] = []
    for item in out:
        if item not in seen:
            seen.add(item)
            uniq.append(item)
    return uniq


def _forecast_from_row(row: dict[str, Any]) -> dict[str, Any]:
    signals = _parse_json(row.get("signals_json"))
    keys = (
        "v2_fc_expected_return_30m",
        "v2_fc_p_up_30m",
        "v2_fc_uncertainty",
        "v2_fc_data_quality",
        "policy_confidence",
        "policy_action",
        "regime_bias_value",
    )
    return {k: signals[k] for k in keys if k in signals}


def row_to_packets(row: dict[str, Any]) -> tuple[dict[str, Any], Optional[dict[str, Any]]]:
    snap_id = str(row.get("snapshot_id") or f"journal-{row['id']}")
    session = str(row["session_date"])
    spot = row.get("spot")
    if spot is None:
        raise ValueError(f"row {row.get('id')} missing spot")
    cand = _candidate_from_row(row)
    candidates = [cand] if cand is not None else []
    signals = _parse_json(row.get("signals_json"))
    try:
        fu = float(signals.get("v2_fc_uncertainty") or 0.0)
    except (TypeError, ValueError):
        fu = 0.0
    try:
        dq = float(signals.get("v2_fc_data_quality") or 1.0)
    except (TypeError, ValueError):
        dq = 1.0
    market = build_market_packet(
        snapshot_id=snap_id,
        session_date=session,
        symbol="SPY",
        underlying_price=spot,
        data_quality=dq,
        forecast_uncertainty=fu,
        hard_vetoes=_hard_vetoes(row),
        forecast=_forecast_from_row(row),
        candidates=candidates,
        generated_at=row.get("ts"),
    )
    outcome = None
    if int(row.get("settled") or 0) == 1:
        action = str(row.get("decision") or "UNKNOWN")
        outcome = build_outcome_packet(
            snapshot_id=snap_id,
            session_date=session,
            symbol="SPY",
            candidate_id=candidates[0]["candidate_id"] if candidates else None,
            action=action,
            realized_pnl=row.get("realized_pnl"),
            settled=True,
            labels={
                "family": row.get("selected_family"),
                "direction": row.get("regime_direction"),
                "was_traded": row.get("was_traded"),
                "source": "shadow_journal_backfill",
            },
            settled_at=None,
        )
    return market, outcome


def fetch_rows(db_path: Path, *, recent_days: int = 0) -> list[dict[str, Any]]:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        sql = "SELECT * FROM evaluations WHERE snapshot_id IS NOT NULL AND snapshot_id != ''"
        if recent_days > 0:
            sql += (
                " AND session_date >= date("
                "(SELECT MAX(session_date) FROM evaluations), ?"
                ")"
            )
            rows = conn.execute(sql, (f"-{recent_days - 1} day",)).fetchall()
        else:
            rows = conn.execute(sql).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def backfill(
    *,
    db_path: Path,
    root: Path,
    stride: int = 1,
    recent_days: int = 0,
    dry_run: bool = False,
) -> dict[str, Any]:
    rows = fetch_rows(db_path, recent_days=recent_days)
    if stride > 1:
        rows = rows[::stride]
    sessions: set[str] = set()
    n_market = 0
    n_outcome = 0
    n_skip = 0
    for row in rows:
        try:
            market, outcome = row_to_packets(row)
        except (KeyError, ValueError, TypeError):
            n_skip += 1
            continue
        sessions.add(str(market["session_date"]))
        if dry_run:
            n_market += 1
            if outcome is not None:
                n_outcome += 1
            continue
        publish_market_packet(market, root=root)
        n_market += 1
        if outcome is not None:
            publish_outcome_packet(outcome, root=root)
            n_outcome += 1
    return {
        "rows_seen": len(rows),
        "markets_written": n_market,
        "outcomes_written": n_outcome,
        "skipped": n_skip,
        "sessions": sorted(sessions),
        "n_sessions": len(sessions),
        "root": str(root),
        "dry_run": dry_run,
    }


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", required=True, help="path to shadow.db")
    ap.add_argument(
        "--root",
        default="",
        help="experience outbox root (default ZERODTE_SPYDER_EXPERIENCE_ROOT)",
    )
    ap.add_argument(
        "--stride",
        type=int,
        default=1,
        help="keep every Nth row (default 1 = all). Use 5–10 for a thinner tape.",
    )
    ap.add_argument(
        "--recent-days",
        type=int,
        default=0,
        help="only backfill the last N calendar sessions (0 = all)",
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    db_path = Path(args.db)
    if not db_path.is_file():
        print(f"journal not found: {db_path}", file=sys.stderr)
        return 2
    root = experience_root(args.root or None)
    if not args.dry_run:
        (root / "snapshots").mkdir(parents=True, exist_ok=True)
        (root / "outcomes").mkdir(parents=True, exist_ok=True)

    result = backfill(
        db_path=db_path,
        root=root,
        stride=max(1, int(args.stride)),
        recent_days=max(0, int(args.recent_days)),
        dry_run=bool(args.dry_run),
    )
    print(json.dumps(result, indent=2))
    if result["n_sessions"] < 3:
        print(
            "WARN: fewer than 3 sessions — Dojo recorded phase may still skip",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
