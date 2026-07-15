"""
prediction/track_parity.py
==========================
Apples-to-apples helpers for legacy / V2 / V3 paper tracks.

Links paper fills back to prediction_store (fill_records + candidate_outcomes)
and builds validation/learning summaries by fill_track.
"""
from __future__ import annotations

import json
import logging
import sqlite3
from typing import Optional

log = logging.getLogger("track_parity")

TRACKS = ("legacy", "v2", "v3")


def family_to_structure(family: Optional[str]) -> Optional[str]:
    """Invert STRUCTURE_TO_FAMILIES for Part 3 → Track A decide()."""
    if not family:
        return None
    try:
        from spread_selector import STRUCTURE_TO_FAMILIES
    except Exception:
        return None
    fam = str(family)
    for code, fams in STRUCTURE_TO_FAMILIES.items():
        if fam in fams:
            return code
    return None


def flatten_part3_signals(part3: Optional[dict], signals: dict) -> None:
    """Write Part 3 decision fields into signals_json (in-place)."""
    if not part3 or not isinstance(signals, dict):
        return
    ds = part3.get("decision_summary") or {}
    ranking = part3.get("ranking") or {}
    if not ds and not ranking:
        return
    mapping = {
        "v3_action": ds.get("action"),
        "v3_statistical_action": ds.get("statistical_action") or ds.get("action"),
        "v3_selected_candidate_id": ds.get("selected_candidate_id"),
        "v3_family": ds.get("family") or ranking.get("top_family"),
        "v3_direction": ds.get("direction"),
        "v3_expected_order_value": ds.get("expected_order_value"),
        "v3_candidate_utility": ds.get("candidate_utility"),
        "v3_p_positive_utility": ds.get("p_positive_utility"),
        "v3_fill_probability": ds.get("fill_probability"),
        "v3_uncertainty": ds.get("uncertainty"),
        "v3_ood_score": ds.get("ood_score"),
        "v3_top_candidate_id": ranking.get("top_candidate_id"),
        "v3_top_score_margin": ranking.get("top_score_margin"),
    }
    for k, v in mapping.items():
        if v is None:
            continue
        if isinstance(v, bool):
            signals[k] = v
        elif isinstance(v, (int, float)):
            signals[k] = float(v)
        else:
            signals[k] = str(v)
    hard = ds.get("hard_vetoes") or []
    if hard:
        signals["v3_hard_veto_count"] = float(len(hard))


def paper_track_summary(paper_db_path: str) -> dict:
    """Aggregate closed paper trades by fill_track for validation/learning."""
    out = {
        t: {"trades": 0, "wins": 0, "total_pnl": 0.0, "win_rate": 0.0}
        for t in TRACKS
    }
    if not paper_db_path:
        return {"by_track": out, "note": "no paper database configured"}
    try:
        conn = sqlite3.connect(f"file:{paper_db_path}?mode=ro", uri=True)
    except sqlite3.Error as exc:
        return {"by_track": out, "note": f"paper db unavailable: {exc}"}
    try:
        try:
            rows = conn.execute(
                "SELECT pnl_dollars, entry_ctx FROM paper_trades"
            ).fetchall()
        except sqlite3.Error:
            return {"by_track": out, "note": "paper_trades table not found"}
    finally:
        conn.close()

    for pnl, ctx_json in rows:
        track = "legacy"
        if ctx_json:
            try:
                ctx = json.loads(ctx_json)
                t = str(ctx.get("fill_track") or "legacy").lower()
                if t in out:
                    track = t
            except (json.JSONDecodeError, TypeError, AttributeError):
                track = "legacy"
        p = float(pnl or 0.0)
        out[track]["trades"] += 1
        out[track]["total_pnl"] = round(out[track]["total_pnl"] + p, 2)
        if p > 0:
            out[track]["wins"] += 1
    for t, bt in out.items():
        n = bt["trades"]
        bt["win_rate"] = round(bt["wins"] / n, 4) if n else 0.0
        bt.pop("wins", None)
    return {
        "by_track": out,
        "tracks_with_trades": sum(1 for t in TRACKS if out[t]["trades"] > 0),
        "total_trades": sum(out[t]["trades"] for t in TRACKS),
    }


def record_paper_fill(store, *, pos, snapshot_id: str, track: str,
                      mode: str = "shadow") -> None:
    """Best-effort FillRecord on paper entry (source=paper)."""
    if store is None or not snapshot_id:
        return
    try:
        from execution.fill_records import FillRecord
        ctx = pos.entry_ctx or {}
        cid = (ctx.get("candidate_id") or ctx.get("v3_selected_candidate_id")
               or getattr(pos, "id", "unknown"))
        mid = float(ctx.get("credit_mid") if ctx.get("credit_mid") is not None
                    else pos.entry_credit)
        rec = FillRecord(
            fill_record_id=f"paper:{pos.id}",
            snapshot_id=str(snapshot_id),
            candidate_id=str(cid),
            session_date=str(pos.opened_at.date()) if hasattr(pos.opened_at, "date")
            else str(pos.opened_at)[:10],
            decision_ts=pos.opened_at.isoformat(),
            submitted_ts=pos.opened_at.isoformat(),
            resolved_ts=pos.opened_at.isoformat(),
            symbol=str(ctx.get("symbol") or "SPY"),
            family=str(pos.family),
            side="credit" if pos.entry_credit >= 0 else "debit",
            n_legs=len(pos.legs or ()),
            limit_credit=float(pos.entry_credit),
            mid_credit_at_submit=mid,
            natural_credit_at_submit=float(pos.entry_credit),
            relative_spread=0.0,
            absolute_spread=0.0,
            option_price_scale=abs(mid) if mid else 1.0,
            quote_age_seconds=0.0,
            minutes_to_close=float(ctx.get("minutes_to_close") or 0.0),
            dominant_regime=ctx.get("regime"),
            filled=True,
            filled_quantity=int(pos.contracts),
            requested_quantity=int(pos.contracts),
            fill_credit=float(pos.entry_credit),
            fill_fraction=1.0,
            fill_fraction_raw=1.0,
            fill_fraction_clipped=1.0,
            source="paper",
            mode=mode,
            diagnostics={"fill_track": track, "paper_position_id": pos.id},
        )
        store.log_fill_record(rec)
    except Exception:
        log.exception("record_paper_fill failed for %s", getattr(pos, "id", "?"))


def settle_paper_outcome(store, *, pos, pnl_dollars: float, exit_reason: str,
                         closed_at) -> None:
    """Write candidate_outcomes from a closed paper trade."""
    if store is None:
        return
    ctx = pos.entry_ctx or {}
    cid = ctx.get("candidate_id") or ctx.get("v3_selected_candidate_id")
    if not cid:
        return
    try:
        store.log_candidate_outcome(str(cid), {
            "settled": 1,
            "pnl_mid": float(pnl_dollars),
            "pnl_policy": float(pnl_dollars),
            "target_hit": 1 if exit_reason == "target" else 0,
            "stop_hit": 1 if exit_reason == "stop" else 0,
            "first_event": str(exit_reason),
            "fill_track": ctx.get("fill_track"),
            "snapshot_id": ctx.get("snapshot_id"),
            "paper_position_id": pos.id,
            "closed_at": closed_at.isoformat() if hasattr(closed_at, "isoformat")
            else str(closed_at),
            "contracts": pos.contracts,
            "family": pos.family,
        })
    except Exception:
        log.exception("settle_paper_outcome failed for %s", cid)
