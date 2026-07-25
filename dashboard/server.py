"""
dashboard/server.py
===================
GET-only FastAPI observability server for the 0DTE pipeline.
"""
from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import os
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from dashboard.auth import AuthMiddleware, ReadOnlyMiddleware, get_dashboard_token
from dashboard.queries import (
    competition_view,
    enrich_paper_summary_with_live,
    fetch_prediction_for_snapshot,
    fetch_sigma_cone_journal,
    gex_variant_summary,
    journal_fetch,
    journal_max_id,
    journal_row,
    paper_summary,
    paper_trades_journal,
    ras_history,
    readiness_summary,
    report_summary,
    trade_insights,
    validation_report_by_id,
    validation_reports,
)
from dashboard.state import heartbeat_state, read_live_state
from market_calendar import market_status

ET = ZoneInfo("America/New_York")
STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(title="0DTE Observability", docs_url=None, redoc_url=None)
app.add_middleware(ReadOnlyMiddleware)
app.add_middleware(AuthMiddleware)

_config: dict = {}


def _configure(db: str, paper_db: str, live_state: str,
               configs_dir: str = "configs",
               prediction_db: str = "prediction_store.sqlite",
               spy_der_live_state: Optional[str] = None,
               spy_der_dojo_latest: Optional[str] = None) -> None:
    _config["db"] = db
    _config["paper_db"] = paper_db
    _config["live_state"] = live_state
    _config["configs_dir"] = configs_dir
    _config["prediction_db"] = prediction_db
    _config["spy_der_live_state"] = spy_der_live_state
    _config["spy_der_dojo_latest"] = spy_der_dojo_latest


@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
async def health():
    return {
        "ok": True,
        "read_only": True,
        "auth_configured": bool(get_dashboard_token()),
    }


@app.get("/api/market-status")
async def api_market_status():
    return market_status()


@app.get("/api/live")
async def api_live():
    path = _config.get("live_state", "live_state.json")
    data = read_live_state(path)
    if data is None:
        # Always return a valid live.v1 envelope so the SPA does not keep
        # rendering stale panels after a missing/rotated state file.
        return heartbeat_state(
            dt.datetime.now(ET),
            status="no_live_state",
            note="No live tick yet — pipeline idle or waiting for market open",
            market_status=market_status(),
        )
    return data


@app.get("/api/ticks")
async def api_ticks(
    session_date: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=500),
    since_id: int = Query(0, ge=0),
):
    db = _config.get("db", "shadow.db")
    if not os.path.isfile(db):
        return {"ticks": [], "note": "journal database not found"}
    if session_date is None:
        session_date = dt.datetime.now(ET).date().isoformat()
    ticks = journal_fetch(db, session_date=session_date, limit=limit, since_id=since_id)
    return {"session_date": session_date, "ticks": ticks}


@app.get("/api/ticks/{row_id}")
async def api_tick_row(row_id: int):
    db = _config.get("db", "shadow.db")
    row = journal_row(db, row_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Tick not found")
    return row


@app.get("/api/paper")
async def api_paper():
    summary = paper_summary(_config.get("paper_db", "paper.sqlite"))
    live = read_live_state(_config.get("live_state", "live_state.json"))
    live_paper = (live or {}).get("paper") if isinstance(live, dict) else None
    return enrich_paper_summary_with_live(summary, live_paper)


@app.get("/api/competition")
async def api_competition():
    """0DTE (deterministic system) vs SPY-DER (AI) head-to-head scoreboard."""
    summary = paper_summary(_config.get("paper_db", "paper.sqlite"))
    live = read_live_state(_config.get("live_state", "live_state.json"))
    live_paper = (live or {}).get("paper") if isinstance(live, dict) else None
    enriched = enrich_paper_summary_with_live(summary, live_paper)
    return competition_view(enriched)


@app.get("/api/trades")
async def api_trades(limit: int = 200):
    return paper_trades_journal(
        _config.get("paper_db", "paper.sqlite"),
        _config.get("live_state", "live_state.json"),
        limit=max(1, min(limit, 500)),
    )


@app.get("/api/trade-insights")
async def api_trade_insights(limit: int = Query(500, ge=1, le=2000)):
    """Trade-journal learning + validation: entry predictions (EV / PoP /
    gate score) held against realized outcomes, segment P&L attribution,
    exit-discipline audit, and ranked plain-language lessons."""
    return trade_insights(_config.get("paper_db", "paper.sqlite"), limit=limit)


@app.get("/api/ras")
async def api_ras(
    position_id: Optional[str] = Query(None),
    session_date: Optional[str] = Query(None),
    limit: int = Query(500, ge=1, le=2000),
):
    """Regime Alignment Score history: per-position score/action timeline
    with the full component breakdown for every evaluation."""
    db = _config.get("db", "shadow.db")
    if not os.path.isfile(db):
        return {"evaluations": [], "note": "journal database not found"}
    return {
        "position_id": position_id,
        "session_date": session_date,
        "evaluations": ras_history(db, position_id=position_id,
                                   session_date=session_date, limit=limit),
    }


@app.get("/api/report")
async def api_report():
    db = _config.get("db", "shadow.db")
    if not os.path.isfile(db):
        return {"note": "journal database not found"}
    return report_summary(db)


@app.get("/api/gex-variants")
async def api_gex_variants(
    session_date: Optional[str] = Query(None),
):
    """PR 9 — settled GEX variant comparison (corr vs P&L, sign disagreement)."""
    db = _config.get("db", "shadow.db")
    if not os.path.isfile(db):
        return {"note": "journal database not found"}
    return gex_variant_summary(db, session_date=session_date)


@app.get("/api/predictions")
async def api_predictions(
    snapshot_id: str = Query(..., min_length=1),
):
    """PR 4+ — PredictionBundle for a journal snapshot_id (read-only)."""
    return fetch_prediction_for_snapshot(
        snapshot_id,
        prediction_db=_config.get("prediction_db", "prediction_store.sqlite"),
        journal_db=_config.get("db", "shadow.db"),
    )


@app.get("/api/sigma-cones")
async def api_sigma_cones(
    session_date: Optional[str] = Query(None),
    timeframe: Optional[str] = Query(None),
    settled: Optional[bool] = Query(None),
    limit: int = Query(200, ge=1, le=1000),
):
    """MTF sigma-cone prediction journal + coverage vs realized spot."""
    return fetch_sigma_cone_journal(
        prediction_db=_config.get("prediction_db", "prediction_store.sqlite"),
        session_date=session_date,
        timeframe=timeframe,
        settled=settled,
        limit=limit,
    )


@app.get("/api/validation")
async def api_validation(
    report_type: Optional[str] = Query(
        None,
        pattern="^(daily|weekly|feature_impact|drift|promotion_candidate)$"),
    limit: int = Query(50, ge=1, le=200),
):
    """Validation report history (daily/weekly pipeline runs and
    feature-impact reports), newest first."""
    db = _config.get("db", "shadow.db")
    if not os.path.isfile(db):
        return {"reports": [], "note": "journal database not found"}
    return {"report_type": report_type,
            "reports": validation_reports(db, report_type=report_type, limit=limit)}


@app.get("/api/validation/{report_id}")
async def api_validation_report(report_id: int):
    db = _config.get("db", "shadow.db")
    report = validation_report_by_id(db, report_id) if os.path.isfile(db) else None
    if report is None:
        raise HTTPException(status_code=404, detail="Validation report not found")
    return report


# --------------------------------------------------------------------------- #
# SPY-DER dashboard adapter (Dojo / Learning — no AI internals)               #
# --------------------------------------------------------------------------- #
def _spyder_bundle() -> dict:
    from integrations.spy_der.dashboard_reader import read_dashboard_bundle
    return read_dashboard_bundle(
        live_path=_config.get("spy_der_live_state"),
        dojo_path=_config.get("spy_der_dojo_latest"),
    )


def _spyder_champion() -> object:
    from integrations.spy_der.champion_reader import resolve_champion_path
    import json as _json
    path = resolve_champion_path(None)
    if not path or not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as handle:
            return _json.load(handle)
    except (OSError, ValueError):
        return {"note": "champion.json unreadable", "path": path}


def _spyder_pending_reviews() -> list:
    """Read-only listing of SPY-DER pending_review JSON files (no promotion)."""
    import json as _json
    root = os.environ.get("SPY_DER_STATE_ROOT", "/var/lib/spy-der")
    pending_dir = Path(root) / "configs" / "pending_review"
    if not pending_dir.is_dir():
        # Also accept a single pending_review.json (legacy layout).
        single = Path(root) / "configs" / "promoted" / "pending_review.json"
        if single.is_file():
            try:
                with open(single, encoding="utf-8") as handle:
                    data = _json.load(handle)
                if isinstance(data, dict):
                    data.setdefault("status", "pending_review")
                    return [data]
            except (OSError, ValueError):
                return []
        return []
    out: list = []
    for path in sorted(pending_dir.glob("*.json")):
        try:
            with open(path, encoding="utf-8") as handle:
                data = _json.load(handle)
            if isinstance(data, dict):
                data.setdefault("status", "pending_review")
                out.append(data)
        except (OSError, ValueError):
            continue
    return out


@app.get("/api/dojo")
async def api_dojo(limit: int = Query(50, ge=1, le=200)):
    """SPY-DER Dojo latest report (``/var/lib/spy-der/reports/dojo/latest.json``)."""
    from integrations.spy_der.dashboard_reader import read_dojo_latest
    latest = read_dojo_latest(_config.get("spy_der_dojo_latest"))
    if latest.get("note") and "summary" not in latest and "report_date" not in latest:
        return {"reports": [], "note": latest.get("note")}
    report = dict(latest)
    report.setdefault("id", 1)
    report.setdefault("report_type", "dojo")
    report.setdefault("summary", latest.get("summary") or "")
    return {"reports": [report][:limit], "source": "spy-der"}


@app.get("/api/dojo/{report_id}")
async def api_dojo_report(report_id: int):
    data = await api_dojo(limit=1)
    reports = data.get("reports") or []
    if not reports:
        raise HTTPException(status_code=404, detail="Dojo report not found")
    report = reports[0]
    if int(report.get("id") or 1) != int(report_id):
        raise HTTPException(status_code=404, detail="Dojo report not found")
    return report


@app.get("/api/learning")
async def api_learning(limit: int = Query(50, ge=1, le=200)):
    """Learning status from SPY-DER dashboard contract (no ALE internals)."""
    bundle = _spyder_bundle()
    live = bundle.get("live") or {}
    dojo_status = bundle.get("dojo_status") or {}
    run = {
        "run_id": "spy-der",
        "mode": live.get("mode") or "shadow",
        "summary": dojo_status.get("summary") or live.get("rationale") or "",
        "diagnostics": [],
        "status": dojo_status.get("latest_status") or live.get("action") or "UNKNOWN",
        "source": "spy-der",
    }
    if live.get("note"):
        return {"runs": [], "note": live.get("note"), "source": "spy-der"}
    return {"runs": [run][:limit], "source": "spy-der", "live": live}


@app.get("/api/candidates")
async def api_candidates(
    status: Optional[str] = Query(
        None,
        pattern="^(candidate|pending_review|promoted|rejected|archived)$"),
    limit: int = Query(50, ge=1, le=200),
):
    """Champion from SPY-DER configs; challengers are not owned by 0DTE."""
    champion = _spyder_champion()
    pending = _spyder_pending_reviews()
    if status:
        pending = [p for p in pending if p.get("status") == status]
    return {
        "champion": champion,
        "candidates": pending[:limit],
        "source": "spy-der",
        "note": (
            None if champion is not None
            else "SPY-DER champion.json not found"
        ),
    }


@app.get("/api/promotions")
async def api_promotions(
    status: Optional[str] = Query(
        None, pattern="^(pending_review|approved|rejected)$"),
    limit: int = Query(50, ge=1, le=200),
):
    """Pending-review queue from SPY-DER configs (display only)."""
    promos = _spyder_pending_reviews()
    if status:
        promos = [p for p in promos if p.get("status") == status]
    return {"promotions": promos[:limit], "source": "spy-der"}


@app.get("/api/feature-scores")
async def api_feature_scores(
    all_history: bool = Query(False),
    limit: int = Query(500, ge=1, le=2000),
):
    """Feature lab moved to SPY-DER; 0DTE no longer owns scores."""
    del all_history, limit
    return {
        "features": [],
        "note": "feature scores are owned by SPY-DER (not available in 0DTE)",
        "source": "spy-der",
    }


@app.get("/api/drift")
async def api_drift(limit: int = Query(30, ge=1, le=200)):
    """Drift reports moved to SPY-DER."""
    del limit
    return {
        "reports": [],
        "note": "drift reports are owned by SPY-DER (not available in 0DTE)",
        "source": "spy-der",
    }


@app.get("/api/spy-der")
async def api_spy_der():
    """Thin SPY-DER dashboard bundle (live_state + dojo latest)."""
    return _spyder_bundle()


@app.get("/api/readiness")
async def api_readiness():
    db = _config.get("db", "shadow.db")
    if not os.path.isfile(db):
        return {"note": "journal database not found"}
    return readiness_summary(db, _config.get("paper_db", "paper.sqlite"))


@app.get("/api/stream")
async def api_stream():
    db = _config.get("db", "shadow.db")

    async def event_generator():
        last_id = journal_max_id(db) if os.path.isfile(db) else 0
        while True:
            await asyncio.sleep(5)
            if not os.path.isfile(db):
                continue
            current = journal_max_id(db)
            if current > last_id:
                last_id = current
                yield f"data: {{\"latest_id\": {current}}}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def main() -> None:
    parser = argparse.ArgumentParser(description="0DTE read-only observability dashboard")
    parser.add_argument("--db", default="shadow.db", help="Journal SQLite path")
    parser.add_argument("--paper-db", default="paper.sqlite", help="Paper trades SQLite path")
    parser.add_argument("--live-state", default="live_state.json", help="Live state JSON path")
    parser.add_argument("--configs-dir", default="configs",
                        help="Legacy configs dir (champion prefers SPY-DER path)")
    parser.add_argument("--spy-der-live-state", default=None,
                        help="SPY-DER live_state.json "
                             "(default: /var/lib/spy-der/live_state.json)")
    parser.add_argument("--spy-der-dojo-latest", default=None,
                        help="SPY-DER dojo latest.json "
                             "(default: /var/lib/spy-der/reports/dojo/latest.json)")
    parser.add_argument("--prediction-db", default=None,
                        help="PredictionStore SQLite (default: <db dir>/prediction_store.sqlite)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    from pathlib import Path
    pred_db = args.prediction_db or str(
        Path(args.db).with_name("prediction_store.sqlite"))

    _configure(args.db, args.paper_db, args.live_state, args.configs_dir,
               prediction_db=pred_db,
               spy_der_live_state=args.spy_der_live_state,
               spy_der_dojo_latest=args.spy_der_dojo_latest)

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
