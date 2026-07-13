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
    candidate_configs,
    feature_scores,
    fetch_prediction_for_snapshot,
    gex_variant_summary,
    journal_fetch,
    journal_max_id,
    journal_row,
    learning_runs,
    paper_summary,
    paper_trades_journal,
    promotions,
    ras_history,
    readiness_summary,
    report_summary,
    validation_report_by_id,
    validation_reports,
)
from dashboard.state import read_live_state
from market_calendar import market_status

ET = ZoneInfo("America/New_York")
STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(title="0DTE Observability", docs_url=None, redoc_url=None)
app.add_middleware(ReadOnlyMiddleware)
app.add_middleware(AuthMiddleware)

_config: dict = {}


def _configure(db: str, paper_db: str, live_state: str,
               configs_dir: str = "configs",
               prediction_db: str = "prediction_store.sqlite") -> None:
    _config["db"] = db
    _config["paper_db"] = paper_db
    _config["live_state"] = live_state
    _config["configs_dir"] = configs_dir
    _config["prediction_db"] = prediction_db


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
        return {
            "ts": None,
            "note": "No live tick yet — pipeline idle or waiting for market open",
            "market": market_status(),
        }
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
    return paper_summary(_config.get("paper_db", "paper.sqlite"))


@app.get("/api/trades")
async def api_trades(limit: int = 200):
    return paper_trades_journal(
        _config.get("paper_db", "paper.sqlite"),
        _config.get("live_state", "live_state.json"),
        limit=max(1, min(limit, 500)),
    )


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
# Adaptive-learning routes (Learning tab)                                     #
# --------------------------------------------------------------------------- #
@app.get("/api/learning")
async def api_learning(limit: int = Query(50, ge=1, le=200)):
    """Learning-cycle history with decoded diagnostics + trial summaries."""
    db = _config.get("db", "shadow.db")
    if not os.path.isfile(db):
        return {"runs": [], "note": "journal database not found"}
    return {"runs": learning_runs(db, limit=limit)}


@app.get("/api/candidates")
async def api_candidates(
    status: Optional[str] = Query(
        None,
        pattern="^(candidate|pending_review|promoted|rejected|archived)$"),
    limit: int = Query(50, ge=1, le=200),
):
    """Champion vs challenger configs: current champion (from the configs
    directory) plus the candidate_configs history."""
    db = _config.get("db", "shadow.db")
    champion = None
    champ_file = os.path.join(_config.get("configs_dir", "configs"),
                              "champion.json")
    if os.path.isfile(champ_file):
        try:
            import json as _json
            with open(champ_file, encoding="utf-8") as f:
                champion = _json.load(f)
        except (OSError, ValueError):
            champion = {"note": "champion.json unreadable"}
    if not os.path.isfile(db):
        return {"champion": champion, "candidates": [],
                "note": "journal database not found"}
    return {"champion": champion,
            "candidates": candidate_configs(db, status=status, limit=limit)}


@app.get("/api/promotions")
async def api_promotions(
    status: Optional[str] = Query(
        None, pattern="^(pending_review|approved|rejected)$"),
    limit: int = Query(50, ge=1, le=200),
):
    """Promotion queue with the rule-by-rule decision breakdown."""
    db = _config.get("db", "shadow.db")
    if not os.path.isfile(db):
        return {"promotions": [], "note": "journal database not found"}
    return {"promotions": promotions(db, status=status, limit=limit)}


@app.get("/api/feature-scores")
async def api_feature_scores(
    all_history: bool = Query(False),
    limit: int = Query(500, ge=1, le=2000),
):
    """Latest feature-lab scores (Pearson/Spearman/MI/permutation importance,
    stability, lifecycle status) per feature."""
    db = _config.get("db", "shadow.db")
    if not os.path.isfile(db):
        return {"features": [], "note": "journal database not found"}
    return {"features": feature_scores(db, latest_only=not all_history,
                                       limit=limit)}


@app.get("/api/drift")
async def api_drift(limit: int = Query(30, ge=1, le=200)):
    """Drift snapshots (validation_reports rows with report_type='drift')."""
    db = _config.get("db", "shadow.db")
    if not os.path.isfile(db):
        return {"reports": [], "note": "journal database not found"}
    return {"reports": validation_reports(db, report_type="drift", limit=limit)}


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
                        help="Directory holding champion.json (Learning tab)")
    parser.add_argument("--prediction-db", default=None,
                        help="PredictionStore SQLite (default: <db dir>/prediction_store.sqlite)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    from pathlib import Path
    pred_db = args.prediction_db or str(
        Path(args.db).with_name("prediction_store.sqlite"))

    _configure(args.db, args.paper_db, args.live_state, args.configs_dir,
               prediction_db=pred_db)

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
