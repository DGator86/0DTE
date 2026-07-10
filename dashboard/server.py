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
    journal_fetch,
    journal_max_id,
    journal_row,
    paper_summary,
    paper_trades_journal,
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


def _configure(db: str, paper_db: str, live_state: str) -> None:
    _config["db"] = db
    _config["paper_db"] = paper_db
    _config["live_state"] = live_state


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


@app.get("/api/validation")
async def api_validation(
    report_type: Optional[str] = Query(None, pattern="^(daily|weekly|feature_impact)$"),
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
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    _configure(args.db, args.paper_db, args.live_state)

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
