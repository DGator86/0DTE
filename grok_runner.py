"""Entrypoint that adds the Grok 4.5 paper trader to the existing shadow loop.

The underlying ShadowRunner, market feeds, Legacy/V2/V3 processing, recorder,
notifier, dashboard state, and deterministic PaperBroker remain unchanged.  A
small wrapper intercepts PaperBroker.on_tick, asks the Grok coordinator for at
most one validated ``grok`` paper intent, and then hands control back to the
existing broker.
"""
from __future__ import annotations

import os

from grok_trader import GrokCoordinator, register_grok_track

# PaperBroker reads PAPER_TRACKS during construction, so register before the
# ShadowRunner imports/constructs it.
register_grok_track()

from paper_broker import PaperConfig
from risk_manager import RiskConfig
from shadow_runner import ShadowRunner, _build_parser


def attach_grok(runner: ShadowRunner, *, db_path: str) -> GrokCoordinator:
    state_dir = os.path.dirname(os.path.abspath(db_path))
    coordinator = GrokCoordinator.from_env(
        broker=runner._paper,
        symbol=runner.symbol,
        state_dir=state_dir,
    )
    original_on_tick = runner._paper.on_tick

    def on_tick_with_grok(now, result):
        outcome = coordinator.on_tick(now, result)
        if outcome.paper_intent is not None:
            result.paper_intents.append(outcome.paper_intent)
        events = outcome.events + original_on_tick(now, result)
        if outcome.paper_intent is not None:
            plan = dict(outcome.paper_intent.get("grok_plan") or {})
            for pos in runner._paper.open_positions:
                if runner._paper._track_of(pos) == "grok" and "grok_plan" not in pos.entry_ctx:
                    pos.entry_ctx["grok_plan"] = plan
        coordinator.audit.persist_open_positions(runner._paper, now)
        return events

    runner._paper.on_tick = on_tick_with_grok
    runner._grok = coordinator
    return coordinator


def main() -> None:
    args = _build_parser().parse_args()
    risk_cfg = None
    if args.max_loss > 0 or args.max_positions > 0 or args.max_gamma > 0:
        risk_cfg = RiskConfig(
            daily_loss_limit=args.max_loss or float("inf"),
            max_open_positions=args.max_positions,
            max_portfolio_gamma=args.max_gamma or float("inf"),
        )
    runner = ShadowRunner(
        symbol=args.symbol,
        db_path=args.db,
        interval_s=args.interval,
        lookback_minutes=args.lookback,
        vix9d=args.vix9d,
        vix=args.vix,
        vix3m=args.vix3m,
        vvix=args.vvix,
        vvix_baseline=args.vvix_baseline,
        risk_cfg=risk_cfg,
        paper_db=args.paper_db,
        paper_cfg=PaperConfig(starting_cash=args.paper_cash),
        live_state_path=args.live_state,
        record_dir=args.record_dir,
        ras_exit=args.ras_exit,
        champion_path=args.champion_path,
        policy_mode=args.policy_mode,
        prediction_db=args.prediction_db,
        use_legacy_directional_tilt=args.use_legacy_directional_tilt,
        enable_v2_parallel=args.enable_v2_parallel,
    )
    attach_grok(runner, db_path=args.db)
    if args.report:
        runner.report()
    elif args.paper_report:
        runner._paper.print_report()
    elif args.settle:
        runner.settle_date(args.settle)
    else:
        runner.run()


if __name__ == "__main__":
    main()
