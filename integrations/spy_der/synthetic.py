"""SyntheticUniverseProvider — wrap 0DTE synthetic generators as MarketPackets.

Exposes matrix_universe / synthetic_world through the contract only.
No Dojo / learning / AI evaluation logic.
"""
from __future__ import annotations

from collections.abc import Iterable
from datetime import date
from typing import Any, Optional, Protocol

from integrations.spy_der.contracts import build_market_packet, candidate_view_to_dict


class UniverseSpecLike(Protocol):
    @property
    def universe_id(self) -> str: ...

    @property
    def start_archetype(self) -> str: ...


class SyntheticUniverseProvider:
    """Generate MarketPacket streams from a UniverseSpec / MarkovWorldFeed."""

    def __init__(self, *, symbol: str = "SPY", max_ticks: Optional[int] = None) -> None:
        self.symbol = symbol
        self.max_ticks = max_ticks

    def generate(self, specification: Any) -> Iterable[dict[str, Any]]:
        """Yield MarketPackets for each tick in the synthetic universe.

        ``specification`` may be a ``matrix_universe.UniverseSpec`` or any
        object with ``universe_id`` / archetype fields that MarkovWorldFeed
        accepts.
        """
        from matrix_universe import MarkovWorldFeed, UniverseSpec

        if isinstance(specification, UniverseSpec):
            spec = specification
        else:
            # Duck-typed minimal construction for protocol consumers.
            spec = UniverseSpec(
                universe_id=str(getattr(specification, "universe_id", "synth")),
                seed=int(getattr(specification, "seed", 1)),
                days=int(getattr(specification, "days", 1)),
                start_archetype=str(
                    getattr(specification, "start_archetype", None)
                    or getattr(specification, "archetype", "compression")
                ),
                persistence_tilt=float(getattr(specification, "persistence_tilt", 1.0)),
                vol_mult=float(getattr(specification, "vol_mult", 1.0)),
                gap_mult=float(getattr(specification, "gap_mult", 1.0)),
                tick_stride=int(getattr(specification, "tick_stride", 1)),
                base_spot=float(getattr(specification, "base_spot", 600.0)),
                generation=int(getattr(specification, "generation", 0)),
                transition_jitter=float(getattr(specification, "transition_jitter", 0.0)),
            )

        feed = MarkovWorldFeed(spec)
        count = 0
        for ts in feed.timestamps():
            if self.max_ticks is not None and count >= self.max_ticks:
                break
            snap = feed.snapshot(ts)
            if snap is None:
                continue
            market = snap.market
            session = market.now.date() if market.now is not None else date.today()
            snap_id = (
                f"{spec.universe_id}-{session.isoformat()}-"
                f"{count:05d}"
            )
            yield build_market_packet(
                snapshot_id=snap_id,
                session_date=session,
                symbol=self.symbol,
                underlying_price=float(market.spot),
                data_quality=1.0,
                forecast_uncertainty=0.5,
                hard_vetoes=(),
                forecast={
                    "archetype": getattr(spec, "start_archetype", ""),
                    "universe_id": spec.universe_id,
                    "net_gex": float(market.net_gex),
                    "spot": float(market.spot),
                },
                candidates=[
                    candidate_view_to_dict(
                        candidate_id=f"{snap_id}-placeholder",
                        family="unknown",
                        direction="both",
                        maximum_loss=1,
                        session_date=session,
                    )
                ],
                track_record={},
                generated_at=market.now,
                risk_max_size_scalar=1.0,
            )
            count += 1


def packets_from_coupled_feed(
    *,
    days: int = 1,
    seed: int = 7,
    symbol: str = "SPY",
    max_ticks: Optional[int] = None,
) -> list[dict[str, Any]]:
    """Convenience: CoupledSyntheticFeed → MarketPacket list (offline demos)."""
    from synthetic_world import CoupledSyntheticFeed, WorldConfig

    feed = CoupledSyntheticFeed(WorldConfig(days=days, seed=seed))
    out: list[dict[str, Any]] = []
    for i, ts in enumerate(feed.timestamps()):
        if max_ticks is not None and i >= max_ticks:
            break
        snap = feed.snapshot(ts)
        if snap is None:
            continue
        market = snap.market
        session = market.now.date() if market.now is not None else date.today()
        snap_id = f"coupled-{seed}-{session.isoformat()}-{i:05d}"
        out.append(
            build_market_packet(
                snapshot_id=snap_id,
                session_date=session,
                symbol=symbol,
                underlying_price=float(market.spot),
                data_quality=1.0,
                forecast_uncertainty=0.5,
                forecast={"source": "coupled_synthetic", "seed": seed},
                candidates=[],
                generated_at=market.now,
            )
        )
    return out
