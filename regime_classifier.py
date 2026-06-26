"""
regime_classifier.py
====================
Deterministic regime classifier with an adaptive ScaleBook.

The classifier answers: given the current market snapshot (GEX positioning,
volatility surface, momentum), what regime are we in, and what engines are
permitted? It does NOT make trade decisions — that is the decision_matrix's job.
This module just labels the market so the decision layer can route correctly.

ScaleBook tracks a rolling percentile for each feature so thresholds read
"above your own recent median" rather than fixed absolute values. This is the
adaptive-scales TODO from HANDOFF §4; the calibration here is the first step.

Engine IDs used by the decision layer:
  "premium_selector"   -> spread_selector (put/call credit, condors, flies)
  "directional_selector" -> debit spreads, long premium (not yet built, §6.7)
  "volatility_selector"  -> long vega / strangle (not yet built)
  "none"               -> stand aside
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Optional


# --------------------------------------------------------------------------- #
# ScaleBook — adaptive percentile tracker                                      #
# --------------------------------------------------------------------------- #
class ScaleBook:
    """
    Maintains a rolling window of observations per feature.
    percentile(name, value) returns 0..1 (where 1 = highest seen recently).
    Useful for converting absolute GEX / ADX / VIX values to relative ranks
    that remain stable across different market regimes.
    """

    def __init__(self, window: int = 60):
        self.window = window
        self._data: dict[str, deque] = {}

    def update(self, name: str, value: float) -> None:
        if name not in self._data:
            self._data[name] = deque(maxlen=self.window)
        if math.isfinite(value):
            self._data[name].append(value)

    def percentile(self, name: str, value: float) -> float:
        """Fraction of stored observations <= value. 0.5 = at the median."""
        buf = self._data.get(name)
        if not buf:
            return 0.5
        below = sum(1 for v in buf if v <= value)
        return below / len(buf)

    def rank(self, name: str, value: float) -> float:
        """Same as percentile but named more intuitively for ranks."""
        return self.percentile(name, value)

    def has(self, name: str) -> bool:
        return name in self._data and len(self._data[name]) > 0


# --------------------------------------------------------------------------- #
# Output types                                                                 #
# --------------------------------------------------------------------------- #
@dataclass
class RegimeState:
    dealer_regime: str        # "long_gamma" | "short_gamma" | "at_flip"
    vol_regime: str           # "low_vol" | "normal_vol" | "high_vol"
    momentum_regime: str      # "bull" | "bear" | "neutral"
    permitted_engines: list[str]
    veto_reasons: list[str]
    info_gain: float          # 0..1, spikes on regime flips
    details: dict             # raw scores for logging

    @property
    def is_premium_ok(self) -> bool:
        return "premium_selector" in self.permitted_engines

    @property
    def is_directional_ok(self) -> bool:
        return "directional_selector" in self.permitted_engines

    @property
    def all_blocked(self) -> bool:
        return not self.permitted_engines


@dataclass
class ClassifierConfig:
    # dealer regime
    flip_buffer_frac: float = 0.0012   # abs((spot-flip)/spot) < this -> at_flip
    min_gex_rank_premium: float = 0.55 # long_gamma but GEX rank too low -> no premium
    min_gex_for_long: float = 0.0      # net_gex must exceed this to be long (in $ units)

    # vol regime
    vix_low: float = 14.0
    vix_high: float = 20.0
    adx_low: float = 14.0
    adx_high: float = 24.0

    # momentum regime
    rsi_bull: float = 55.0
    rsi_bear: float = 45.0

    # catalyst hard stop
    block_on_catalyst: bool = True


# --------------------------------------------------------------------------- #
# Classifier                                                                   #
# --------------------------------------------------------------------------- #
class RegimeClassifier:
    def __init__(self, cfg: Optional[ClassifierConfig] = None,
                 scale_book: Optional[ScaleBook] = None):
        self.cfg = cfg or ClassifierConfig()
        self.scale_book = scale_book or ScaleBook()
        self._prior: Optional[RegimeState] = None

    def classify(self, snapshot: dict) -> RegimeState:
        """
        snapshot keys used (all optional / defaulted):
          spot, net_gex, gamma_flip, gex_pct_rank,
          vix, vix9d, vix3m, adx, rsi, has_catalyst
        """
        cfg = self.cfg
        sb = self.scale_book

        spot = float(snapshot.get("spot", 600.0))
        net_gex = float(snapshot.get("net_gex", 0.0))
        flip = float(snapshot.get("gamma_flip", spot))
        gex_rank = float(snapshot.get("gex_pct_rank", 0.5))
        vix = float(snapshot.get("vix", 15.0))
        vix9d = float(snapshot.get("vix9d", vix))
        vix3m = float(snapshot.get("vix3m", vix))
        adx = float(snapshot.get("adx", 15.0))
        rsi = float(snapshot.get("rsi", 50.0))
        has_catalyst = bool(snapshot.get("has_catalyst", False))

        # Update scale book
        for k, v in [("net_gex", net_gex), ("gex_rank", gex_rank),
                     ("vix", vix), ("adx", adx), ("rsi", rsi)]:
            sb.update(k, v)

        vetoes: list[str] = []

        # ---- dealer regime ----
        flip_dist_frac = abs(spot - flip) / spot if spot > 0 else 0.0
        if flip_dist_frac < cfg.flip_buffer_frac:
            dealer = "at_flip"
        elif net_gex > cfg.min_gex_for_long and spot > flip:
            dealer = "long_gamma"
        else:
            dealer = "short_gamma"

        # ---- vol regime ----
        if vix <= cfg.vix_low and adx <= cfg.adx_low:
            vol = "low_vol"
        elif vix >= cfg.vix_high or adx >= cfg.adx_high:
            vol = "high_vol"
        else:
            vol = "normal_vol"

        # Term structure backwardation check (VIX9d > VIX or VIX > VIX3M)
        if vix9d > vix * 1.02 or vix > vix3m * 1.01:
            vol = "high_vol"
            vetoes.append("term_structure_inverted")

        # ---- momentum regime ----
        if rsi >= cfg.rsi_bull and spot > flip:
            momentum = "bull"
        elif rsi <= cfg.rsi_bear and spot < flip:
            momentum = "bear"
        else:
            momentum = "neutral"

        # ---- permitted engines ----
        engines: list[str] = []

        if cfg.block_on_catalyst and has_catalyst:
            vetoes.append("catalyst")
        elif dealer == "long_gamma" and vol in ("low_vol", "normal_vol"):
            if gex_rank >= cfg.min_gex_rank_premium:
                engines.append("premium_selector")
            else:
                vetoes.append(f"gex_rank_too_low:{gex_rank:.2f}<{cfg.min_gex_rank_premium}")
        elif dealer == "short_gamma":
            engines.append("directional_selector")
        elif dealer == "at_flip":
            # Transitional: neither engine until price accepts one side
            pass

        if vol == "high_vol" and dealer != "short_gamma":
            vetoes.append("high_vol_suppresses_premium")
            engines = [e for e in engines if e != "premium_selector"]

        details = {
            "dealer_regime": dealer,
            "vol_regime": vol,
            "momentum_regime": momentum,
            "gex_rank": round(gex_rank, 3),
            "flip_dist_frac": round(flip_dist_frac, 4),
            "vix": vix, "adx": adx, "rsi": rsi,
        }

        # ---- information gain vs prior ----
        info = _info_gain(self._prior, dealer, vol, momentum)

        state = RegimeState(
            dealer_regime=dealer,
            vol_regime=vol,
            momentum_regime=momentum,
            permitted_engines=engines,
            veto_reasons=vetoes,
            info_gain=round(info, 3),
            details=details,
        )
        self._prior = state
        return state


def _info_gain(prior: Optional[RegimeState],
               dealer: str, vol: str, momentum: str) -> float:
    """Fraction of the 3 axes that flipped from prior. 1.0 = full flip."""
    if prior is None:
        return 1.0
    changes = sum([
        prior.dealer_regime != dealer,
        prior.vol_regime != vol,
        prior.momentum_regime != momentum,
    ])
    return changes / 3.0


# --------------------------------------------------------------------------- #
# Demo                                                                         #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    clf = RegimeClassifier()

    long_gamma_snap = dict(
        spot=602.5, net_gex=4.2e9, gamma_flip=596.0, gex_pct_rank=0.88,
        vix=13.0, vix9d=12.1, vix3m=15.2, adx=12.5, rsi=52.0, has_catalyst=False,
    )
    short_gamma_snap = dict(
        spot=588.0, net_gex=-1.1e9, gamma_flip=593.0, gex_pct_rank=0.40,
        vix=18.0, vix9d=19.5, vix3m=17.0, adx=28.0, rsi=38.0, has_catalyst=False,
    )
    catalyst_snap = dict(
        spot=602.0, net_gex=3.0e9, gamma_flip=596.0, gex_pct_rank=0.80,
        vix=14.0, vix9d=13.5, vix3m=15.0, adx=13.0, rsi=51.0, has_catalyst=True,
    )

    for label, snap in [
        ("long_gamma_ranging", long_gamma_snap),
        ("short_gamma_trending", short_gamma_snap),
        ("catalyst_day", catalyst_snap),
    ]:
        r = clf.classify(snap)
        print(f"\n=== {label} ===")
        print(f"  dealer={r.dealer_regime}  vol={r.vol_regime}  momentum={r.momentum_regime}")
        print(f"  engines={r.permitted_engines}  vetoes={r.veto_reasons}  info_gain={r.info_gain}")
