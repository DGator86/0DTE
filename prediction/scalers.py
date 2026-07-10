"""
prediction/scalers.py
=====================
Robust adaptive standardization scales for Prediction Engine V2
(docs/PREDICTION_ENGINE_V2_HANDOFF.md §10).

What the legacy ScaleBook (regime_classifier.py) got wrong for market data,
and what this module fixes:

1. TIMEFRAME MIXING — the matrix updated native feature statistics under the
   feature name alone, pooling e.g. the 1-minute ema_slope distribution with
   the 1-day one. Here, native features are keyed ``"{name}:{timeframe}"``
   and snapshot features by name; state under one key can never bleed into
   another.

2. UPDATE-BEFORE-SCORE — the current observation was entered into the
   Welford state before its own standardized value was computed, giving it a
   small contemporaneous influence on its own score. RobustScaleBook is
   read/write split: ``std()`` never mutates, and callers must ``update()``
   only AFTER the observation has been scored. The class advertises this
   contract via ``SCORE_BEFORE_UPDATE = True`` so consumers (mtf_matrix)
   can distinguish it from the legacy book without an import cycle.

3. LIFETIME MEMORY — Welford statistics weigh a print from three months ago
   the same as one from three minutes ago, so a volatility regime shift
   permanently distorts the scale. Here mean/variance are exponentially
   decayed with a per-timeframe half-life (fast timeframes forget faster),
   and updates are winsorized so a single insane print cannot blow up the
   scale it will later be judged against.

4. UNVERSIONED STATE — persisted scales carried no schema/config identity,
   so state written under one feature definition could silently load under
   another. Persistence here embeds a state version and a configuration
   hash; incompatible or corrupt state is REJECTED (re-warm from priors),
   never silently reinterpreted.

Cold start keeps the existing semantics: below n_min samples ``std()``
returns the caller's fixed prior and ``reliability()`` ramps 0.4 → 1.0,
matching the legacy ScaleBook so downstream reliability weighting is
unchanged.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Optional

STATE_VERSION = "rsb-1"

# Half-lives in UPDATES (one update per orchestrator tick, nominally one
# minute). Derived from the spec's session half-lives (§10.4) at ~390
# one-minute ticks per regular session:
#   1m/5m: 5 sessions, 15m/30m: 10, 1h: 20, 4h/1d: 60, snapshot: 20.
TICKS_PER_SESSION = 390
DEFAULT_HALF_LIVES: dict[str, float] = {
    "1m": 5 * TICKS_PER_SESSION,
    "5m": 5 * TICKS_PER_SESSION,
    "15m": 10 * TICKS_PER_SESSION,
    "30m": 10 * TICKS_PER_SESSION,
    "1h": 20 * TICKS_PER_SESSION,
    "4h": 60 * TICKS_PER_SESSION,
    "1d": 60 * TICKS_PER_SESSION,
    "snapshot": 20 * TICKS_PER_SESSION,
}


def scale_key(name: str, timeframe: Optional[str] = None) -> str:
    """Canonical state key: ``name`` for snapshot features,
    ``name:timeframe`` for native (per-timeframe) features."""
    return name if timeframe is None else f"{name}:{timeframe}"


@dataclass
class RobustScaleBook:
    """
    Exponentially decayed mean/variance per key, with winsorized updates.

    Contract: ``std()``/``reliability()`` are read-only; call ``update()``
    only AFTER the current observation has been scored against the existing
    state (see SCORE_BEFORE_UPDATE).
    """
    # Marker consumed by mtf_matrix.build_matrix to select the lagged,
    # per-timeframe code path without importing this module.
    SCORE_BEFORE_UPDATE = True

    n_min: int = 30
    winsor_k: float = 8.0        # clip |x - mean| at winsor_k * std once warm
    half_lives: dict = field(default_factory=lambda: dict(DEFAULT_HALF_LIVES))
    default_half_life: float = 20 * TICKS_PER_SESSION
    # key -> [n, W, mean, S]: exponentially weighted Welford state, where n is
    # the raw sample count (reliability ramp), W the decayed total weight
    # (converges to the half-life-implied window size), mean the decayed
    # mean, and S the decayed sum of squared deviations. With no decay this
    # reduces EXACTLY to classic Welford, so early warm-up behaves like the
    # legacy book instead of underestimating variance for thousands of ticks.
    _stats: dict = field(default_factory=dict)

    # -- configuration identity -------------------------------------------------
    def config_hash(self) -> str:
        payload = json.dumps({
            "state_version": STATE_VERSION,
            "n_min": self.n_min,
            "winsor_k": self.winsor_k,
            "half_lives": {k: self.half_lives[k] for k in sorted(self.half_lives)},
            "default_half_life": self.default_half_life,
        }, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def _alpha(self, key: str) -> float:
        # Half-life is chosen by the key's timeframe suffix; snapshot keys
        # (no ":") use the snapshot bucket.
        tf = key.rsplit(":", 1)[1] if ":" in key else "snapshot"
        hl = float(self.half_lives.get(tf, self.default_half_life))
        return 1.0 - 2.0 ** (-1.0 / max(1.0, hl))

    # -- read side (never mutates) ----------------------------------------------
    def std(self, key: str, default: float) -> float:
        s = self._stats.get(key)
        if not s or s[0] < 2 or s[1] <= 1.0:
            return default
        var = s[3] / (s[1] - 1.0)
        sd = math.sqrt(var) if var > 0 else default
        return sd if sd > 1e-12 else default

    def reliability(self, key: str) -> float:
        """Ramp 0.4 -> 1.0 as samples accumulate (legacy semantics)."""
        s = self._stats.get(key)
        if not s:
            return 0.4
        return 0.4 + 0.6 * min(1.0, s[0] / self.n_min)

    # -- write side ---------------------------------------------------------------
    def update(self, key: str, x: float) -> None:
        if x is None or not math.isfinite(x):
            return
        s = self._stats.setdefault(key, [0, 0.0, 0.0, 0.0])
        lam = 1.0 - self._alpha(key)          # per-step decay of old weight
        n, w, mean, sq = s
        d = float(x) - mean
        # Winsorize once the scale is warm so one extreme print cannot
        # permanently widen the distribution it will be judged against.
        cur = self.std(key, 0.0)
        if n >= self.n_min and cur > 0:
            lim = self.winsor_k * cur
            d = max(-lim, min(lim, d))
        x_eff = mean + d
        w_new = lam * w + 1.0
        mean_new = mean + d / w_new
        s[0] = n + 1
        s[1] = w_new
        s[2] = mean_new
        s[3] = lam * sq + d * (x_eff - mean_new)

    # -- persistence ------------------------------------------------------------
    # Versioned + config-hashed: state persisted under a different scaler
    # definition is rejected (re-warm), never silently reinterpreted.
    def to_dict(self) -> dict:
        return {
            "meta": {"state_version": STATE_VERSION,
                     "config_hash": self.config_hash()},
            "stats": {k: list(v) for k, v in self._stats.items()},
        }

    def load_dict(self, data: dict) -> bool:
        """Load persisted state. Returns True when accepted; on any
        incompatibility (missing/mismatched version or config hash, corrupt
        payload) the book resets to cold start and returns False."""
        self._stats = {}
        try:
            meta = data.get("meta") or {}
            if meta.get("state_version") != STATE_VERSION:
                return False
            if meta.get("config_hash") != self.config_hash():
                return False
            self._stats = {
                str(k): [int(v[0]), float(v[1]), float(v[2]), float(v[3])]
                for k, v in (data.get("stats") or {}).items()
            }
            return True
        except Exception:
            self._stats = {}
            return False


if __name__ == "__main__":
    # Demo: the same raw stream produces different scales per timeframe key,
    # and the current observation never influences its own score.
    import random

    rng = random.Random(7)
    book = RobustScaleBook()
    for i in range(500):
        book.update(scale_key("ema_slope", "1m"), rng.gauss(0.0, 0.05))
        book.update(scale_key("ema_slope", "1d"), rng.gauss(0.0, 1.5))
    print(f"ema_slope:1m  std={book.std('ema_slope:1m', 999):.4f} "
          f"(prior would be 0.05)")
    print(f"ema_slope:1d  std={book.std('ema_slope:1d', 999):.4f} "
          f"(pooled legacy book would have blended these)")
    d = book.to_dict()
    fresh = RobustScaleBook()
    assert fresh.load_dict(d), "round-trip must be accepted"
    assert not RobustScaleBook(n_min=99).load_dict(d), \
        "different config must be rejected"
    print("persistence round-trip OK; incompatible config rejected")
