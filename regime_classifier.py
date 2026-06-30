"""
regime_classifier.py
====================
Deterministic regime classifier. Consumes whatever of the state vector is live
and emits, per tick:

    - a confidence score (0..100) for each market regime
    - the dominant regime and the PERMITTED strategy engine
    - hard vetoes (catalyst, short-gamma, etc.) that override the scores
    - global information gain (how much the state shifted since last tick)

It is NOT machine-learned. Each regime is a reliability-weighted blend of
standardized features with hand-set prior weights. Those weights are exactly
what the journal's realized-P&L regression will later calibrate -- this module
is the structured prior that closes that loop.

Graceful degradation is the core feature: any input that is missing, stale, or
low-quality contributes reliability 0 and drops out; each regime renormalizes
over the features actually available. So it runs today on ~30 computable
features and tightens as order-flow / L2 feeds come online.

Standardization helpers match the feature-matrix spec.
NOT financial advice.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional

from gate_scorer import MarketSnapshot
from rnd_extractor import RiskNeutralDensity, EdgeReport


# --------------------------------------------------------------------------- #
# Standardization helpers                                                      #
# --------------------------------------------------------------------------- #
def clip100(x: float) -> float:
    return min(100.0, max(0.0, x))


def P(p: float) -> float:                      # already 0..1
    return clip100(100.0 * p)


def S(x: float, scale: float) -> float:        # signed -> 50 neutral
    return clip100(50.0 + 50.0 * math.tanh(x / scale)) if scale > 0 else 50.0


def N(x: float, scale: float) -> float:        # near-level -> 100 at zero distance
    return clip100(100.0 * math.exp(-abs(x) / scale)) if scale > 0 else 0.0


# --------------------------------------------------------------------------- #
# Adaptive scale book (Welford running mean/var per named quantity)            #
# --------------------------------------------------------------------------- #
@dataclass
class ScaleBook:
    n_min: int = 30
    _stats: dict = field(default_factory=dict)   # name -> [n, mean, M2]

    def update(self, name: str, x: float):
        if x is None or not math.isfinite(x):
            return
        s = self._stats.setdefault(name, [0, 0.0, 0.0])
        s[0] += 1
        d = x - s[1]
        s[1] += d / s[0]
        s[2] += d * (x - s[1])

    def std(self, name: str, default: float) -> float:
        s = self._stats.get(name)
        if not s or s[0] < 2:
            return default
        var = s[2] / (s[0] - 1)
        sd = math.sqrt(var) if var > 0 else default
        return sd if sd > 1e-12 else default

    def reliability(self, name: str) -> float:
        """Ramp 0->1 as samples accumulate; 0.4 floor so cold start still informs."""
        s = self._stats.get(name)
        if not s:
            return 0.4
        return 0.4 + 0.6 * min(1.0, s[0] / self.n_min)


# --------------------------------------------------------------------------- #
# Classifier input bundle                                                      #
# --------------------------------------------------------------------------- #
@dataclass
class ClassifierContext:
    market: MarketSnapshot
    rnd: Optional[RiskNeutralDensity] = None
    edge: Optional[EdgeReport] = None
    feature_ages: dict = field(default_factory=dict)   # name -> seconds since update


# --------------------------------------------------------------------------- #
# Feature specs: each returns a standardized 0..100 value or None              #
# --------------------------------------------------------------------------- #
@dataclass
class FeatureSpec:
    name: str
    domain: str
    fn: Callable[[ClassifierContext, ScaleBook], Optional[float]]
    cadence: str = "1m"
    staleness_s: float = 300.0


def _f(name, domain, fn, cadence="1m", staleness=300.0):
    return FeatureSpec(name, domain, fn, cadence, staleness)


def _safe(getter):
    try:
        v = getter()
        return v if (v is None or math.isfinite(v)) else None
    except Exception:
        return None


# ---- feature implementations (only those computable from live modules) ----
def _build_features() -> list[FeatureSpec]:
    F = []

    # Dealer
    F.append(_f("gamma_sign", "dealer",
                lambda c, sb: S(c.market.net_gex, sb.std("net_gex", 2e9))))
    F.append(_f("gamma_magnitude", "dealer",
                lambda c, sb: P(c.market.gex_pct_rank)))
    F.append(_f("flip_cushion", "dealer",
                lambda c, sb: S((c.market.spot - c.market.gamma_flip) / c.market.spot,
                                sb.std("flip_cushion", 0.004))))
    F.append(_f("flip_proximity", "dealer",
                lambda c, sb: N((c.market.spot - c.market.gamma_flip) / c.market.spot, 0.003)))

    def _channel_tight(c, sb):
        w = (c.market.call_wall - c.market.put_wall) / c.market.spot
        return clip100(100.0 * math.exp(-w / 0.012))
    F.append(_f("channel_tightness", "dealer", _channel_tight))

    def _wall_prox(c, sb):
        dc = abs(c.market.call_wall - c.market.spot) / c.market.spot
        dp = abs(c.market.spot - c.market.put_wall) / c.market.spot
        return N(min(dc, dp), 0.003)
    F.append(_f("wall_proximity", "dealer", _wall_prox))

    # Volatility
    F.append(_f("term_structure", "vol",
                lambda c, sb: S((c.market.vix3m - c.market.vix) / c.market.vix,
                                sb.std("term_structure", 0.08))))
    F.append(_f("vvix_elevation", "vol",
                lambda c, sb: clip100(100.0 * (c.market.vvix / c.market.vvix_baseline - 1.0) / 0.30 + 50.0
                                      if False else
                                      clip100(50.0 + 50.0 * math.tanh((c.market.vvix - c.market.vvix_baseline) / max(0.10 * c.market.vvix_baseline, 1e-6))))))
    F.append(_f("richness", "vol",
                lambda c, sb: P(c.edge.richness_signal) if c.edge else None))
    F.append(_f("rv_expansion", "vol",
                lambda c, sb: clip100(50.0 + 50.0 * math.tanh(
                    (c.market.bb_width / c.market.bb_width_baseline - 1.0) /
                    max(sb.std("bbw_ratio", 0.25), 1e-6)))))

    # Distribution shape
    F.append(_f("skew_dir", "shape",
                lambda c, sb: S(c.rnd.skew(), sb.std("rnd_skew", 0.20)) if c.rnd else None))
    F.append(_f("tail_heaviness", "shape",
                lambda c, sb: clip100(100.0 * c.rnd.excess_kurtosis() / 3.0) if c.rnd else None))

    # Trend
    F.append(_f("adx_strength", "trend",
                lambda c, sb: clip100(100.0 * c.market.adx / 50.0)))
    F.append(_f("rsi_centered", "trend",
                lambda c, sb: clip100(100.0 - abs(c.market.rsi - 50.0) * 2.0)))
    F.append(_f("bb_compression", "trend",
                lambda c, sb: clip100(100.0 * (1.0 - c.market.bb_width / c.market.bb_width_baseline))))
    F.append(_f("vwap_reversion", "trend",
                lambda c, sb: clip100(100.0 * min(c.market.vwap_reversion_count, 6) / 6.0)))

    # Order flow
    F.append(_f("cvd_persistence", "flow",
                lambda c, sb: S(c.market.cvd_slope, sb.std("cvd_slope", 0.4))))
    F.append(_f("tick_two_sided", "flow",
                lambda c, sb: clip100(100.0 * math.exp(-c.market.tick_abs_mean / 600.0))))

    return F


FEATURES = _build_features()


# --------------------------------------------------------------------------- #
# Regime definitions: (feature_name, weight, invert)                           #
# invert=True uses (100 - value). Confidence = Σ w·rel·v / Σ w·rel.            #
# --------------------------------------------------------------------------- #
REGIME_WEIGHTS: dict[str, list[tuple]] = {
    "compression": [
        ("gamma_sign", 1.5, False), ("flip_cushion", 1.0, False),
        ("channel_tightness", 1.2, False), ("bb_compression", 1.0, False),
        ("adx_strength", 1.3, True), ("tick_two_sided", 0.8, False),
        ("richness", 0.6, False),
    ],
    "range_confidence": [
        ("gamma_sign", 1.2, False), ("channel_tightness", 1.0, False),
        ("wall_proximity", 0.8, False), ("rsi_centered", 0.8, False),
        ("vwap_reversion", 0.8, False), ("adx_strength", 1.2, True),
    ],
    "mean_reversion_confidence": [
        ("gamma_sign", 1.0, False), ("vwap_reversion", 1.0, False),
        ("wall_proximity", 1.0, False), ("adx_strength", 1.0, True),
        ("flip_cushion", 0.6, False),
    ],
    "trend": [
        ("adx_strength", 1.5, False), ("bb_compression", 1.0, True),
        ("cvd_persistence", 0.8, False), ("rsi_centered", 0.8, True),
        ("rv_expansion", 0.8, False),
    ],
    "directional_confidence": [
        ("adx_strength", 1.2, False), ("cvd_persistence", 1.0, False),
        ("skew_dir", 0.6, False), ("vwap_reversion", 0.6, True),
    ],
    "expansion": [
        ("gamma_sign", 1.3, True), ("flip_proximity", 1.0, False),
        ("rv_expansion", 1.2, False), ("vvix_elevation", 1.0, False),
        ("tail_heaviness", 0.8, False), ("term_structure", 0.8, True),
    ],
    "breakout_confidence": [
        ("gamma_sign", 1.2, True), ("flip_proximity", 1.1, False),
        ("adx_strength", 1.0, False), ("rv_expansion", 1.0, False),
        ("tick_two_sided", 0.8, True),
    ],
    "dealer_stability": [
        ("gamma_sign", 1.5, False), ("gamma_magnitude", 1.0, False),
        ("flip_cushion", 1.0, False), ("flip_proximity", 0.8, True),
    ],
    "volatility_confidence": [   # "is premium rich" (high = favorable to sell vol)
        ("richness", 1.5, False), ("term_structure", 1.0, False),
        ("vvix_elevation", 1.0, True), ("rv_expansion", 0.8, True),
    ],
}


# --------------------------------------------------------------------------- #
# Hard vetoes: force premium-selling regimes down, regardless of scores        #
# --------------------------------------------------------------------------- #
PREMIUM_REGIMES = {"compression", "range_confidence", "mean_reversion_confidence",
                   "volatility_confidence"}

# Only these compete to be the dominant tradeable regime. dealer_stability and
# volatility_confidence are structural context, reported but never "dominant".
TRADEABLE_REGIMES = ["compression", "range_confidence", "mean_reversion_confidence",
                     "trend", "directional_confidence", "expansion", "breakout_confidence"]
SUPPORT_REGIMES = ["dealer_stability", "volatility_confidence"]

ALL_ENGINES = {"premium_selling", "directional", "vol_expansion"}


def _vetoes(ctx: ClassifierContext) -> list[tuple]:
    """Return (reason, blocked_engines). Catalyst is a true hard stop; the rest
    block premium selling but PERMIT directional / vol-expansion engines."""
    m = ctx.market
    v = []
    if m.has_catalyst:
        v.append((f"catalyst:{m.catalyst_label or 'event'}", set(ALL_ENGINES)))
    if m.net_gex <= 0:
        v.append(("short_gamma_regime", {"premium_selling"}))
    if m.spot < m.gamma_flip:
        v.append(("below_gamma_flip", {"premium_selling"}))
    if m.vix >= m.vix3m:
        v.append(("term_backwardation", {"premium_selling"}))
    if m.adx >= 25:
        v.append(("trending", {"premium_selling"}))
    return v


# --------------------------------------------------------------------------- #
# Output                                                                       #
# --------------------------------------------------------------------------- #
@dataclass
class RegimeState:
    confidences: dict             # regime -> 0..100
    reliabilities: dict           # regime -> 0..1 (coverage-weighted)
    dominant_regime: str
    permitted_engine: str         # premium_selling | directional | vol_expansion | none
    vetoes: list
    global_information_gain: float
    standardized: dict            # feature -> value (for IG + journaling)
    stand_down: bool


ENGINE_MAP = {
    "compression": "premium_selling", "range_confidence": "premium_selling",
    "mean_reversion_confidence": "premium_selling", "volatility_confidence": "premium_selling",
    "trend": "directional", "directional_confidence": "directional",
    "expansion": "vol_expansion", "breakout_confidence": "vol_expansion",
    "dealer_stability": "premium_selling",
}


# --------------------------------------------------------------------------- #
# Classifier                                                                   #
# --------------------------------------------------------------------------- #
@dataclass
class ClassifierConfig:
    min_dominant_confidence: float = 55.0
    min_regime_reliability: float = 0.35
    ig_stand_down: float = 70.0        # global IG above this => regime unstable, stand down
    update_scales: bool = True


@dataclass
class RegimeClassifier:
    scales: ScaleBook = field(default_factory=ScaleBook)
    cfg: ClassifierConfig = field(default_factory=ClassifierConfig)

    def _standardize(self, ctx: ClassifierContext) -> dict:
        out = {}
        for spec in FEATURES:
            val = _safe(lambda: spec.fn(ctx, self.scales))
            if val is None:
                out[spec.name] = (None, 0.0)
                continue
            # staleness down-weights reliability
            age = ctx.feature_ages.get(spec.name, 0.0)
            stale = 1.0 if age <= spec.staleness_s else max(0.0, 1.0 - (age - spec.staleness_s) / spec.staleness_s)
            rel = self.scales.reliability(spec.name) * stale
            out[spec.name] = (clip100(val), rel)
        return out

    def _update_scales(self, ctx: ClassifierContext):
        m = ctx.market
        self.scales.update("net_gex", m.net_gex)
        self.scales.update("flip_cushion", (m.spot - m.gamma_flip) / m.spot)
        self.scales.update("term_structure", (m.vix3m - m.vix) / m.vix)
        self.scales.update("bbw_ratio", m.bb_width / m.bb_width_baseline)
        self.scales.update("cvd_slope", m.cvd_slope)
        if ctx.rnd is not None:
            self.scales.update("rnd_skew", ctx.rnd.skew())

    def classify(self, ctx: ClassifierContext,
                 prev_standardized: Optional[dict] = None) -> RegimeState:
        if self.cfg.update_scales:
            self._update_scales(ctx)

        std = self._standardize(ctx)
        vetoes = _vetoes(ctx)
        blocked_engines = set()
        for _, eng in vetoes:
            blocked_engines |= eng

        confidences, reliabilities = {}, {}
        for regime, weights in REGIME_WEIGHTS.items():
            num = den = relsum = wsum = 0.0
            for fname, w, invert in weights:
                v, rel = std.get(fname, (None, 0.0))
                if v is None or rel <= 0:
                    wsum += w
                    continue
                vv = (100.0 - v) if invert else v
                num += w * rel * vv
                den += w * rel
                relsum += w * rel
                wsum += w
            conf = (num / den) if den > 0 else 0.0
            coverage = (relsum / wsum) if wsum > 0 else 0.0
            # knock down any regime whose mapped engine is blocked, so reported
            # confidence stays honest
            eng = ENGINE_MAP.get(regime)
            if eng in blocked_engines:
                conf = min(conf, 15.0)
            confidences[regime] = round(conf, 1)
            reliabilities[regime] = round(coverage, 3)

        # global information gain: mean |Δ| of standardized features vs prev tick
        ig = 0.0
        if prev_standardized:
            deltas = []
            for k, (v, _) in std.items():
                pv = prev_standardized.get(k, (None,))[0]
                if v is not None and pv is not None:
                    deltas.append(abs(v - pv))
            if deltas:
                ig = clip100(sum(deltas) / len(deltas) * 2.0)   # scale: 50-pt avg move -> 100

        # dominant tradeable regime: only TRADEABLE_REGIMES whose engine is not
        # blocked and whose reliability clears the floor
        eligible = {
            r: confidences[r] for r in TRADEABLE_REGIMES
            if reliabilities[r] >= self.cfg.min_regime_reliability
            and ENGINE_MAP.get(r) not in blocked_engines
        }
        dominant = max(eligible, key=eligible.get) if eligible else "none"
        top_conf = eligible.get(dominant, 0.0)

        stand_down = (top_conf < self.cfg.min_dominant_confidence
                      or ig >= self.cfg.ig_stand_down
                      or dominant == "none")
        engine = "none" if stand_down else ENGINE_MAP.get(dominant, "none")

        return RegimeState(
            confidences=confidences, reliabilities=reliabilities,
            dominant_regime=dominant, permitted_engine=engine,
            vetoes=[r for r, _ in vetoes], global_information_gain=round(ig, 1),
            standardized=std, stand_down=stand_down,
        )


# --------------------------------------------------------------------------- #
# Demo                                                                         #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import datetime as dt
    from zoneinfo import ZoneInfo
    ET = ZoneInfo("America/New_York")

    def mk(**o):
        spot = o.get("spot", 600.0)
        return MarketSnapshot(
            spot=spot, net_gex=o.get("net_gex", 4e9),
            gamma_flip=o.get("gamma_flip", spot - 7),
            call_wall=o.get("call_wall", spot + 5), put_wall=o.get("put_wall", spot - 5),
            gex_pct_rank=o.get("gex_pct_rank", 0.85),
            vix9d=o.get("vix9d", 12.0), vix=o.get("vix", 13.0), vix3m=o.get("vix3m", 15.0),
            vvix=o.get("vvix", 92.0), vvix_baseline=95.0,
            straddle_breakeven=4.0, expected_range=3.2,
            adx=o.get("adx", 12.0), rsi=o.get("rsi", 51.0),
            bb_width=o.get("bb_width", 1.4), bb_width_baseline=2.0,
            vwap=spot, vwap_reversion_count=o.get("vwap_rev", 5),
            tick_abs_mean=o.get("tick", 450.0), cvd_slope=o.get("cvd", 0.05),
            now=dt.datetime(2026, 6, 25, 11, 30, tzinfo=ET),
            has_catalyst=o.get("cat", False), catalyst_label=o.get("cat_lbl", ""),
        )

    clf = RegimeClassifier()
    scenarios = {
        "A clean compression": mk(),
        "B short-gamma breakout": mk(net_gex=-1.2e9, gamma_flip=601, adx=29,
                                     bb_width=3.2, vvix=120, tick=950, cvd=0.8, rsi=68),
        "C CPI catalyst": mk(cat=True, cat_lbl="CPI"),
        "D trending up": mk(adx=27, cvd=0.6, bb_width=2.6, rsi=64, vwap_rev=1),
    }
    for tag, m in scenarios.items():
        st = clf.classify(ClassifierContext(market=m))
        top3 = sorted(st.confidences.items(), key=lambda kv: kv[1], reverse=True)[:3]
        print(f"\n{tag}")
        print(f"  dominant={st.dominant_regime}  engine={st.permitted_engine}  stand_down={st.stand_down}")
        print(f"  vetoes={st.vetoes}")
        print(f"  top: " + ", ".join(f"{r}={c}" for r, c in top3))
