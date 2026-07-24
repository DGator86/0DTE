"""
regime_calibration.py
=====================
The regime-label bridge and empirical transition calibration for the Markov
universe simulator (matrix_universe.py).

Why this exists
---------------
The simulator's regimes (`pin`, `drift_up`, `drift_down`, `compression`,
`breakout`) and archetypes (`calm_pin`, `grind_up`, …) are a DISTINCT taxonomy
from the live Legacy/V3 regime labels. Recorded SPY sessions therefore cannot
be fed to a transition estimator directly — a real minute has no "pin" or
"breakout" stamp on it. This module supplies the missing bridge:

  1. LABEL   observable per-minute features (trend strength, realized vol,
             range expansion, distance to the pin) -> a simulator regime; and
             a session's aggregate -> a simulator archetype. Feed-agnostic:
             `features_from_snapshot` extracts the inputs from any DataFeed's
             TickSnapshot, so the same labeler runs on the simulator's own
             output and on recorded real ticks.
  2. ESTIMATE  count labeled transitions and row-normalize (Laplace-smoothed,
             canonical fallback for unseen states) -> empirical transition
             matrices with the exact shape the generator consumes.
  3. CALIBRATE  run a feed, label every observed snapshot, estimate both
             layers -> a calibrated config that `MarkovWorldFeed` can consume
             via its `arch_transition` / `regime_transition` overrides.

Validation without real data
----------------------------
`estimate_transitions` is round-trip testable against a known matrix. The
labeler is testable against the simulator's OWN ground-truth `situation_log`
(does it recover the latent regime from the observable features it emitted?).
So the whole pipeline is exercised end-to-end in CI even though the container
has no recorded SPY tape; on the VPS the identical code path consumes
`chain_store.RecordedFeed` instead.

Honest limit: labeling latent regimes from noisy one-minute features is
approximate by construction (a single breakout minute's noise dwarfs its
drift). Calibrated matrices are a data-informed PRIOR to spar against, not
ground truth — treat a calibrated run as another challenger, judged by the
same journal readouts as everything else.

NOT financial advice.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from matrix_universe import (
    ARCHETYPES, REGIMES, _ARCH_TRANSITION, _REGIME_TRANSITION,
)

MINUTES_PER_YEAR = 252 * 390


# --------------------------------------------------------------------------- #
# Feature extraction (feed-agnostic)                                          #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RegimeFeatures:
    """The minimal observable set the labeler needs, extractable from any
    feed's TickSnapshot (simulator or recorded)."""
    ret_recent: float        # signed trailing return (trend direction/size)
    rv_recent: float         # annualized realized vol of the trailing window
    adx: float               # trend strength (0..100)
    range_ratio: float       # bb_width / bb_width_baseline (>1 = expanding)
    dist_to_pin: float       # |spot - pin| / spot
    gex_sign: float          # +1 long gamma, -1 short


def features_from_snapshot(snap, window: int = 15) -> Optional[RegimeFeatures]:
    """Extract RegimeFeatures from a unified_loop.TickSnapshot. Returns None
    when the snapshot lacks the bars needed for a trailing window."""
    m = snap.market
    bars = snap.bars
    if bars is None or len(bars.close) < 2:
        return None
    closes = np.asarray(bars.close, dtype=float)
    w = min(window, len(closes) - 1)
    ret_recent = float(closes[-1] / closes[-1 - w] - 1.0)
    logret = np.diff(np.log(closes[-(w + 1):]))
    rv_recent = float(np.std(logret) * math.sqrt(MINUTES_PER_YEAR)) if len(logret) else 0.0

    baseline = getattr(m, "bb_width_baseline", None) or 0.0
    width = getattr(m, "bb_width", None) or 0.0
    range_ratio = (width / baseline) if baseline else 1.0

    # pin proxy: midpoint of the dealer walls when present, else gamma flip
    call_wall = getattr(m, "call_wall", None)
    put_wall = getattr(m, "put_wall", None)
    if call_wall is not None and put_wall is not None:
        pin = 0.5 * (call_wall + put_wall)
    else:
        pin = getattr(m, "gamma_flip", None) or m.spot
    dist_to_pin = abs(m.spot - pin) / m.spot if m.spot else 0.0
    gex_sign = 1.0 if (getattr(m, "net_gex", 0.0) or 0.0) > 0 else -1.0

    return RegimeFeatures(ret_recent, rv_recent, m.adx or 0.0,
                          range_ratio, dist_to_pin, gex_sign)


# --------------------------------------------------------------------------- #
# The labeler (the bridge)                                                    #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SessionContext:
    """Per-session scale so the labeler judges a minute RELATIVE to the day it
    lives in — a pin minute inside a crash day is calm *for that day* even
    though its absolute vol is high. Absolute features track the archetype's
    vol level, not the minute's regime, so session-relative is essential."""
    rv_median: float                 # median trailing rv over the session
    window: int = 15                 # feature window (minutes)


@dataclass(frozen=True)
class LabelConfig:
    """Thresholds for mapping SESSION-RELATIVE features to simulator regimes.
    Tuned against the simulator's own ground truth; exposed so a recalibration
    can retune against real internals without a code change."""
    breakout_rv_z: float = 1.7       # rv >= this * session median = breakout
    compression_rv_z: float = 0.6    # rv <= this * session median = compression
    trend_z: float = 0.6             # |move| >= this * expected window move = drift
    abs_rv_ref: float = 0.14         # fallback session rv when no context given
    abs_move_ref: float = 0.0009     # fallback expected window move
    # archetype layer (day aggregates; net/gap are stride-robust, rv relative)
    arch_hi_rv_z: float = 1.20       # day rv >= this * cross-session median = stressed
    arch_lo_rv_z: float = 0.85       # day rv <= this * median = quiet
    arch_gap_abs: float = 0.006      # |overnight gap| above this = gap_shock
    arch_crash_net: float = 0.008    # day net move <= -this + stressed = crash
    arch_squeeze_net: float = 0.008  # day net move >= this + stressed = squeeze
    arch_grind_net: float = 0.004    # |day net move| above this = grind
    arch_calm_gex: float = 0.4       # mean gex sign above this + quiet = calm_pin


def label_regime(f: RegimeFeatures, ctx: Optional[SessionContext] = None,
                 cfg: Optional[LabelConfig] = None) -> str:
    """Map observable features to one of the five simulator regimes, judged
    relative to the session. Priority: breakout (locally much more volatile) >
    compression (locally much calmer) > directional drift (a move beyond the
    day's expected window move) > pin. Latent-regime labeling from one-minute
    features is approximate by construction — see labeler_accuracy()."""
    cfg = cfg or LabelConfig()
    if ctx is not None and ctx.rv_median > 0:
        rv_z = f.rv_recent / ctx.rv_median
        move_scale = ctx.rv_median / math.sqrt(MINUTES_PER_YEAR) * math.sqrt(ctx.window)
    else:
        rv_z = f.rv_recent / cfg.abs_rv_ref
        move_scale = cfg.abs_move_ref
    ret_n = f.ret_recent / move_scale if move_scale > 0 else 0.0

    if rv_z >= cfg.breakout_rv_z:
        return "breakout"
    if rv_z <= cfg.compression_rv_z:
        return "compression"
    if ret_n >= cfg.trend_z:
        return "drift_up"
    if ret_n <= -cfg.trend_z:
        return "drift_down"
    return "pin"


def label_archetype(day_returns: np.ndarray, day_rv: float, gap: float,
                    mean_gex_sign: float, rv_ref: Optional[float] = None,
                    cfg: Optional[LabelConfig] = None) -> str:
    """Map a session's aggregate to one of the eight archetypes.

      day_returns   per-step returns for the session
      day_rv        the session's realized vol (any consistent annualization)
      gap           overnight gap fraction (open vs prior close)
      mean_gex_sign mean sign of net gamma over the session (+1 long, -1 short)
      rv_ref        cross-session median rv; when given, the stressed/quiet cut
                    is RELATIVE to it (stride-robust). Falls back to day_rv.

    `net` (total day move) and `|gap|` are stride-robust and do the heavy
    separation; rv is only used relatively.
    """
    cfg = cfg or LabelConfig()
    net = float(np.sum(day_returns)) if len(day_returns) else 0.0
    ref = rv_ref or day_rv or 1e-9
    hi_vol = day_rv >= cfg.arch_hi_rv_z * ref
    lo_vol = day_rv <= cfg.arch_lo_rv_z * ref
    if abs(gap) >= cfg.arch_gap_abs:
        return "gap_shock"
    if hi_vol and net <= -cfg.arch_crash_net:
        return "crash"
    if hi_vol and net >= cfg.arch_squeeze_net:
        return "squeeze_melt_up"
    if hi_vol:
        return "vol_expansion"
    if net >= cfg.arch_grind_net:
        return "grind_up"
    if net <= -cfg.arch_grind_net:
        return "grind_down"
    if lo_vol and mean_gex_sign > cfg.arch_calm_gex:
        return "calm_pin"
    return "range_chop"


# --------------------------------------------------------------------------- #
# Empirical transition estimation                                            #
# --------------------------------------------------------------------------- #
def estimate_transitions(sequences: list[list[str]], states: list[str],
                         smoothing: float = 0.5,
                         prior: Optional[dict] = None) -> dict[str, dict[str, float]]:
    """Empirical row-stochastic transition matrix from label sequences.

      sequences   list of label sequences (each a list of state names)
      states      the full ordered state set (rows and columns)
      smoothing   Laplace alpha added to every count (>0 keeps rows proper and
                  never assigns a hard zero to an unobserved transition)
      prior       optional {state: {state: p}} used verbatim for any row that
                  never appears as a source (else a smoothing-only uniform row)

    Deterministic; no RNG.
    """
    counts = {s: {t: 0.0 for t in states} for s in states}
    seen = {s: False for s in states}
    for seq in sequences:
        for a, b in zip(seq, seq[1:]):
            if a in counts and b in counts[a]:
                counts[a][b] += 1.0
                seen[a] = True

    out: dict[str, dict[str, float]] = {}
    for s in states:
        if not seen[s] and prior and s in prior:
            out[s] = {t: float(prior[s].get(t, 0.0)) for t in states}
            total = sum(out[s].values()) or 1.0
            out[s] = {t: v / total for t, v in out[s].items()}
            continue
        row = {t: counts[s][t] + smoothing for t in states}
        total = sum(row.values())
        out[s] = {t: v / total for t, v in row.items()}
    return out


# --------------------------------------------------------------------------- #
# Calibration from a feed (the real-data path, testable on the simulator)     #
# --------------------------------------------------------------------------- #
@dataclass
class Calibration:
    arch_transition: dict[str, dict[str, float]]
    regime_transition: dict[str, dict[str, dict[str, float]]]
    n_sessions: int
    n_minutes: int
    source: str

    def to_dict(self) -> dict:
        return {
            "arch_transition": self.arch_transition,
            "regime_transition": self.regime_transition,
            "n_sessions": self.n_sessions,
            "n_minutes": self.n_minutes,
            "source": self.source,
        }


def calibrate_from_feed(feed_factory: Callable, timestamps: list,
                        label_cfg: Optional[LabelConfig] = None,
                        smoothing: float = 0.5,
                        source: str = "feed") -> Calibration:
    """Label every observed snapshot of a feed and estimate both transition
    layers. Works on ANY DataFeed — a MarkovWorldFeed (for tests) or a
    chain_store.RecordedFeed of real ticks (on the VPS). Session boundaries
    come from the ET date of each tick; the per-session archetype is labeled
    from that session's aggregate, and per-archetype regime sequences feed the
    minute-scale estimator."""
    from zoneinfo import ZoneInfo
    ET = ZoneInfo("America/New_York")

    feed = feed_factory()
    # pass 1: collect raw features per session (need the whole session to set
    # the session-relative scale before labeling any minute)
    per_session: dict[str, dict] = {}
    prev_close: Optional[float] = None
    for t in timestamps:
        snap = feed.snapshot(t)
        if snap is None:
            continue
        day = t.astimezone(ET).date().isoformat()
        feats = features_from_snapshot(snap)
        if feats is None:
            continue
        sess = per_session.setdefault(
            day, {"feats": [], "step_rets": [], "gex_signs": [],
                  "first_spot": snap.market.spot, "open_prev": prev_close})
        sess["feats"].append(feats)
        # per-step close-to-close return for the day's realized vol / net move
        if prev_close:
            sess["step_rets"].append(snap.market.spot / prev_close - 1.0)
        sess["gex_signs"].append(feats.gex_sign)
        prev_close = snap.market.spot

    # per-session day realized vol first, so the archetype cut is relative to
    # the cross-session median (stride-robust)
    day_rv: dict[str, float] = {}
    for day, s in per_session.items():
        sr = np.asarray(s["step_rets"], dtype=float)
        day_rv[day] = float(np.std(sr) * math.sqrt(MINUTES_PER_YEAR)) if len(sr) > 1 else 0.0
    rv_ref = float(np.median([v for v in day_rv.values() if v > 0])) if day_rv else 0.0

    # pass 2: per-session context -> label regimes + archetype
    arch_seq: list[str] = []
    regime_seqs_by_arch: dict[str, list[list[str]]] = {a: [] for a in ARCHETYPES}
    n_minutes = 0
    for day in sorted(per_session):
        s = per_session[day]
        feats = s["feats"]
        rv_median = float(np.median([f.rv_recent for f in feats])) if feats else 0.0
        ctx = SessionContext(rv_median=rv_median)
        regimes = [label_regime(f, ctx, label_cfg) for f in feats]

        step_rets = np.asarray(s["step_rets"], dtype=float)
        gap = ((s["first_spot"] / s["open_prev"] - 1.0)
               if s["open_prev"] else 0.0)
        mean_gex = float(np.mean(s["gex_signs"])) if s["gex_signs"] else 0.0
        arch = label_archetype(step_rets, day_rv[day], gap, mean_gex,
                               rv_ref=rv_ref, cfg=label_cfg)
        arch_seq.append(arch)
        regime_seqs_by_arch[arch].append(regimes)
        n_minutes += len(regimes)

    arch_T = estimate_transitions([arch_seq], list(ARCHETYPES),
                                  smoothing=smoothing, prior=_ARCH_TRANSITION)
    regime_T: dict[str, dict[str, dict[str, float]]] = {}
    for arch in ARCHETYPES:
        regime_T[arch] = estimate_transitions(
            regime_seqs_by_arch[arch], list(REGIMES),
            smoothing=smoothing, prior=_REGIME_TRANSITION[arch])

    return Calibration(arch_T, regime_T, len(per_session), n_minutes, source)


def labeler_accuracy(feed) -> dict:
    """Diagnostic: how often the labeler recovers the simulator's OWN latent
    regime from the features it emitted, using per-session context. Only
    meaningful on a MarkovWorldFeed (ground-truth situation_log). Latent-regime
    labeling is approximate by construction (the simulator's minute returns are
    noise-dominated), so accuracy well above the 1/|regimes| chance line — not
    near 1.0 — is the honest bar."""
    from zoneinfo import ZoneInfo
    ET = ZoneInfo("America/New_York")
    ticks = feed.timestamps()
    stride = feed.spec.tick_stride

    # pass 1: features + true labels grouped by session
    by_day: dict[str, list] = {}
    for j, t in enumerate(ticks):
        snap = feed.snapshot(t)
        if snap is None:
            continue
        feats = features_from_snapshot(snap)
        if feats is None:
            continue
        true = feed.situation_log[j * stride].regime
        by_day.setdefault(t.astimezone(ET).date().isoformat(), []).append((feats, true))

    confusion = {r: {q: 0 for q in REGIMES} for r in REGIMES}
    correct = total = 0
    for rows in by_day.values():
        rv_median = float(np.median([f.rv_recent for f, _ in rows]))
        ctx = SessionContext(rv_median=rv_median)
        for feats, true in rows:
            pred = label_regime(feats, ctx)
            confusion[true][pred] += 1
            total += 1
            correct += 1 if pred == true else 0
    return {"accuracy": (correct / total) if total else None,
            "n": total, "confusion": confusion,
            "chance": 1.0 / len(REGIMES)}


def archetype_labeler_accuracy(feeds: list) -> dict:
    """Diagnostic: how often label_archetype recovers each generated session's
    true archetype. Day aggregates separate far better than minute regimes, so
    this is the reliable layer of the calibration."""
    from zoneinfo import ZoneInfo
    ET = ZoneInfo("America/New_York")
    correct = total = 0
    for feed in feeds:
        ticks = feed.timestamps()
        by_day: dict[str, dict] = {}
        prev = None
        for t in ticks:
            snap = feed.snapshot(t)
            if snap is None:
                continue
            day = t.astimezone(ET).date().isoformat()
            s = by_day.setdefault(day, {"rets": [], "gex": [],
                                        "first": snap.market.spot, "prev": prev})
            if prev:
                s["rets"].append(snap.market.spot / prev - 1.0)
            s["gex"].append(1.0 if snap.market.net_gex > 0 else -1.0)
            prev = snap.market.spot
        day_rv = {}
        for day, s in by_day.items():
            r = np.asarray(s["rets"], dtype=float)
            day_rv[day] = float(np.std(r) * math.sqrt(MINUTES_PER_YEAR)) if len(r) > 1 else 0.0
        rv_ref = float(np.median([v for v in day_rv.values() if v > 0])) if day_rv else 0.0
        for day, s in by_day.items():
            rets = np.asarray(s["rets"], dtype=float)
            gap = (s["first"] / s["prev"] - 1.0) if s["prev"] else 0.0
            pred = label_archetype(rets, day_rv[day], gap,
                                   float(np.mean(s["gex"])), rv_ref=rv_ref)
            true = feed.day_archetype.get(day)
            if true is not None:
                total += 1
                correct += 1 if pred == true else 0
    return {"accuracy": (correct / total) if total else None, "n": total,
            "chance": 1.0 / len(ARCHETYPES)}


if __name__ == "__main__":
    from matrix_universe import MarkovWorldFeed, UniverseSpec

    feed = MarkovWorldFeed(UniverseSpec("demo", 5, 6, "range_chop", tick_stride=3))
    acc = labeler_accuracy(feed)
    print(f"labeler accuracy vs ground truth: {acc['accuracy']:.1%} "
          f"over {acc['n']} ticks (chance = {1/len(REGIMES):.0%})")

    cal = calibrate_from_feed(
        lambda: MarkovWorldFeed(UniverseSpec("demo", 5, 6, "range_chop", tick_stride=3)),
        feed.timestamps(), source="demo")
    print(f"calibrated from {cal.n_sessions} sessions / {cal.n_minutes} minutes")
    print("pin->pin (calibrated, calm_pin):",
          round(cal.regime_transition["calm_pin"]["pin"]["pin"], 3))
