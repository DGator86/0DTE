"""
mtf_matrix.py
=============
Multi-timeframe standardized feature matrix (0-100 scale) + per-TF regime rows.

No external deps: stdlib math only.

The matrix is the single input tensor for the decision layer. Every variable
is normalized to [0, 100] using one of four helper transforms:

  clip100(x)        - raw 0..100 value, clamped
  P(x, lo, hi)      - linear ramp lo->0, hi->100. Magnitude variable (100=strong).
  S(x, lo, hi)      - symmetric around midpoint: lo->0, mid->50, hi->100.
                       For signed variables where 50 = neutral.
  N(x, mean, std)   - Gaussian CDF * 100: median->50, +2σ->97.7.

build_matrix(inp) returns a flat dict {var_name: 0..100}.
regime_rows(inp, matrix) returns per-TF regime dicts used by decision_matrix.

110-variable layout:
  32 native   = 4 TFs × 8 indicators
  24 snapshot = dealer/vol positioning
  18 derived  = cross-TF spreads + momentum comparisons
  (remaining capacity for future additions)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #
def clip100(x: float) -> float:
    return max(0.0, min(100.0, x))


def P(x: float, lo: float, hi: float) -> float:
    """Linear ramp [lo, hi] -> [0, 100]. Magnitude variable."""
    if hi == lo:
        return 50.0
    return clip100(100.0 * (x - lo) / (hi - lo))


def S(x: float, lo: float, hi: float) -> float:
    """Symmetric: mid -> 50. Directional variable (50 = neutral)."""
    mid = 0.5 * (lo + hi)
    half = 0.5 * (hi - lo)
    if half == 0:
        return 50.0
    return clip100(50.0 + 50.0 * (x - mid) / half)


def N(x: float, mean: float, std: float) -> float:
    """Gaussian CDF * 100. Median -> 50."""
    if std <= 0:
        return 50.0
    z = (x - mean) / std
    return clip100(100.0 * _normcdf(z))


def _normcdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


# --------------------------------------------------------------------------- #
# Input type                                                                   #
# --------------------------------------------------------------------------- #
# Imported from resample.py at runtime to avoid circular imports.
# We repeat the struct here so mtf_matrix.py has no deps.
@dataclass
class MTFInput:
    """
    native:   {"1m": {"adx": float, ...}, "5m": {...}, "15m": {...}, "1h": {...}}
    snapshot: gate_scorer.MarketSnapshot-compatible field dict.
    """
    native: dict[str, dict[str, float]]
    snapshot: dict[str, float] = field(default_factory=dict)


TIMEFRAMES = ("1m", "5m", "15m", "1h")

# Scale references for native indicators (these are the PRIORS; ideally sourced
# from ScaleBook at runtime — see HANDOFF §4 "Adaptive scales TODO").
_ADX_REF_HI = 30.0        # ADX >= this => full-trend score
_RSI_LO, _RSI_HI = 20.0, 80.0
_EMA_DIST_HI = 0.01       # ±1% from EMA -> ±50 symmetric swing
_BB_HI = 0.04             # BB width fraction at the high reference
_RV_HI = 0.06             # annualized RV at the high reference
_CVD_RANGE = 1.0          # CVD lives in (-1, +1)
_VWAP_DIST_HI = 0.005     # ±0.5% vwap dist -> ±50 symmetric
_TICK_HI = 1000.0         # |TICK| mean at thrust reference


def _native_vars(tf: str, ind: dict) -> dict[str, float]:
    out = {}
    prefix = tf + "_"

    def g(k: float) -> float:
        v = ind.get(k, float("nan"))
        return float("nan") if (v is None or math.isnan(v)) else v

    adx = g("adx")
    out[prefix + "adx"] = P(adx, 0.0, _ADX_REF_HI) if math.isfinite(adx) else 50.0

    rsi = g("rsi")
    out[prefix + "rsi"] = S(rsi, _RSI_LO, _RSI_HI) if math.isfinite(rsi) else 50.0

    ema = g("ema_dist")
    out[prefix + "ema_dist"] = S(ema, -_EMA_DIST_HI, _EMA_DIST_HI) if math.isfinite(ema) else 50.0

    bb = g("bb_width")
    out[prefix + "bb_width"] = P(bb, 0.0, _BB_HI) if math.isfinite(bb) else 50.0

    rv = g("rv")
    out[prefix + "rv"] = P(rv, 0.0, _RV_HI) if math.isfinite(rv) else 50.0

    cvd = g("cvd")
    out[prefix + "cvd"] = S(cvd, -_CVD_RANGE, _CVD_RANGE) if math.isfinite(cvd) else 50.0

    vd = g("vwap_dist")
    out[prefix + "vwap_dist"] = S(vd, -_VWAP_DIST_HI, _VWAP_DIST_HI) if math.isfinite(vd) else 50.0

    ta = g("tick_abs")
    out[prefix + "tick_abs"] = P(ta, 0.0, _TICK_HI) if math.isfinite(ta) else 50.0

    return out


def _snapshot_vars(snap: dict) -> dict[str, float]:
    out = {}

    def g(k, default=float("nan")):
        v = snap.get(k, default)
        return default if (v is None or (isinstance(v, float) and math.isnan(v))) else float(v)

    spot = g("spot", 600.0)
    net_gex = g("net_gex", 0.0)
    flip = g("gamma_flip", spot)
    call_wall = g("call_wall", spot + 3.0)
    put_wall = g("put_wall", spot - 3.0)
    gex_rank = g("gex_pct_rank", 0.5)
    vix = g("vix", 15.0)
    vix9d = g("vix9d", vix)
    vix3m = g("vix3m", vix)
    vvix = g("vvix", 95.0)
    vvix_baseline = g("vvix_baseline", 95.0)
    adx = g("adx", 15.0)
    rsi = g("rsi", 50.0)
    straddle_be = g("straddle_breakeven", 0.0)
    exp_range = g("expected_range", 0.0)
    bb_w = g("bb_width", 0.0)
    bb_base = g("bb_width_baseline", bb_w)

    # Signed net GEX in standardized form (100 = very positive, 0 = very negative)
    out["snap_gex_signed"] = N(net_gex / 1e9, 0.0, 2.0)
    out["snap_gex_rank"] = P(gex_rank, 0.0, 1.0)

    # Flip distance: above flip = high, below = low
    flip_dist = (spot - flip) / spot if spot > 0 else 0.0
    out["snap_flip_dist"] = S(flip_dist, -0.015, 0.015)

    # Wall proximity (100 = right at a wall, 0 = far from both)
    d_call = abs(call_wall - spot) / spot if spot > 0 else 0.05
    d_put = abs(spot - put_wall) / spot if spot > 0 else 0.05
    nearest = min(d_call, d_put)
    out["snap_wall_prox"] = P(1.0 - nearest / 0.01, 0.0, 1.0)   # 0-1% to wall

    # Vol structure
    out["snap_vix"] = P(vix, 10.0, 30.0)
    out["snap_vix9d_vix"] = S(vix9d / vix if vix > 0 else 1.0, 0.8, 1.2)
    out["snap_vix_vix3m"] = S(vix / vix3m if vix3m > 0 else 1.0, 0.8, 1.2)
    out["snap_vvix_ratio"] = S(vvix / vvix_baseline if vvix_baseline > 0 else 1.0, 0.7, 1.3)

    # Straddle richness
    if exp_range > 0:
        rich = straddle_be / exp_range
        out["snap_straddle_rich"] = P(rich, 0.8, 1.5)
    else:
        out["snap_straddle_rich"] = 50.0

    # Trend / momentum from snapshot
    out["snap_adx"] = P(adx, 0.0, 30.0)
    out["snap_rsi"] = S(rsi, 20.0, 80.0)

    # BB compression vs baseline
    if bb_base > 0:
        bb_ratio = bb_w / bb_base
        out["snap_bb_compress"] = P(1.0 - bb_ratio, -0.5, 0.5)
    else:
        out["snap_bb_compress"] = 50.0

    # Directional features
    out["snap_cvd_slope"] = S(g("cvd_slope", 0.0), -1.0, 1.0)
    out["snap_tick_calm"] = P(1.0 - g("tick_abs_mean", 600.0) / 1000.0, 0.0, 1.0)

    # Position above call/put wall (100 = above call wall = near resistance)
    out["snap_above_call_wall"] = P(spot - call_wall, -3.0, 3.0)
    out["snap_above_put_wall"] = P(spot - put_wall, -3.0, 3.0)

    return out


def _derived_vars(native_mat: dict, snap: dict) -> dict[str, float]:
    """Cross-TF spreads and agreement scores."""
    out = {}

    # Trend agreement across timeframes (all-same-direction ADX)
    adx_vals = [native_mat.get(tf + "_adx", 50.0) for tf in TIMEFRAMES]
    out["derived_adx_agreement"] = 100.0 - float(
        sum(abs(a - b) for a, b in zip(adx_vals, adx_vals[1:])) / max(len(adx_vals) - 1, 1)
    )

    # RSI momentum coherence (1h RSI - 1m RSI): positive = higher-TF bullish
    rsi_1m = native_mat.get("1m_rsi", 50.0)
    rsi_1h = native_mat.get("1h_rsi", 50.0)
    out["derived_rsi_tf_spread"] = S(rsi_1h - rsi_1m, -30.0, 30.0)

    # CVD divergence (1m vs 1h): disagreement = early warning of reversal
    cvd_1m = native_mat.get("1m_cvd", 50.0)
    cvd_1h = native_mat.get("1h_cvd", 50.0)
    out["derived_cvd_divergence"] = S(cvd_1m - cvd_1h, -50.0, 50.0)

    # BB contraction uniformity across TFs (low variance = all compressed together)
    bb_vals = [native_mat.get(tf + "_bb_width", 50.0) for tf in TIMEFRAMES]
    bb_mean = sum(bb_vals) / len(bb_vals)
    bb_var = sum((v - bb_mean) ** 2 for v in bb_vals) / len(bb_vals)
    out["derived_bb_uniform_compress"] = P(100.0 - bb_mean, 0.0, 100.0)

    # VWAP alignment across TFs
    vd_vals = [native_mat.get(tf + "_vwap_dist", 50.0) for tf in TIMEFRAMES]
    vd_mean = sum(vd_vals) / len(vd_vals)
    out["derived_vwap_align"] = S(vd_mean, 20.0, 80.0)

    # Combined ranging score: high = low ADX, low BB, calm tick
    snap_adx = native_mat.get("snap_adx", 50.0)
    tick_calm = native_mat.get("snap_tick_calm", 50.0)
    bb_compress = native_mat.get("snap_bb_compress", 50.0)
    out["derived_ranging_score"] = clip100(
        (100.0 - snap_adx) * 0.4 + tick_calm * 0.3 + bb_compress * 0.3
    )

    return out


# --------------------------------------------------------------------------- #
# Top-level                                                                    #
# --------------------------------------------------------------------------- #
def build_matrix(inp: MTFInput) -> dict[str, float]:
    """Build the full standardized 0-100 feature matrix from an MTFInput."""
    mat: dict[str, float] = {}

    # 1. Native per-TF indicators
    for tf in TIMEFRAMES:
        ind = inp.native.get(tf, {})
        mat.update(_native_vars(tf, ind))

    # 2. Snapshot variables
    snap_vars = _snapshot_vars(inp.snapshot)
    mat.update(snap_vars)

    # 3. Derived cross-TF
    mat.update(_derived_vars(mat, inp.snapshot))

    return mat


def regime_rows(inp: MTFInput, matrix: Optional[dict] = None) -> list[dict]:
    """
    Per-TF regime classification dict list. Used by decision_matrix._dominant
    to find the consensus regime across timeframes.

    Returns list of dicts, one per TF:
      {"tf": str, "dealer_regime": str, "vol_regime": str, "momentum_regime": str,
       "adx": float, "rsi": float, "cvd": float, ...}
    """
    if matrix is None:
        matrix = build_matrix(inp)

    rows = []
    for tf in TIMEFRAMES:
        adx_s = matrix.get(tf + "_adx", 50.0)
        rsi_s = matrix.get(tf + "_rsi", 50.0)
        cvd_s = matrix.get(tf + "_cvd", 50.0)

        # dealer regime proxy from snapshot (same for all TFs — it's a snapshot)
        flip_dist = matrix.get("snap_flip_dist", 50.0)
        gex_signed = matrix.get("snap_gex_signed", 50.0)

        if gex_signed >= 60.0 and flip_dist >= 55.0:
            dealer = "long_gamma"
        elif gex_signed <= 40.0 or flip_dist <= 45.0:
            dealer = "short_gamma"
        else:
            dealer = "at_flip"

        # vol regime from per-TF ADX + snapshot VIX
        vix_s = matrix.get("snap_vix", 50.0)
        if adx_s <= 40.0 and vix_s <= 40.0:
            vol = "low_vol"
        elif adx_s >= 70.0 or vix_s >= 70.0:
            vol = "high_vol"
        else:
            vol = "normal_vol"

        # momentum from RSI
        if rsi_s >= 60.0:
            momentum = "bull"
        elif rsi_s <= 40.0:
            momentum = "bear"
        else:
            momentum = "neutral"

        rows.append({
            "tf": tf,
            "dealer_regime": dealer,
            "vol_regime": vol,
            "momentum_regime": momentum,
            "adx_score": round(adx_s, 1),
            "rsi_score": round(rsi_s, 1),
            "cvd_score": round(cvd_s, 1),
        })
    return rows


# --------------------------------------------------------------------------- #
# Demo                                                                         #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    # Build a minimal MTFInput manually (no bars needed)
    native = {}
    for tf in TIMEFRAMES:
        native[tf] = {
            "adx": 12.5, "rsi": 52.0, "ema_dist": 0.001,
            "bb_width": 0.012, "rv": 0.02, "cvd": 0.08,
            "vwap_dist": 0.0005, "tick_abs": 480.0,
        }

    snap = {
        "spot": 602.50, "net_gex": 4.2e9, "gamma_flip": 596.0,
        "call_wall": 603.0, "put_wall": 598.0, "gex_pct_rank": 0.88,
        "vix": 13.0, "vix9d": 12.1, "vix3m": 15.2,
        "vvix": 92.0, "vvix_baseline": 95.0,
        "straddle_breakeven": 4.1, "expected_range": 3.2,
        "adx": 12.5, "rsi": 52.0,
        "bb_width": 1.5, "bb_width_baseline": 2.1,
        "cvd_slope": 0.03, "tick_abs_mean": 480.0,
    }

    inp = MTFInput(native=native, snapshot=snap)
    mat = build_matrix(inp)

    print(f"Matrix has {len(mat)} variables")
    print("\nSample:")
    for k in ("1m_adx", "1m_rsi", "snap_gex_signed", "snap_flip_dist",
              "snap_straddle_rich", "derived_ranging_score"):
        print(f"  {k}: {mat[k]:.1f}")

    rows = regime_rows(inp, mat)
    print("\nRegime rows:")
    for r in rows:
        print(f"  {r['tf']}: dealer={r['dealer_regime']}  vol={r['vol_regime']}  "
              f"momentum={r['momentum_regime']}")
    print("mtf_matrix OK")
