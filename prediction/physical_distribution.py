"""
prediction/physical_distribution.py
===================================
Independent physical density for Prediction Engine V2
(docs/PREDICTION_ENGINE_V2_HANDOFF.md §12).

The physical forecast is built from a PredictionBundle (or an explicit
PhysicalForecast) and the risk-neutral density shape. It must NOT depend on
the routed structure, candidate family, direction, gate outcome, or
hand-authored conviction — those are policy outputs, and feeding them back
into the density that prices the same trade is the circular tilt this module
replaces (§3.5 / §12.5).

Phase-one transform (§12.3):
  1. take the RND shape;
  2. center it;
  3. scale deviations to the forecast physical standard deviation;
  4. shift the mean to the predicted return;
  5. blend toward the RND under model uncertainty (§12.4);
  6. clip negatives, renormalize;
  7. report moments and transformation quality.

Legacy `RNDConfig.dir_drift_frac` tilt remains available behind
`UnifiedOrchestrator.use_legacy_directional_tilt` during migration.

NOT financial advice.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
from scipy.stats import norm

from prediction.contracts import PredictionBundle

PHYSICAL_DIST_VERSION = "v2.0.0-pr5"


# --------------------------------------------------------------------------- #
# Input / output contracts                                                     #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PhysicalForecast:
    """Independent forecast inputs for the physical density (§12.1).

    expected_return / return_q* are decimal LOG returns (repo convention).
    expected_realized_move is a non-negative simple-return excursion scale
    (matches prediction/labels.remaining_realized_move). uncertainty in [0, 1].
    """
    expected_return: float
    return_q10: float
    return_q50: float
    return_q90: float
    expected_realized_move: float
    volatility_scale: float = 1.0
    skew_adjustment: float = 0.0          # reserved; phase one leaves at 0
    uncertainty: float = 0.0
    model_version: str = PHYSICAL_DIST_VERSION

    def __post_init__(self):
        if not (0.0 <= float(self.uncertainty) <= 1.0):
            raise ValueError(
                f"PhysicalForecast.uncertainty must be in [0, 1], "
                f"got {self.uncertainty!r}")
        if float(self.expected_realized_move) < 0.0:
            raise ValueError(
                f"PhysicalForecast.expected_realized_move must be >= 0, "
                f"got {self.expected_realized_move!r}")
        if float(self.volatility_scale) <= 0.0:
            raise ValueError(
                f"PhysicalForecast.volatility_scale must be > 0, "
                f"got {self.volatility_scale!r}")


@dataclass(frozen=True)
class PhysicalDensityResult:
    """Normalized physical density on the RND grid, plus audit moments."""
    grid: np.ndarray
    density: np.ndarray                   # integrates to ~1 over grid
    moments: dict = field(default_factory=dict)
    quality: dict = field(default_factory=dict)
    forecast: Optional[PhysicalForecast] = None
    mode: str = "v2"                      # "v2" | "realized_vol" | "vrp"
    model_version: str = PHYSICAL_DIST_VERSION

    def pdf(self, grid: np.ndarray) -> np.ndarray:
        """Callable-compatible interpolator onto an arbitrary price grid."""
        return np.interp(np.asarray(grid, dtype=float), self.grid, self.density,
                         left=0.0, right=0.0)

    def as_callable(self) -> Callable[[np.ndarray], np.ndarray]:
        g0, d0 = self.grid, self.density

        def _pdf(grid: np.ndarray) -> np.ndarray:
            return np.interp(np.asarray(grid, dtype=float), g0, d0,
                             left=0.0, right=0.0)
        return _pdf

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "model_version": self.model_version,
            "moments": dict(self.moments),
            "quality": dict(self.quality),
            "forecast": (None if self.forecast is None else {
                "expected_return": self.forecast.expected_return,
                "return_q10": self.forecast.return_q10,
                "return_q50": self.forecast.return_q50,
                "return_q90": self.forecast.return_q90,
                "expected_realized_move": self.forecast.expected_realized_move,
                "volatility_scale": self.forecast.volatility_scale,
                "skew_adjustment": self.forecast.skew_adjustment,
                "uncertainty": self.forecast.uncertainty,
                "model_version": self.forecast.model_version,
            }),
        }


# --------------------------------------------------------------------------- #
# Forecast extraction from PredictionBundle                                    #
# --------------------------------------------------------------------------- #
def forecast_from_bundle(bundle: PredictionBundle,
                         *, horizon: str = "close",
                         default_uncertainty: float = 0.5
                         ) -> Optional[PhysicalForecast]:
    """
    Lift a PhysicalForecast out of a PredictionBundle.

    Prefers the requested horizon's return fields, then falls back to 30m,
    then any available expected_return_*. Returns None when the bundle does
    not carry enough continuous-return information to build a density.
    """
    def _get(*names):
        for n in names:
            v = getattr(bundle, n, None)
            if isinstance(v, (int, float)) and math.isfinite(v):
                return float(v)
        return None

    if horizon == "close":
        mu = _get("expected_return_close", "expected_return_60m",
                  "expected_return_30m", "expected_return_15m")
        q10 = _get("return_q10_close", "return_q10_30m")
        q50 = _get("return_q50_close", "return_q50_30m", "expected_return_close",
                   "expected_return_30m")
        q90 = _get("return_q90_close", "return_q90_30m")
        move = _get("expected_realized_move_close", "expected_realized_move_30m")
    else:
        mu = _get(f"expected_return_{horizon}", "expected_return_30m",
                  "expected_return_close")
        q10 = _get(f"return_q10_{horizon}", "return_q10_30m", "return_q10_close")
        q50 = _get(f"return_q50_{horizon}", "return_q50_30m",
                   f"expected_return_{horizon}")
        q90 = _get(f"return_q90_{horizon}", "return_q90_30m", "return_q90_close")
        move = _get(f"expected_realized_move_{horizon}",
                    "expected_realized_move_close", "expected_realized_move_30m")

    if mu is None and q50 is not None:
        mu = q50
    if mu is None:
        return None
    if q50 is None:
        q50 = mu
    # symmetric placeholder quantiles when the quantile model is absent
    if q10 is None or q90 is None:
        width = move if (move is not None and move > 0) else 0.005
        # convert a simple-return width into an approximate log-return half-width
        half = math.log1p(width)
        q10 = mu - half if q10 is None else q10
        q90 = mu + half if q90 is None else q90
    if move is None or move <= 0:
        # fall back to a robust scale from the interquantile range
        move = max(abs(math.expm1(q90)) , abs(math.expm1(q10)), 1e-6)

    unc = bundle.uncertainty
    if not isinstance(unc, (int, float)) or not math.isfinite(unc):
        unc = default_uncertainty
    unc = float(np.clip(unc, 0.0, 1.0))

    versions = bundle.model_versions or {}
    ver = (versions.get("group")
           or versions.get("volatility")
           or versions.get(f"quantiles_{horizon}")
           or PHYSICAL_DIST_VERSION)

    # enforce q10 <= q50 <= q90 (rearrangement) before freezing
    q10, q50, q90 = sorted((float(q10), float(q50), float(q90)))
    return PhysicalForecast(
        expected_return=float(mu),
        return_q10=q10, return_q50=q50, return_q90=q90,
        expected_realized_move=float(max(move, 0.0)),
        volatility_scale=1.0, skew_adjustment=0.0,
        uncertainty=unc, model_version=str(ver),
    )


# --------------------------------------------------------------------------- #
# Density construction                                                         #
# --------------------------------------------------------------------------- #
def _moments(grid: np.ndarray, dens: np.ndarray) -> dict:
    dx = float(grid[1] - grid[0]) if len(grid) > 1 else 1.0
    area = float(np.sum(dens) * dx)
    if area <= 0:
        return {"mean": float("nan"), "std": float("nan"), "skew": float("nan"),
                "area": area}
    m = float(np.sum(grid * dens) * dx / area)
    var = float(np.sum((grid - m) ** 2 * dens) * dx / area)
    std = float(math.sqrt(max(var, 0.0)))
    skew = float("nan")
    if std > 0:
        skew = float(np.sum(((grid - m) / std) ** 3 * dens) * dx / area)
    return {"mean": m, "std": std, "variance": var, "skew": skew, "area": area}


def _renormalize(grid: np.ndarray, dens: np.ndarray) -> np.ndarray:
    dens = np.clip(np.asarray(dens, dtype=float), 0.0, None)
    dx = float(grid[1] - grid[0]) if len(grid) > 1 else 1.0
    area = float(np.sum(dens) * dx)
    if area <= 0:
        return dens
    return dens / area


def _target_std(forecast: PhysicalForecast, forward: float) -> float:
    """
    Physical price-std target from the forecast.

    Prefer the log-return interquantile range (q10/q90 ≈ ±1.28155 σ under a
    normal approximation); fall back to expected_realized_move as a 1-σ
    simple-return scale. Always multiplied by volatility_scale.
    """
    z90 = float(norm.ppf(0.9))                        # ~1.28155
    half = 0.5 * (forecast.return_q90 - forecast.return_q10)
    if half > 0 and z90 > 0:
        sigma_log = half / z90
        # lognormal price std ≈ F * exp(μ) * sqrt(exp(σ²)-1) ≈ F_pred * σ
        # for the small σ typical of remaining-session 0DTE moves
        pred_mean = forward * math.exp(forecast.expected_return)
        std = pred_mean * math.sqrt(max(math.exp(sigma_log ** 2) - 1.0, 0.0))
        if std <= 0:
            std = pred_mean * sigma_log
    else:
        pred_mean = forward * math.exp(forecast.expected_return)
        std = pred_mean * max(forecast.expected_realized_move, 0.0)
    return float(max(std * forecast.volatility_scale, 0.0))


def build_physical_density(rnd, forecast: PhysicalForecast, *,
                           scale_min: float = 0.5,
                           scale_max: float = 1.5,
                           ) -> PhysicalDensityResult:
    """
    Build the V2 physical density from an RND + independent PhysicalForecast.

    The transform does not accept — and therefore cannot depend on — any
    routed structure, direction, conviction, or gate result (§12.2).
    """
    grid = np.asarray(rnd.grid, dtype=float)
    rn_pdf = np.asarray(rnd.pdf, dtype=float)
    rn_pdf = _renormalize(grid, rn_pdf)
    F = float(rnd.forward)
    rn_m = _moments(grid, rn_pdf)

    # --- center, scale, shift (§12.3) ----------------------------------------
    predicted_mean = F * math.exp(forecast.expected_return)
    rn_mean = rn_m["mean"] if math.isfinite(rn_m["mean"]) else F
    rn_std = rn_m["std"] if (math.isfinite(rn_m["std"]) and rn_m["std"] > 0) else None
    target_std = _target_std(forecast, F)

    if rn_std is None or rn_std <= 0 or target_std <= 0:
        # degenerate: fall back to a pure mean-shifted RND (no rescale)
        scale = 1.0
    else:
        scale = float(np.clip(target_std / rn_std, scale_min, scale_max))

    # source locations whose RND density we pull onto `grid` after affine map:
    #   y = predicted_mean + scale * (x - rn_mean)  =>  x = rn_mean + (y - predicted_mean)/scale
    src = rn_mean + (grid - predicted_mean) / scale
    forecast_pdf = np.interp(src, grid, rn_pdf, left=0.0, right=0.0)
    # Jacobian of the affine map is 1/scale; absorb via renormalization
    forecast_pdf = _renormalize(grid, forecast_pdf)

    # --- uncertainty blend toward RND (§12.4) --------------------------------
    # High uncertainty => more weight on the risk-neutral density, which also
    # shrinks the directional mean shift and widens toward the RN dispersion.
    u = float(np.clip(forecast.uncertainty, 0.0, 1.0))
    conf = 1.0 - u
    blended = conf * forecast_pdf + u * rn_pdf
    blended = _renormalize(grid, blended)

    mom = _moments(grid, blended)
    # quality: how close did we land to the (confidence-weighted) targets?
    target_mean = conf * predicted_mean + u * rn_mean
    target_std_blended = conf * (scale * rn_std if rn_std else 0.0) + u * (rn_std or 0.0)
    quality = {
        "integrate_error": abs(mom["area"] - 1.0),
        "mean_error": (abs(mom["mean"] - target_mean)
                       if math.isfinite(mom["mean"]) else float("inf")),
        "std_error": (abs(mom["std"] - target_std_blended)
                      if (math.isfinite(mom["std"]) and target_std_blended > 0)
                      else float("inf")),
        "scale": scale,
        "confidence_weight": conf,
        "uncertainty": u,
        "predicted_mean": predicted_mean,
        "target_std": target_std,
        "rn_mean": rn_mean,
        "rn_std": rn_std if rn_std is not None else float("nan"),
    }
    moments = {
        **mom,
        "rn_mean": rn_mean,
        "rn_std": rn_std if rn_std is not None else float("nan"),
        "predicted_mean": predicted_mean,
        "target_std": target_std,
        "var_ratio": ((rn_std ** 2) / max(mom["std"] ** 2, 1e-18)
                      if (rn_std and math.isfinite(mom["std"])) else float("nan")),
    }
    return PhysicalDensityResult(
        grid=grid, density=blended, moments=moments, quality=quality,
        forecast=forecast, mode="v2", model_version=forecast.model_version,
    )


def density_moments(pdf: Callable, grid: np.ndarray) -> dict:
    """Moments of an arbitrary callable density on a price grid (audit helper)."""
    dens = _renormalize(grid, np.asarray(pdf(grid), dtype=float))
    return _moments(grid, dens)
