"""
forecast_stabilizer.py
======================
Whiplash control for the SPY forecast target — the piece the sigma cone was
missing. A raw per-tick forecast (the median close/target the cone points at)
is noisy; showing it unfiltered makes the cone jump every bar, and naively
smoothing it introduces lag exactly when the market turns. This module
separates the three concerns the way a trading forecast should:

  1. Forecast inertia   — confidence-weighted exponential updating. A
     low-confidence, stable-regime reading moves the target slowly; a
     high-confidence reading during a genuine regime change moves it fast:

         y_hat_t = (1 - alpha_t) * y_hat_{t-1} + alpha_t * y_raw
         alpha_t = clip( alpha_base * C_t * R_t , alpha_min, alpha_max )

     C_t = model confidence in [0,1], R_t = regime-change intensity in [0,1].

  2. Adaptive deadband  — do not move the visible target for trivial noise:

         update only if |y_hat_t - y_hat_{t-1}| > k * sigma_short

  3. Directional hysteresis — require MORE evidence to reverse the forecast
     lean than to continue it (continue < neutralize < reverse), so bar noise
     cannot flip the cone bullish/bearish every minute. The deadband
     threshold is scaled by which transition the raw move implies relative to
     the current lean (above/below spot).

  4. Structural-break override — none of the above should apply when there is
     a genuine regime break (VWAP loss with volume, gamma-flip breach + failed
     reclaim, wall breakout with acceptance, abrupt vol expansion, order-flow
     reversal, macro release). On a break the smoothing is bypassed and the
     target snaps toward the raw reading. Without this, smoothing is slow and
     stupid precisely when the market changes.

Pure, deterministic, and side-effect free: the caller holds one
`ForecastStabilizer` per session and calls `update()` each tick. No numpy, no
I/O — trivially testable and safe to sit on the live path.

NOT financial advice.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# --------------------------------------------------------------------------- #
# Config                                                                      #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class StabilizerConfig:
    alpha_base: float = 0.6        # base gain before confidence/regime scaling
    alpha_min: float = 0.02        # never fully freeze — always creep toward truth
    alpha_max: float = 1.0         # break override may use alpha_max
    deadband_k: float = 0.5        # deadband = k * sigma_short (continue case)
    # hysteresis multipliers on the deadband, by the transition the raw implies
    continue_mult: float = 1.0     # extend the current lean: easiest
    neutralize_mult: float = 1.6   # pull the target back toward spot
    reverse_mult: float = 2.8      # flip to the other side of spot: hardest
    # a raw target within this fraction of spot counts as "at spot" (no lean)
    flat_frac: float = 0.0003
    break_alpha: float = 0.9       # gain applied on a structural-break override


# --------------------------------------------------------------------------- #
# Structural-break signals                                                     #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class BreakSignals:
    """Genuine structural breaks that must bypass the smoothing. Each is a
    boolean the live loop sets from confirmed conditions (not raw crossings —
    e.g. vwap_loss_confirmed means VWAP lost WITH volume confirmation, not a
    one-tick dip)."""
    vwap_loss_confirmed: bool = False
    gamma_flip_failed_reclaim: bool = False
    call_wall_acceptance: bool = False
    put_wall_acceptance: bool = False
    vol_expansion: bool = False
    orderflow_reversal: bool = False
    correlation_shock: bool = False
    macro_release: bool = False

    def active(self) -> tuple[str, ...]:
        return tuple(n for n, v in self.__dict__.items() if v)

    def any(self) -> bool:
        return any(self.__dict__.values())


# --------------------------------------------------------------------------- #
# Result                                                                       #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class StabilizedForecast:
    target: float                  # the stabilized target to render
    raw_target: float              # the unfiltered input
    prev_target: Optional[float]   # the target before this update
    changed: bool                  # did the visible target move?
    alpha: float                   # gain actually applied (0 if held)
    deadband: float                # threshold used this tick
    transition: str                # "seed"|"continue"|"neutralize"|"reverse"|"hold"
    lean: int                      # -1/0/+1 : target below/at/above spot
    override: tuple[str, ...]      # active break signals, empty if none

    def to_dict(self) -> dict:
        return {
            "target": round(self.target, 4),
            "raw_target": round(self.raw_target, 4),
            "prev_target": (round(self.prev_target, 4)
                            if self.prev_target is not None else None),
            "changed": self.changed,
            "alpha": round(self.alpha, 4),
            "deadband": round(self.deadband, 5),
            "transition": self.transition,
            "lean": self.lean,
            "override": list(self.override),
        }


# --------------------------------------------------------------------------- #
# Stabilizer                                                                    #
# --------------------------------------------------------------------------- #
def _clip(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


class ForecastStabilizer:
    """One per session. Deterministic; call update() each tick."""

    def __init__(self, cfg: Optional[StabilizerConfig] = None) -> None:
        self.cfg = cfg or StabilizerConfig()
        self._target: Optional[float] = None
        self._lean: int = 0            # sign(target - spot) at last accepted update

    # -- introspection -------------------------------------------------------
    @property
    def target(self) -> Optional[float]:
        return self._target

    @property
    def lean(self) -> int:
        return self._lean

    def reset(self) -> None:
        """Clear state — call at each session open so a new day never inherits
        yesterday's target."""
        self._target = None
        self._lean = 0

    # -- the update ----------------------------------------------------------
    def update(self, raw_target: float, spot: float, sigma_short: float,
               confidence: float = 1.0, regime_change_intensity: float = 1.0,
               breaks: Optional[BreakSignals] = None) -> StabilizedForecast:
        """
        raw_target              the tick's unfiltered forecast target (price)
        spot                    current spot (defines the bullish/bearish lean)
        sigma_short             short-horizon price sigma (deadband scale)
        confidence              C_t in [0,1] — model confidence in raw_target
        regime_change_intensity R_t in [0,1] — 0 stable .. 1 strong change
        breaks                  confirmed structural-break signals (override)
        """
        cfg = self.cfg
        breaks = breaks or BreakSignals()
        sigma_short = max(float(sigma_short), 1e-9)
        raw_lean = _lean_of(raw_target, spot, cfg.flat_frac * spot)

        # -- first tick of the session: seed straight from the raw reading ---
        if self._target is None:
            self._target = float(raw_target)
            self._lean = raw_lean
            return StabilizedForecast(
                target=self._target, raw_target=raw_target, prev_target=None,
                changed=True, alpha=1.0, deadband=0.0, transition="seed",
                lean=raw_lean, override=())

        prev = self._target

        # -- structural-break override: bypass deadband + hysteresis ---------
        if breaks.any():
            alpha = _clip(cfg.break_alpha, cfg.alpha_min, cfg.alpha_max)
            new = (1.0 - alpha) * prev + alpha * raw_target
            self._target = new
            self._lean = _lean_of(new, spot, cfg.flat_frac * spot)
            return StabilizedForecast(
                target=new, raw_target=raw_target, prev_target=prev,
                changed=abs(new - prev) > 1e-9, alpha=alpha, deadband=0.0,
                transition="reverse" if raw_lean != self._lean and raw_lean != 0
                           else "continue",
                lean=self._lean, override=breaks.active())

        # -- classify the transition the raw move implies --------------------
        transition, mult = _classify(prev, raw_target, spot, self._lean, cfg)
        deadband = cfg.deadband_k * sigma_short * mult

        # -- adaptive deadband + hysteresis: hold on insufficient evidence ---
        # (the classification is still reported so callers can see WHY it held —
        # a reverse held by the raised reverse threshold vs a trivial continue)
        if abs(raw_target - prev) <= deadband:
            return StabilizedForecast(
                target=prev, raw_target=raw_target, prev_target=prev,
                changed=False, alpha=0.0, deadband=deadband,
                transition=transition, lean=self._lean, override=())

        # -- confidence-weighted exponential update --------------------------
        alpha = _clip(cfg.alpha_base * _clip(confidence, 0.0, 1.0)
                      * _clip(regime_change_intensity, 0.0, 1.0),
                      cfg.alpha_min, cfg.alpha_max)
        new = (1.0 - alpha) * prev + alpha * raw_target
        self._target = new
        self._lean = _lean_of(new, spot, cfg.flat_frac * spot)
        return StabilizedForecast(
            target=new, raw_target=raw_target, prev_target=prev,
            changed=abs(new - prev) > 1e-9, alpha=alpha, deadband=deadband,
            transition=transition, lean=self._lean, override=())


# --------------------------------------------------------------------------- #
# helpers                                                                       #
# --------------------------------------------------------------------------- #
def _lean_of(target: float, spot: float, flat_abs: float) -> int:
    d = target - spot
    if abs(d) <= flat_abs:
        return 0
    return 1 if d > 0 else -1


def _classify(prev: float, raw: float, spot: float, cur_lean: int,
              cfg: StabilizerConfig) -> tuple[str, float]:
    """Which transition does moving from prev toward raw imply, relative to the
    current lean (target vs spot)? Continue (extend lean) is easiest, reverse
    (cross spot to the other side) is hardest."""
    raw_lean = _lean_of(raw, spot, cfg.flat_frac * spot)
    move = raw - prev
    if cur_lean == 0:
        # no established lean yet: any move is a "continue"
        return "continue", cfg.continue_mult
    # does the raw push further in the lean's direction, or back toward/through spot?
    extending = (move > 0 and cur_lean > 0) or (move < 0 and cur_lean < 0)
    if extending and (raw_lean == cur_lean or raw_lean == 0):
        return "continue", cfg.continue_mult
    if raw_lean == -cur_lean and raw_lean != 0:
        return "reverse", cfg.reverse_mult
    return "neutralize", cfg.neutralize_mult
