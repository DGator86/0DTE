"""
prediction/structural_state.py
==============================
Canonical expanded dealer-structure state for Prediction Engine V3 Part 2
(docs/PREDICTION_ENGINE_V3_PART2_FORECASTING.md §5–§8, PR 7).

GEX variants are preserved in parallel. Missing sources stay missing —
never silently replaced with 0.0. Compatibility properties fall back
hybrid → OI → volume and record provenance in diagnostics.

Research / shadow only. Does not alter live gates or order routing.

NOT financial advice.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Optional, Sequence

STRUCTURAL_STATE_VERSION = "v3.0.0"

_FALLBACK_ORDER_DEFAULT = ("hybrid", "oi", "volume")
_VARIANT_KEYS = ("oi", "volume", "hybrid", "weekly")


@dataclass
class StructuralStateConfig:
    velocity_windows: tuple[int, ...] = (1, 5, 15)
    stability_window_minutes: int = 15
    concentration_top_n: int = 5
    epsilon: float = 1e-9
    fallback_order: tuple[str, ...] = _FALLBACK_ORDER_DEFAULT


@dataclass(frozen=True)
class StructuralState:
    """Expanded V3 structural state (§5.3)."""

    ts: str
    symbol: str
    spot: float

    # Open-interest-based dealer structure
    net_gex_oi: Optional[float] = None
    gamma_flip_oi: Optional[float] = None
    call_wall_oi: Optional[float] = None
    put_wall_oi: Optional[float] = None

    # Volume-based dealer structure
    net_gex_volume: Optional[float] = None
    gamma_flip_volume: Optional[float] = None
    call_wall_volume: Optional[float] = None
    put_wall_volume: Optional[float] = None

    # Hybrid dealer structure
    net_gex_hybrid: Optional[float] = None
    gamma_flip_hybrid: Optional[float] = None
    call_wall_hybrid: Optional[float] = None
    put_wall_hybrid: Optional[float] = None

    # Magnitude and concentration
    gex_percentile: Optional[float] = None
    gex_concentration: Optional[float] = None
    gex_hhi: Optional[float] = None
    largest_strike_share: Optional[float] = None
    top_three_strike_share: Optional[float] = None
    gex_disagreement: Optional[float] = None
    gex_sign_agreement: Optional[float] = None

    # Structural movement
    flip_velocity_1m: Optional[float] = None
    flip_velocity_5m: Optional[float] = None
    flip_velocity_15m: Optional[float] = None
    call_wall_velocity_5m: Optional[float] = None
    put_wall_velocity_5m: Optional[float] = None

    # Structural stability
    flip_stability: Optional[float] = None
    call_wall_stability: Optional[float] = None
    put_wall_stability: Optional[float] = None

    # Geometry normalized by expected remaining move
    distance_to_flip_expected_move: Optional[float] = None
    distance_to_call_wall_expected_move: Optional[float] = None
    distance_to_put_wall_expected_move: Optional[float] = None
    wall_channel_width_expected_move: Optional[float] = None

    # Quality and provenance
    source_ages: dict[str, float] = field(default_factory=dict)
    source_versions: dict[str, str] = field(default_factory=dict)
    quality_score: float = 0.0
    version: str = STRUCTURAL_STATE_VERSION
    diagnostics: dict = field(default_factory=dict)

    # ---- compatibility (§8) -------------------------------------------------
    def _fallback_level(self, field_stem: str) -> tuple[Optional[float], Optional[str]]:
        order = tuple(
            self.diagnostics.get("fallback_order") or _FALLBACK_ORDER_DEFAULT
        )
        attr_map = {
            "net_gex": "net_gex",
            "gamma_flip": "gamma_flip",
            "call_wall": "call_wall",
            "put_wall": "put_wall",
        }
        base = attr_map[field_stem]
        for src in order:
            val = getattr(self, f"{base}_{src}", None)
            if val is not None and _finite(val):
                return float(val), src
        return None, None

    @property
    def net_gex(self) -> Optional[float]:
        v, _ = self._fallback_level("net_gex")
        return v

    @property
    def gamma_flip(self) -> Optional[float]:
        v, _ = self._fallback_level("gamma_flip")
        return v

    @property
    def call_wall(self) -> Optional[float]:
        v, _ = self._fallback_level("call_wall")
        return v

    @property
    def put_wall(self) -> Optional[float]:
        v, _ = self._fallback_level("put_wall")
        return v

    def compatibility_provenance(self) -> dict[str, Optional[str]]:
        """Selected source for each compatibility property."""
        out: dict[str, Optional[str]] = {}
        for stem in ("net_gex", "gamma_flip", "call_wall", "put_wall"):
            _, src = self._fallback_level(stem)
            out[stem] = src
        return out

    def to_dict(self) -> dict:
        d = asdict(self)
        # Expose compatibility views without implying zeros
        d["net_gex"] = self.net_gex
        d["gamma_flip"] = self.gamma_flip
        d["call_wall"] = self.call_wall
        d["put_wall"] = self.put_wall
        d["compatibility_provenance"] = self.compatibility_provenance()
        return d

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "StructuralState":
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        payload = {k: v for k, v in d.items() if k in known}
        return cls(**payload)

    def to_legacy_policy_state(self):
        """
        Explicit V3 → legacy policy conversion (§8).

        Unavailable levels remain 0.0 on the legacy contract (historical
        gate convention) but provenance is recorded in `notes`.
        """
        from policy.contracts import StructuralState as LegacyStructuralState

        prov = self.compatibility_provenance()
        notes_parts = [
            f"v3_structural={self.version}",
            f"net_gex_src={prov.get('net_gex')}",
            f"flip_src={prov.get('gamma_flip')}",
            f"cw_src={prov.get('call_wall')}",
            f"pw_src={prov.get('put_wall')}",
        ]
        return LegacyStructuralState(
            spot=float(self.spot) if _finite(self.spot) else 0.0,
            net_gex=float(self.net_gex) if self.net_gex is not None else 0.0,
            gamma_flip=(
                float(self.gamma_flip) if self.gamma_flip is not None else 0.0
            ),
            call_wall=(
                float(self.call_wall) if self.call_wall is not None else 0.0
            ),
            put_wall=float(self.put_wall) if self.put_wall is not None else 0.0,
            gex_pct_rank=(
                float(self.gex_percentile)
                if self.gex_percentile is not None
                else 0.5
            ),
            notes=";".join(notes_parts),
        )


# ---- pure feature helpers (§6) -----------------------------------------------


def gex_disagreement(
    gex_oi: Optional[float],
    gex_volume: Optional[float],
    *,
    epsilon: float = 1e-9,
) -> Optional[float]:
    """Bounded [0, 1] disagreement between OI and volume GEX (§6.1)."""
    if gex_oi is None or gex_volume is None:
        return None
    if not (_finite(gex_oi) and _finite(gex_volume)):
        return None
    denom = abs(gex_oi) + abs(gex_volume) + epsilon
    val = abs(gex_oi - gex_volume) / denom
    return float(_clip(val, 0.0, 1.0))


def gex_sign_agreement(
    variants: Sequence[Optional[float]],
) -> Optional[float]:
    """
    Fraction of available finite variants sharing the majority sign (§6.1).

    Returns None when fewer than two variants are available.
    """
    signs = []
    for v in variants:
        if v is None or not _finite(v):
            continue
        if v > 0:
            signs.append(1)
        elif v < 0:
            signs.append(-1)
        else:
            signs.append(0)
    if len(signs) < 2:
        return None
    # Majority among non-zero if any; else zeros agree
    nonzero = [s for s in signs if s != 0]
    if not nonzero:
        return 1.0
    majority = 1 if nonzero.count(1) >= nonzero.count(-1) else -1
    agree = sum(1 for s in signs if s == majority or s == 0)
    return float(agree / len(signs))


def multi_variant_disagreement_stats(
    variants: Mapping[str, Optional[float]],
    *,
    epsilon: float = 1e-9,
) -> dict:
    """Extra disagreement diagnostics when ≥3 variants present (§6.1)."""
    vals = [(k, float(v)) for k, v in variants.items()
            if v is not None and _finite(v)]
    out: dict = {"n_variants": len(vals)}
    if len(vals) < 2:
        return out
    numbers = [v for _, v in vals]
    mean = sum(numbers) / len(numbers)
    if abs(mean) > epsilon:
        var = sum((x - mean) ** 2 for x in numbers) / len(numbers)
        out["coefficient_of_variation"] = float(math.sqrt(var) / abs(mean))
    else:
        out["coefficient_of_variation"] = None
    # Max normalized pairwise difference
    max_pair = 0.0
    for i in range(len(numbers)):
        for j in range(i + 1, len(numbers)):
            a, b = numbers[i], numbers[j]
            pair = abs(a - b) / (abs(a) + abs(b) + epsilon)
            if pair > max_pair:
                max_pair = pair
    out["max_normalized_pairwise_difference"] = float(_clip(max_pair, 0.0, 1.0))
    sorted_vals = sorted(numbers)
    mid = len(sorted_vals) // 2
    if len(sorted_vals) % 2:
        out["median_variant"] = float(sorted_vals[mid])
    else:
        out["median_variant"] = float(
            0.5 * (sorted_vals[mid - 1] + sorted_vals[mid]))
    out["range_across_variants"] = float(max(numbers) - min(numbers))
    out["sign_agreement"] = gex_sign_agreement([v for _, v in vals])
    return out


def concentration_metrics(
    abs_gamma_by_strike: Mapping[float, float],
    *,
    top_n: int = 5,
    epsilon: float = 1e-9,
) -> dict:
    """
    Concentration / HHI / strike-share metrics (§6.2).

    `abs_gamma_by_strike` maps strike → absolute gamma contribution.
    """
    if not abs_gamma_by_strike:
        return {
            "gex_concentration": None,
            "gex_hhi": None,
            "largest_strike_share": None,
            "top_three_strike_share": None,
            "top_five_strike_share": None,
            "n_strikes_80pct": None,
        }
    total = sum(abs(float(v)) for v in abs_gamma_by_strike.values())
    if total <= epsilon:
        return {
            "gex_concentration": None,
            "gex_hhi": None,
            "largest_strike_share": None,
            "top_three_strike_share": None,
            "top_five_strike_share": None,
            "n_strikes_80pct": None,
        }
    shares = sorted(
        (abs(float(v)) / total for v in abs_gamma_by_strike.values()),
        reverse=True,
    )
    top_n = max(1, int(top_n))
    conc = float(sum(shares[:top_n]))
    hhi = float(sum(s * s for s in shares))
    largest = float(shares[0])
    top3 = float(sum(shares[:3]))
    top5 = float(sum(shares[:5]))
    cum = 0.0
    n80 = len(shares)
    for i, s in enumerate(shares, start=1):
        cum += s
        if cum >= 0.80:
            n80 = i
            break
    return {
        "gex_concentration": float(_clip(conc, 0.0, 1.0)),
        "gex_hhi": float(_clip(hhi, 0.0, 1.0)),
        "largest_strike_share": float(_clip(largest, 0.0, 1.0)),
        "top_three_strike_share": float(_clip(top3, 0.0, 1.0)),
        "top_five_strike_share": float(_clip(top5, 0.0, 1.0)),
        "n_strikes_80pct": int(n80),
    }


def level_velocity(
    current: Optional[float],
    past: Optional[float],
    spot: float,
    *,
    epsilon: float = 1e-9,
) -> Optional[float]:
    """(current - past) / max(spot, eps) — as-of-safe (§6.3–6.4)."""
    if current is None or past is None:
        return None
    if not (_finite(current) and _finite(past) and _finite(spot)):
        return None
    return float((current - past) / max(abs(spot), epsilon))


def structural_stability(
    levels: Sequence[Optional[float]],
    expected_remaining_move: Optional[float],
    *,
    epsilon: float = 1e-9,
) -> Optional[float]:
    """
    1 - clip(rolling_std(level) / max(expected_move, eps), 0, 1) (§6.5).

    Uses only provided prior+current levels (caller must exclude future).
    """
    vals = [float(v) for v in levels if v is not None and _finite(v)]
    if len(vals) < 2:
        return None
    if expected_remaining_move is None or not _finite(expected_remaining_move):
        return None
    if expected_remaining_move < 0:
        return None
    mean = sum(vals) / len(vals)
    var = sum((x - mean) ** 2 for x in vals) / len(vals)
    std = math.sqrt(var)
    ratio = std / max(float(expected_remaining_move), epsilon)
    return float(1.0 - _clip(ratio, 0.0, 1.0))


def expected_move_geometry(
    spot: float,
    gamma_flip: Optional[float],
    call_wall: Optional[float],
    put_wall: Optional[float],
    expected_remaining_move: Optional[float],
    *,
    epsilon: float = 1e-9,
) -> dict:
    """Normalized distances to flip / walls (§6.6)."""
    out = {
        "distance_to_flip_expected_move": None,
        "distance_to_call_wall_expected_move": None,
        "distance_to_put_wall_expected_move": None,
        "wall_channel_width_expected_move": None,
    }
    if expected_remaining_move is None or not _finite(expected_remaining_move):
        return out
    if expected_remaining_move < 0 or not _finite(spot):
        return out
    denom = max(float(expected_remaining_move), epsilon)
    if gamma_flip is not None and _finite(gamma_flip):
        out["distance_to_flip_expected_move"] = float(
            (spot - gamma_flip) / denom)
    if call_wall is not None and _finite(call_wall):
        out["distance_to_call_wall_expected_move"] = float(
            (call_wall - spot) / denom)
    if put_wall is not None and _finite(put_wall):
        out["distance_to_put_wall_expected_move"] = float(
            (spot - put_wall) / denom)
    if (call_wall is not None and put_wall is not None
            and _finite(call_wall) and _finite(put_wall)):
        out["wall_channel_width_expected_move"] = float(
            (call_wall - put_wall) / denom)
    return out


# ---- builder (§7) ------------------------------------------------------------


class StructuralStateBuilder:
    """Deterministic as-of-safe StructuralState construction."""

    def __init__(self, cfg: Optional[StructuralStateConfig] = None):
        self.cfg = cfg or StructuralStateConfig()

    def build(
        self,
        *,
        ts: str,
        symbol: str,
        spot: float,
        expected_remaining_move: Optional[float],
        current_sources: Mapping[str, Mapping[str, Any]],
        historical_states: Sequence[Mapping[str, Any]],
        source_ages: Optional[Mapping[str, float]] = None,
        source_versions: Optional[Mapping[str, str]] = None,
        gex_percentile: Optional[float] = None,
    ) -> StructuralState:
        cfg = self.cfg
        diagnostics: dict[str, Any] = {
            "fallback_order": list(cfg.fallback_order),
            "missing_inputs": [],
            "invalid_geometry": [],
            "fallback_sources": {},
        }

        if not _finite(spot):
            diagnostics["invalid_geometry"].append("non_finite_spot")
        if (expected_remaining_move is not None
                and _finite(expected_remaining_move)
                and expected_remaining_move < 0):
            diagnostics["invalid_geometry"].append("negative_expected_remaining_move")
            expected_remaining_move = None

        # Extract per-variant levels (None if absent / non-finite)
        levels: dict[str, dict[str, Optional[float]]] = {}
        for variant in _VARIANT_KEYS:
            src = current_sources.get(variant) or {}
            if not src:
                if variant in ("oi", "volume", "hybrid"):
                    diagnostics["missing_inputs"].append(f"source:{variant}")
                levels[variant] = {
                    "net_gex": None, "gamma_flip": None,
                    "call_wall": None, "put_wall": None,
                }
                continue
            levels[variant] = {
                "net_gex": _opt_float(src.get("net_gex")),
                "gamma_flip": _opt_float(src.get("gamma_flip")),
                "call_wall": _opt_float(src.get("call_wall")),
                "put_wall": _opt_float(src.get("put_wall")),
            }
            cw = levels[variant]["call_wall"]
            pw = levels[variant]["put_wall"]
            if (cw is not None and pw is not None and cw < pw):
                diagnostics["invalid_geometry"].append(
                    f"call_wall_below_put_wall:{variant}")
            for key, val in levels[variant].items():
                if val is None and key in src and src.get(key) is not None:
                    diagnostics["invalid_geometry"].append(
                        f"non_finite_{variant}_{key}")

        # Concentration from preferred absolute-gamma map
        abs_map = None
        abs_source = None
        for pref in cfg.fallback_order:
            src = current_sources.get(pref) or {}
            raw = src.get("abs_gamma_by_strike")
            if isinstance(raw, Mapping) and raw:
                abs_map = {float(k): abs(float(v)) for k, v in raw.items()
                           if _finite(v)}
                abs_source = pref
                break
        conc = concentration_metrics(
            abs_map or {}, top_n=cfg.concentration_top_n, epsilon=cfg.epsilon)
        if abs_source:
            diagnostics["concentration_source"] = abs_source
        else:
            diagnostics["missing_inputs"].append("abs_gamma_by_strike")
            # Fall back to precomputed concentration if provided
            for pref in cfg.fallback_order:
                src = current_sources.get(pref) or {}
                c = _opt_float(src.get("gex_concentration"))
                if c is not None:
                    conc["gex_concentration"] = float(_clip(c, 0.0, 1.0))
                    diagnostics["concentration_source"] = f"{pref}:precomputed"
                    break

        # Disagreement (OI vs volume) + multi-variant stats
        disagree = gex_disagreement(
            levels["oi"]["net_gex"], levels["volume"]["net_gex"],
            epsilon=cfg.epsilon)
        sign_agree = gex_sign_agreement([
            levels["oi"]["net_gex"],
            levels["volume"]["net_gex"],
            levels["hybrid"]["net_gex"],
        ])
        variant_gex = {
            k: levels[k]["net_gex"] for k in ("oi", "volume", "hybrid", "weekly")
        }
        diagnostics["multi_variant"] = multi_variant_disagreement_stats(
            variant_gex, epsilon=cfg.epsilon)

        # Compatibility provenance (recorded even before object exists)
        fallback_sources: dict[str, Optional[str]] = {}
        for stem in ("net_gex", "gamma_flip", "call_wall", "put_wall"):
            chosen = None
            for src in cfg.fallback_order:
                val = levels.get(src, {}).get(stem)
                if val is not None:
                    chosen = src
                    break
            fallback_sources[stem] = chosen
        diagnostics["fallback_sources"] = fallback_sources

        # Velocities from historical_states (prior only — caller responsibility)
        hist = list(historical_states)
        flip_v = {}
        for window in cfg.velocity_windows:
            past = _state_at_least_ago(hist, ts, window)
            flip_v[window] = level_velocity(
                _compat_level(levels, "gamma_flip", cfg.fallback_order),
                None if past is None else _hist_level(past, "gamma_flip",
                                                      cfg.fallback_order),
                spot, epsilon=cfg.epsilon,
            )
        past5 = _state_at_least_ago(hist, ts, 5)
        cw_vel = level_velocity(
            _compat_level(levels, "call_wall", cfg.fallback_order),
            None if past5 is None else _hist_level(past5, "call_wall",
                                                   cfg.fallback_order),
            spot, epsilon=cfg.epsilon,
        )
        pw_vel = level_velocity(
            _compat_level(levels, "put_wall", cfg.fallback_order),
            None if past5 is None else _hist_level(past5, "put_wall",
                                                   cfg.fallback_order),
            spot, epsilon=cfg.epsilon,
        )

        # Stability over configured window using prior+current levels
        stab_hist = _states_within(hist, ts, cfg.stability_window_minutes)
        flip_levels = [
            _hist_level(h, "gamma_flip", cfg.fallback_order) for h in stab_hist
        ] + [_compat_level(levels, "gamma_flip", cfg.fallback_order)]
        cw_levels = [
            _hist_level(h, "call_wall", cfg.fallback_order) for h in stab_hist
        ] + [_compat_level(levels, "call_wall", cfg.fallback_order)]
        pw_levels = [
            _hist_level(h, "put_wall", cfg.fallback_order) for h in stab_hist
        ] + [_compat_level(levels, "put_wall", cfg.fallback_order)]

        geom = expected_move_geometry(
            spot,
            _compat_level(levels, "gamma_flip", cfg.fallback_order),
            _compat_level(levels, "call_wall", cfg.fallback_order),
            _compat_level(levels, "put_wall", cfg.fallback_order),
            expected_remaining_move,
            epsilon=cfg.epsilon,
        )

        quality = _quality_score(
            levels=levels,
            disagree=disagree,
            invalid=diagnostics["invalid_geometry"],
            missing=diagnostics["missing_inputs"],
            source_ages=source_ages or {},
        )
        diagnostics["quality_components"] = {
            "n_variants_present": sum(
                1 for v in ("oi", "volume", "hybrid")
                if levels[v]["net_gex"] is not None
            ),
        }

        ages = {str(k): float(v) for k, v in (source_ages or {}).items()
                if _finite(v)}
        versions = {str(k): str(v)
                    for k, v in (source_versions or {}).items()}

        return StructuralState(
            ts=str(ts),
            symbol=str(symbol),
            spot=float(spot) if _finite(spot) else float("nan"),
            net_gex_oi=levels["oi"]["net_gex"],
            gamma_flip_oi=levels["oi"]["gamma_flip"],
            call_wall_oi=levels["oi"]["call_wall"],
            put_wall_oi=levels["oi"]["put_wall"],
            net_gex_volume=levels["volume"]["net_gex"],
            gamma_flip_volume=levels["volume"]["gamma_flip"],
            call_wall_volume=levels["volume"]["call_wall"],
            put_wall_volume=levels["volume"]["put_wall"],
            net_gex_hybrid=levels["hybrid"]["net_gex"],
            gamma_flip_hybrid=levels["hybrid"]["gamma_flip"],
            call_wall_hybrid=levels["hybrid"]["call_wall"],
            put_wall_hybrid=levels["hybrid"]["put_wall"],
            gex_percentile=_opt_float(gex_percentile),
            gex_concentration=conc.get("gex_concentration"),
            gex_hhi=conc.get("gex_hhi"),
            largest_strike_share=conc.get("largest_strike_share"),
            top_three_strike_share=conc.get("top_three_strike_share"),
            gex_disagreement=disagree,
            gex_sign_agreement=sign_agree,
            flip_velocity_1m=flip_v.get(1),
            flip_velocity_5m=flip_v.get(5),
            flip_velocity_15m=flip_v.get(15),
            call_wall_velocity_5m=cw_vel,
            put_wall_velocity_5m=pw_vel,
            flip_stability=structural_stability(
                flip_levels, expected_remaining_move, epsilon=cfg.epsilon),
            call_wall_stability=structural_stability(
                cw_levels, expected_remaining_move, epsilon=cfg.epsilon),
            put_wall_stability=structural_stability(
                pw_levels, expected_remaining_move, epsilon=cfg.epsilon),
            distance_to_flip_expected_move=geom[
                "distance_to_flip_expected_move"],
            distance_to_call_wall_expected_move=geom[
                "distance_to_call_wall_expected_move"],
            distance_to_put_wall_expected_move=geom[
                "distance_to_put_wall_expected_move"],
            wall_channel_width_expected_move=geom[
                "wall_channel_width_expected_move"],
            source_ages=ages,
            source_versions=versions,
            quality_score=float(quality),
            version=STRUCTURAL_STATE_VERSION,
            diagnostics={
                **diagnostics,
                "top_five_strike_share": conc.get("top_five_strike_share"),
                "n_strikes_80pct": conc.get("n_strikes_80pct"),
            },
        )


def sources_from_gex_bundle(bundle) -> dict[str, dict]:
    """Lift a `gex.contracts.GexVariantBundle` into builder current_sources."""
    out: dict[str, dict] = {}
    for name, snap in (
        ("oi", getattr(bundle, "oi", None)),
        ("weekly", getattr(bundle, "weekly", None)),
        ("volume", getattr(bundle, "volume", None)),
        ("hybrid", getattr(bundle, "hybrid", None)),
    ):
        if snap is None:
            continue
        if not getattr(snap, "is_finite", True):
            # Still record walls/flip if finite even when net_gex is nan
            pass
        entry = {
            "net_gex": _opt_float(getattr(snap, "net_gex", None)),
            "gamma_flip": _opt_float(getattr(snap, "gamma_flip", None)),
            "call_wall": _opt_float(getattr(snap, "call_wall", None)),
            "put_wall": _opt_float(getattr(snap, "put_wall", None)),
            "gex_concentration": _opt_float(
                getattr(snap, "gex_concentration", None)),
            "quality_score": _opt_float(getattr(snap, "quality_score", None)),
        }
        out[name] = entry
    return out


# ---- internals ---------------------------------------------------------------


def _finite(x) -> bool:
    try:
        return x is not None and math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def _opt_float(x) -> Optional[float]:
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v):
        return None
    return v


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _compat_level(
    levels: Mapping[str, Mapping[str, Optional[float]]],
    stem: str,
    order: Sequence[str],
) -> Optional[float]:
    for src in order:
        val = levels.get(src, {}).get(stem)
        if val is not None:
            return val
    return None


def _hist_level(
    hist: Mapping[str, Any],
    stem: str,
    order: Sequence[str],
) -> Optional[float]:
    # Prefer explicit compatibility key, then per-variant
    direct = _opt_float(hist.get(stem))
    if direct is not None:
        return direct
    for src in order:
        val = _opt_float(hist.get(f"{stem}_{src}"))
        if val is not None:
            return val
    return None


def _parse_ts_minutes(ts: str) -> Optional[float]:
    """Best-effort epoch-minutes from ISO timestamp or numeric string."""
    if ts is None:
        return None
    try:
        return float(ts)
    except (TypeError, ValueError):
        pass
    try:
        from datetime import datetime
        s = str(ts).replace("Z", "+00:00")
        dt = datetime.fromisoformat(s)
        return dt.timestamp() / 60.0
    except (TypeError, ValueError):
        return None


def _state_at_least_ago(
    hist: Sequence[Mapping[str, Any]],
    ts: str,
    minutes: int,
) -> Optional[Mapping[str, Any]]:
    """Newest historical state at least `minutes` before `ts`."""
    now_m = _parse_ts_minutes(ts)
    if now_m is None:
        return None
    target = now_m - float(minutes)
    best = None
    best_t = None
    for h in hist:
        ht = _parse_ts_minutes(str(h.get("ts", "")))
        if ht is None:
            continue
        if ht > now_m:
            # Refuse future-updated values during reconstruction
            continue
        if ht <= target and (best_t is None or ht > best_t):
            best = h
            best_t = ht
    return best


def _states_within(
    hist: Sequence[Mapping[str, Any]],
    ts: str,
    minutes: int,
) -> list[Mapping[str, Any]]:
    now_m = _parse_ts_minutes(ts)
    if now_m is None:
        return []
    lo = now_m - float(minutes)
    out = []
    for h in hist:
        ht = _parse_ts_minutes(str(h.get("ts", "")))
        if ht is None or ht > now_m or ht < lo:
            continue
        out.append(h)
    return out


def _quality_score(
    *,
    levels: Mapping[str, Mapping[str, Optional[float]]],
    disagree: Optional[float],
    invalid: Sequence[str],
    missing: Sequence[str],
    source_ages: Mapping[str, float],
) -> float:
    score = 1.0
    present = sum(
        1 for v in ("oi", "volume", "hybrid")
        if levels.get(v, {}).get("net_gex") is not None
    )
    if present == 0:
        return 0.0
    score *= present / 3.0
    if disagree is not None:
        score *= 1.0 - 0.5 * float(disagree)
    if invalid:
        score *= max(0.0, 1.0 - 0.15 * len(invalid))
    if missing:
        score *= max(0.0, 1.0 - 0.05 * len(missing))
    # Soft age penalty
    ages = [float(a) for a in source_ages.values() if _finite(a)]
    if ages:
        max_age = max(ages)
        if max_age > 60:
            score *= 0.7
        elif max_age > 30:
            score *= 0.85
    return float(_clip(score, 0.0, 1.0))
