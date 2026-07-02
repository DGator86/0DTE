"""MCConfig: the regime knobs are injectable, defaults preserve behavior."""
from __future__ import annotations

import mc


def test_default_cfg_matches_implicit():
    kw = dict(spot=600, target=602, stop=599.5, flip=599.5, minutes_left=120,
              iv_annual=0.13, regime="trend", win_R=2.0, seed=1)
    assert mc.project(**kw) == mc.project(cfg=mc.MCConfig(), **kw)


def test_custom_cfg_changes_dynamics():
    kw = dict(spot=600, lower_short=599, upper_short=602, flip=600.0,
              minutes_left=120, iv_annual=0.10, regime="pin", win_R=0.27, seed=1)
    base = mc.project_range(**kw)
    loose = mc.project_range(cfg=mc.MCConfig(pin_vol_mult=1.4, pin_revert_k=0.0), **kw)
    # weaker pin + more vol -> more range breaches -> lower survival probability
    assert loose.p_target < base.p_target
