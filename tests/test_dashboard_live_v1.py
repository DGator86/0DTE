"""
PR D — dashboard consumes live.v1 only.

Fixture-driven contract checks for app.js / index.html (no browser runtime):
schema gate, section accessors, feed badges, absence of flat aliases and
cross-version fallbacks.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from dashboard.live_schema import LIVE_SCHEMA_VERSION, validate_live_v1

ROOT = Path(__file__).resolve().parent.parent
APP_JS = ROOT / "dashboard" / "static" / "app.js"
INDEX = ROOT / "dashboard" / "static" / "index.html"
FIXTURE = Path(__file__).resolve().parent / "fixtures" / "live_state_baseline.json"


def _js() -> str:
    return APP_JS.read_text(encoding="utf-8")


def _html() -> str:
    return INDEX.read_text(encoding="utf-8")


def test_fixture_is_valid_live_v1_without_flat_aliases():
    fixture = json.loads(FIXTURE.read_text())
    payload = fixture["payload"]
    assert payload["schema_version"] == LIVE_SCHEMA_VERSION
    assert validate_live_v1(payload) == []
    for flat in ("doing", "why", "inputs", "v2_signals", "parallel", "part3",
                 "ts", "status", "note", "feed_source", "chain_available",
                 "sigma_cones"):
        assert flat not in payload
    assert payload["system"]["compat_flat_keys"] is False
    assert "overall_status" in payload["feeds"]
    for src in ("spot", "bars", "option_chain", "settlement"):
        assert src in payload["feeds"]
        assert "status" in payload["feeds"][src]


def test_appjs_schema_gate_and_section_accessors():
    js = _js()
    for name in (
        "requireLiveV1", "showLiveUnavailable", "liveTs", "liveStatus",
        "liveNote", "liveDoing", "liveWhy", "liveInputs", "liveParallel",
        "liveV2Signals", "livePart3", "liveSigmaCones", "liveFeedDown",
        "liveIdle", "feedStatusCls",
    ):
        assert f"function {name}(" in js, name
    assert "LIVE_SCHEMA_VERSION" in js
    assert 'schema_version !== LIVE_SCHEMA_VERSION' in js
    assert "live = requireLiveV1(live)" in js
    assert len(re.findall(r"setInterval\(refresh\b", js)) == 1


def test_appjs_feed_badge_from_feeds_not_truthy_chain():
    js = _js()
    html = _html()
    assert 'id="meta-feeds"' in html
    assert 'id="meta-feeds-detail"' in html
    assert 'id="meta-feed"' not in html  # old single Feed chip removed
    assert "feeds.overall_status" in js
    assert "option_chain" in js
    assert "settlement" in js
    assert "chain_available ?" not in js
    assert "live.chain_available" not in js
    assert "feedStatusCls" in js


def test_appjs_no_flat_alias_reads():
    js = _js()
    # Direct property reads of removed flat aliases
    for pat in (
        r"\blive\.doing\b",
        r"\blive\.why\b",
        r"\blive\.inputs\b",
        r"\blive\.v2_signals\b",
        r"\blive\.parallel\b",
        r"\blive\.part3\b",
        r"\blive\.sigma_cones\b",
        r"\blive\.feed_source\b",
        r"\blive\.chain_available\b",
        r"\blive\.ts\b",
        r"\blive\.status\b",
        r"\blive\.note\b",
    ):
        assert re.search(pat, js) is None, pat


def test_appjs_no_cross_version_policy_or():
    js = _js()
    for needle in (
        "v2_policy_structure || s.policy_structure",
        "v2_policy_action || s.policy_action",
        "v2_policy_direction || s.policy_direction",
        "v2.action || s.v2_policy_action || s.policy_action",
        "v2.structure || s.v2_policy_structure || s.policy_structure",
    ):
        assert needle not in js, needle


def test_v1_panels_still_use_legacy_section():
    """Legacy signal / why / regime still wired through liveDoing / liveWhy."""
    js = _js()
    assert "function renderSignal(live)" in js
    assert "liveDoing(live)" in js
    assert "liveWhy(live)" in js
    assert "function renderWhy(live)" in js
    assert "function renderRegime(live)" in js
