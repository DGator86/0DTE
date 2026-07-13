(function () {
  "use strict";

  /* ============================================================
     0DTE COMMAND CENTER — single-page, read-only, outputs only.
     ============================================================ */

  const TOKEN_KEY = "zerodte_dashboard_token";
  const REFRESH_MS = 15000;

  let marketAnchor = null;
  let countdownTimer = null;
  let refreshTimer = null;
  let lastChartData = null; // {ticks, live, market, trades} kept for resize redraws

  // Chart time-axis zoom/pan (screen-space transform over the base layout).
  // z=1 pan=0 is the classic full-session view; wheel/pinch/buttons zoom,
  // drag pans. Marker hit-boxes are rebuilt on every draw for the tooltip.
  const chartView = { z: 1, pan: 0 };
  const CH_PADL = 8, CH_PADR = 62;
  let chartHits = [];               // [{x, y, label}] in CSS px, current draw

  function chartPlotW() {
    const cv = $("chart");
    return cv ? cv.clientWidth - CH_PADL - CH_PADR : 0;
  }

  function clampChartView() {
    chartView.z = Math.max(1, Math.min(40, chartView.z));
    const maxPan = chartPlotW() * (chartView.z - 1);
    chartView.pan = Math.max(0, Math.min(maxPan, chartView.pan));
  }

  function chartZoomAt(xCss, factor) {
    const anchor = (xCss - CH_PADL + chartView.pan) / chartView.z;
    chartView.z *= factor;
    chartView.z = Math.max(1, Math.min(40, chartView.z));
    chartView.pan = anchor * chartView.z - (xCss - CH_PADL);
    clampChartView();
    drawChart();
  }

  /* ---------------- auth / fetch ---------------- */
  function getToken() {
    const params = new URLSearchParams(window.location.search);
    const q = params.get("token");
    if (q) {
      sessionStorage.setItem(TOKEN_KEY, q);
      params.delete("token");
      const clean = params.toString();
      history.replaceState({}, "", window.location.pathname + (clean ? "?" + clean : ""));
    }
    return sessionStorage.getItem(TOKEN_KEY) || "";
  }

  function authHeaders() {
    const token = getToken();
    return token ? { Authorization: "Bearer " + token } : {};
  }

  async function api(path) {
    const res = await fetch(path, { headers: authHeaders() });
    if (res.status === 401) {
      sessionStorage.removeItem(TOKEN_KEY);
      showAuth();
      throw new Error("Unauthorized");
    }
    if (!res.ok) throw new Error(await res.text());
    return res.json();
  }

  const $ = (id) => document.getElementById(id);
  function showAuth() { $("auth-gate").classList.remove("hidden"); $("app").classList.add("hidden"); }
  function showApp()  { $("auth-gate").classList.add("hidden");  $("app").classList.remove("hidden"); }

  /* ---------------- formatting helpers ---------------- */
  const num = (v) => (typeof v === "number" && isFinite(v) ? v : (v != null && isFinite(+v) ? +v : null));
  function fmt(v, d = 2) { const n = num(v); return n == null ? "—" : n.toFixed(d); }
  function money(v, d = 0) { const n = num(v); return n == null ? "—" : (n < 0 ? "-$" : "$") + Math.abs(n).toFixed(d); }
  function pct(v, d = 1) { const n = num(v); return n == null ? "—" : (n * 100).toFixed(d) + "%"; }
  function sign(v, d = 2) { const n = num(v); if (n == null) return "—"; return (n >= 0 ? "+" : "") + n.toFixed(d); }
  function esc(s) {
    return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;")
      .replace(/>/g, "&gt;").replace(/"/g, "&quot;");
  }
  function compact(v) {
    const n = num(v);
    if (n == null) return "—";
    const a = Math.abs(n);
    if (a >= 1e9) return (n / 1e9).toFixed(2) + "B";
    if (a >= 1e6) return (n / 1e6).toFixed(2) + "M";
    if (a >= 1e3) return (n / 1e3).toFixed(1) + "K";
    return n.toFixed(2);
  }
  function etTime(iso) {
    if (!iso) return "—";
    return new Date(iso).toLocaleTimeString("en-US", { timeZone: "America/New_York", hour: "2-digit", minute: "2-digit" });
  }
  function fmtDuration(sec) {
    const s = Math.max(0, Math.floor(sec));
    const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), ss = s % 60;
    const p = [];
    if (h) p.push(h + "h");
    p.push(m + "m");
    p.push(String(ss).padStart(2, "0") + "s");
    return p.join(" ");
  }
  function arr(x) { return Array.isArray(x) ? x : (x == null ? [] : [x]); }
  function strikes(x) { const a = arr(x); return a.length ? a.join("/") : "—"; }

  function metricCard(k, v, cls) {
    return `<div class="metric"><span class="k">${esc(k)}</span><span class="v${cls ? " " + cls : ""}">${v}</span></div>`;
  }

  /* ---------------- top bar ---------------- */
  function renderTopbar(live, market, ticks) {
    const inp = live.inputs || {};
    const spot = num(inp.spot);
    $("spot-px").textContent = spot != null ? spot.toFixed(2) : "—";

    // session change vs first tick of the day
    const first = ticks.find((t) => num(t.spot) != null);
    const base = first ? num(first.spot) : null;
    const chgEl = $("spot-chg");
    if (spot != null && base != null && base !== 0) {
      const diff = spot - base;
      chgEl.textContent = `${sign(diff)} (${sign((diff / base) * 100)}%)`;
      chgEl.style.color = diff >= 0 ? "var(--green)" : "var(--red)";
    } else {
      chgEl.textContent = "";
    }

    $("meta-feed").textContent = live.feed_source || "—";
    $("meta-chain").textContent = live.chain_available ? "live" : "n/a";

    // freshness
    const freshEl = $("meta-fresh");
    const wrap = $("meta-fresh-wrap");
    if (live.ts) {
      const age = (Date.now() - new Date(live.ts).getTime()) / 1000;
      freshEl.textContent = age < 90 ? etTime(live.ts) : Math.floor(age / 60) + "m ago";
      wrap.classList.toggle("warn", age > 180);
    } else {
      freshEl.textContent = "—";
      wrap.classList.remove("warn");
    }
  }

  function renderMarketPill(status) {
    const pill = $("market-pill");
    pill.classList.toggle("open", !!status.is_open);
    pill.classList.toggle("closed", !status.is_open);
    $("market-label").textContent = status.is_open
      ? (status.session_type === "early_close" ? "Early close" : "Market open")
      : "Market closed";
    marketAnchor = {
      is_open: status.is_open,
      fetchedAt: Date.now(),
      secondsRemaining: status.is_open ? status.seconds_until_close : status.seconds_until_open,
      next_close: status.next_close,
    };
    tickCountdown();
  }

  function tickCountdown() {
    if (!marketAnchor) return;
    const elapsed = (Date.now() - marketAnchor.fetchedAt) / 1000;
    const remaining = Math.max(0, (marketAnchor.secondsRemaining || 0) - elapsed);
    $("market-countdown").textContent =
      (marketAnchor.is_open ? "· closes " : "· opens ") + fmtDuration(remaining);
    if (remaining <= 0) loadMarketStatus();
  }

  async function loadMarketStatus() {
    try { renderMarketPill(await api("/api/market-status")); }
    catch (e) { /* ignore */ }
  }

  /* ---------------- signal / verdict ---------------- */
  function renderSignal(live) {
    const d = live.doing || {};
    const w = live.why || {};
    $("signal-time").textContent = live.ts ? etTime(live.ts) + " ET" : "—";

    const feedDown = live.status === "feed_not_ready" || live.status === "feed_error";
    const idle = !live.ts || (live.status && live.status !== "live");
    let cls = "wait", word = "WAIT", sub = "";
    if (feedDown) {
      cls = "stop"; word = "NO FEED";
      sub = "feed not ready — check data source";
    } else if (idle) {
      cls = "wait"; word = "STANDBY";
      sub = (live.market && live.market.is_open) ? "pipeline idle — awaiting tick" : "market closed — awaiting session";
    } else if (d.stand_down) {
      cls = "stop"; word = "STAND DOWN";
      sub = (w.stand_down_reason || "regime veto").replace(/_/g, " ");
    } else if (d.decision === "TRADE" && d.gate_pass) {
      cls = "go"; word = "TRADE";
      sub = "gate passed · engine armed";
    } else if (d.decision === "NO_TRADE" || d.structure === "NT") {
      cls = "wait"; word = "NO TRADE";
      sub = (w.no_trade_reason || "conditions not met").replace(/_/g, " ");
    } else {
      cls = "wait"; word = d.decision || "WAIT";
      sub = d.permitted_engine ? "engine: " + d.permitted_engine : "monitoring";
    }
    const v = $("verdict");
    v.className = "verdict " + cls;
    $("verdict-word").textContent = word;
    $("verdict-sub").textContent = sub || "—";

    // direction / structure chips
    const dir = (d.direction || "").toLowerCase();
    const dirCls = dir.includes("call") || dir.includes("bull") ? "call"
      : dir.includes("put") || dir.includes("bear") ? "put" : "";
    const chips = [];
    if (d.structure && d.structure !== "NT") chips.push(`<span class="tag-chip big">${esc(d.structure)}</span>`);
    if (d.direction) chips.push(`<span class="tag-chip big ${dirCls}">${esc(d.direction)}</span>`);
    if (d.conviction) chips.push(`<span class="tag-chip">${esc(d.conviction)}</span>`);
    if (d.dominant_regime) chips.push(`<span class="tag-chip">${esc(d.dominant_regime)}</span>`);
    $("dirline").innerHTML = chips.join("");

    // gate gauge
    const gate = num(d.gate_score);
    $("gate-num").textContent = gate != null ? gate.toFixed(1) : "—";
    const gatePctW = gate != null ? Math.max(0, Math.min(100, gate)) : 0;
    const gateBar = $("gate-bar");
    gateBar.style.width = gatePctW + "%";
    gateBar.parentElement.className = "bar " + (d.gate_pass ? "green" : gate >= 40 ? "amber" : "red");

    // size gauge (mult typically 0..1.5)
    const size = num(d.final_size_mult);
    $("size-num").textContent = size != null ? "×" + size.toFixed(2) : "—";
    $("size-bar").style.width = (size != null ? Math.max(0, Math.min(100, (size / 1.5) * 100)) : 0) + "%";
  }

  /* ---------------- playbook (latest journaled tick) ---------------- */
  function renderPlaybook(latest, live) {
    const t = latest || {};
    const inp = live.inputs || {};
    const s = tickSignals(t) || {};
    $("playbook-fam").textContent = t.selected_family || (live.doing && live.doing.structure) || "—";

    // The candidate journaled on a NO_TRADE tick is the measurement loop's
    // would-be pick (kept so settlement can score its hypothetical P&L).
    // It can be a degenerate penny structure — flag it so it never reads
    // as a trade recommendation.
    const isDiagnostic = !!t.selected_family && t.decision !== "TRADE" && !num(t.was_traded);
    $("playbook-diag").hidden = !isDiagnostic;

    const evCls = num(t.ev) > 0 ? "pos" : num(t.ev) < 0 ? "neg" : "";
    const liveEv = num(s.phys_live_ev) != null ? num(s.phys_live_ev) : num(t.ev);
    const shadowEv = num(s.phys_v2_shadow_ev);
    const delta = (liveEv != null && shadowEv != null) ? shadowEv - liveEv : null;
    const cards = [
      metricCard("Short", strikes(t.short_strikes)),
      metricCard("Long", strikes(t.long_strikes)),
      metricCard("Credit", money(t.credit, 2) === "—" ? "—" : "$" + fmt(t.credit, 2)),
      metricCard("Max loss", money(t.max_loss, 0), "neg"),
      metricCard("Exp. value", money(t.ev, 0), evCls),
      metricCard("EV / risk", fmt(t.ev_per_risk, 2), num(t.ev_per_risk) > 0 ? "pos" : ""),
      metricCard("Prob. profit", pct(t.prob_profit)),
      metricCard("Prob. touch", pct(t.prob_touch_short), "warn"),
      metricCard("Theta", fmt(t.theta, 2), "pos"),
      metricCard("Gamma", fmt(t.gamma, 3)),
      metricCard("Cand. score", fmt(t.candidate_score, 1)),
      metricCard("Breakeven", fmt(inp.straddle_breakeven, 2)),
      metricCard("Density mode", esc(s.phys_density_mode || "—"), "info"),
      metricCard("Live EV", money(liveEv, 0), num(liveEv) > 0 ? "pos" : num(liveEv) < 0 ? "neg" : ""),
      metricCard("V2 shadow EV", money(shadowEv, 0), num(shadowEv) > 0 ? "pos" : num(shadowEv) < 0 ? "neg" : ""),
      metricCard("V2 − live Δ", money(delta, 0), num(delta) > 0 ? "pos" : num(delta) < 0 ? "neg" : ""),
      metricCard("V2 var ratio", fmt(s.phys_v2_var_ratio, 2)),
      metricCard("V2 uncertainty", fmt(s.phys_v2_uncertainty, 2),
        num(s.phys_v2_uncertainty) > 0.5 ? "warn" : ""),
    ];
    if (num(s.v2_utility_score) != null) {
      cards.push(metricCard("V2 utility", fmt(s.v2_utility_score, 2),
        num(s.v2_rank_disagreement) ? "warn" : "pos"));
    }
    if (num(t.liquidity_score) != null) {
      cards.push(metricCard("Liquidity", fmt(t.liquidity_score, 2)));
    }
    if (num(t.credit_expected) != null) {
      cards.push(metricCard("Credit exp.", "$" + fmt(t.credit_expected, 2)));
    }
    $("playbook-metrics").innerHTML = cards.join("");
  }

  /* ---------------- parallel Legacy vs V2 ---------------- */
  function renderParallel(live, latest) {
    const p = (live && live.parallel) || {};
    const leg = p.legacy || {};
    const v2 = p.v2 || {};
    const s = tickSignals(latest) || (live && live.v2_signals) || {};
    const mode = v2.mode || s.policy_mode || "—";
    $("parallel-mode").textContent = mode;
    const disagree = num(v2.disagreement) === 1 || num(s.policy_disagreement) === 1;
    const fallback = num(v2.fallback_used) === 1 || num(s.policy_fallback_used) === 1;

    $("parallel-compare").innerHTML = `
      <div class="policy-side${disagree ? " disagree" : ""}">
        <div class="ps-label">Legacy (matrix)</div>
        <div class="ps-struct">${esc(leg.structure || "NT")}</div>
        <div class="ps-meta">${esc(leg.decision || "—")}${leg.direction ? " · " + esc(leg.direction) : ""}</div>
      </div>
      <div class="policy-side${disagree ? " disagree" : ""}">
        <div class="ps-label">V2 policy</div>
        <div class="ps-struct">${esc(v2.structure || s.v2_policy_structure || "—")}</div>
        <div class="ps-meta">${esc(v2.action || s.v2_policy_action || "—")}${(v2.direction || s.v2_policy_direction) ? " · " + esc(v2.direction || s.v2_policy_direction) : ""}</div>
      </div>`;

    $("parallel-metrics").innerHTML = [
      metricCard("Legacy gate", leg.gate_pass == null ? "—" : (leg.gate_pass ? "PASS" : "FAIL"),
        leg.gate_pass ? "pos" : (leg.gate_pass === false ? "warn" : "")),
      metricCard("Legacy size", fmt(leg.size_mult, 2)),
      metricCard("V2 size cap", fmt(v2.size_cap != null ? v2.size_cap : s.policy_size_cap, 2)),
      metricCard("V2 conf.", fmt(v2.confidence != null ? v2.confidence : s.v2_policy_confidence, 2)),
      metricCard("V2 unc.", fmt(v2.uncertainty != null ? v2.uncertainty : s.v2_policy_uncertainty, 2),
        num(v2.uncertainty != null ? v2.uncertainty : s.v2_policy_uncertainty) > 0.5 ? "warn" : ""),
      metricCard("Source", esc(v2.source || s.policy_source || "—")),
    ].join("");

    const chips = [];
    if (disagree) chips.push('<span class="chip disagree">disagreement</span>');
    if (fallback) chips.push('<span class="chip fallback">fallback legacy</span>');
    if (!chips.length) chips.push('<span class="chip ok">aligned / dual-run</span>');
    $("parallel-chips").innerHTML = chips.join("");
  }

  function renderPhysDensity(latest, live) {
    const s = mergeV2Signals(latest, live);
    $("phys-mode").textContent = s.phys_density_mode || "—";
    if (s.phys_density_mode == null && s.phys_v2_mean == null && s.phys_live_ev == null) {
      $("phys-metrics").innerHTML = '<p class="empty">No physical-density signals yet</p>';
      return;
    }
    $("phys-metrics").innerHTML = [
      metricCard("Mode", esc(s.phys_density_mode || "—")),
      metricCard("Live EV", fmt(s.phys_live_ev, 3)),
      metricCard("V2 shadow EV", fmt(s.phys_v2_shadow_ev, 3)),
      metricCard("V2 mean", fmt(s.phys_v2_mean, 4)),
      metricCard("V2 std", fmt(s.phys_v2_std, 4)),
      metricCard("Var ratio", fmt(s.phys_v2_var_ratio, 3)),
      metricCard("Exp. return", fmt(s.phys_v2_expected_return, 5)),
      metricCard("Uncertainty", fmt(s.phys_v2_uncertainty, 2)),
    ].join("");
  }

  function renderRanker(latest, live) {
    const s = mergeV2Signals(latest, live);
    const has = s.v2_top_candidate_id != null || s.v2_rank_disagreement != null
      || s.v2_utility_score != null;
    if (!has) {
      $("ranker-metrics").innerHTML = '<p class="empty">No candidate-ranker signals yet</p>';
      return;
    }
    $("ranker-metrics").innerHTML = [
      metricCard("Disagreement", s.v2_rank_disagreement == null ? "—"
        : (s.v2_rank_disagreement ? "yes" : "no"),
        s.v2_rank_disagreement ? "warn" : "pos"),
      metricCard("V2 top", esc(String(s.v2_top_candidate_id || "—").slice(0, 12))),
      metricCard("Legacy top", esc(String(s.legacy_top_candidate_id || "—").slice(0, 12))),
      metricCard("V2 utility", fmt(s.v2_utility_score, 3)),
      metricCard("V2 top family", esc(s.v2_top_family || "—")),
      metricCard("Legacy top fam.", esc(s.legacy_top_family || "—")),
    ].join("");
  }

  function renderV2Playbook(latest, live) {
    const s = mergeV2Signals(latest, live);
    const t = latest || {};
    const cards = [];
    if (num(s.phys_v2_shadow_ev) != null) {
      cards.push(metricCard("V2 shadow EV", fmt(s.phys_v2_shadow_ev, 3)));
    }
    if (num(t.v2_utility_score) != null || num(s.v2_utility_score) != null) {
      cards.push(metricCard("Utility", fmt(t.v2_utility_score != null
        ? t.v2_utility_score : s.v2_utility_score, 3)));
    }
    if (num(t.credit_expected) != null) {
      cards.push(metricCard("Credit exp.", "$" + fmt(t.credit_expected, 2)));
    }
    if (num(s.gex_rank_warm) != null) {
      cards.push(metricCard("GEX warm", num(s.gex_rank_warm) ? "yes" : "no",
        num(s.gex_rank_warm) ? "pos" : "warn"));
    }
    $("v2-playbook-metrics").innerHTML = cards.length
      ? cards.join("")
      : '<p class="empty">No V2 economics on latest tick</p>';
  }
  function renderPolicy(latest) {
    const s = tickSignals(latest) || {};
    const mode = s.policy_mode || "—";
    $("policy-mode").textContent = mode;
    const disagree = num(s.policy_disagreement) === 1;
    const fallback = num(s.policy_fallback_used) === 1;
    const legStruct = s.legacy_policy_structure || s.policy_structure || "—";
    const legAct = s.legacy_policy_action || s.policy_action || "—";
    const legDir = s.legacy_policy_direction || s.policy_direction || "";
    const v2Struct = s.v2_policy_structure || "—";
    const v2Act = s.v2_policy_action || (fallback ? "fallback" : "—");
    const v2Dir = s.v2_policy_direction || "";

    if (!s.policy_mode && !s.policy_action && !s.legacy_policy_structure) {
      $("policy-compare").innerHTML = '<p class="empty">No policy signals on latest tick</p>';
      $("policy-metrics").innerHTML = "";
      $("policy-chips").innerHTML = "";
      return;
    }

    $("policy-compare").innerHTML = `
      <div class="policy-side${disagree ? " disagree" : ""}">
        <div class="ps-label">Legacy (authoritative in shadow)</div>
        <div class="ps-struct">${esc(legStruct || "NT")}</div>
        <div class="ps-meta">${esc(legAct)}${legDir ? " · " + esc(legDir) : ""}</div>
      </div>
      <div class="policy-side${disagree ? " disagree" : ""}">
        <div class="ps-label">V2 prediction policy</div>
        <div class="ps-struct">${esc(v2Struct || "—")}</div>
        <div class="ps-meta">${esc(v2Act)}${v2Dir ? " · " + esc(v2Dir) : ""}</div>
      </div>`;

    $("policy-metrics").innerHTML = [
      metricCard("Source", esc(s.policy_source || "—"),
        s.policy_source === "fallback_legacy" ? "warn" : ""),
      metricCard("Confidence", fmt(s.policy_confidence != null ? s.policy_confidence
        : s.v2_policy_confidence, 2)),
      metricCard("Uncertainty", fmt(s.policy_uncertainty != null ? s.policy_uncertainty
        : s.v2_policy_uncertainty, 2),
        num(s.policy_uncertainty) > 0.5 ? "warn" : ""),
      metricCard("Size cap", fmt(s.policy_size_cap, 2)),
      metricCard("V2 conf.", fmt(s.v2_policy_confidence, 2)),
      metricCard("Version", esc(s.policy_version || s.v2_policy_version || "—")),
    ].join("");

    const chips = [];
    if (disagree) chips.push('<span class="chip disagree">disagreement</span>');
    if (fallback) chips.push('<span class="chip fallback">fallback legacy</span>');
    if (s.policy_hard_vetoes) {
      String(s.policy_hard_vetoes).split(",").filter(Boolean).forEach((v) =>
        chips.push(`<span class="chip veto">${esc(v.trim())}</span>`));
    }
    if (s.policy_rationale) {
      chips.push(`<span class="chip info" title="${esc(s.policy_rationale)}">${esc(String(s.policy_rationale).slice(0, 80))}</span>`);
    }
    if (!chips.length) chips.push('<span class="chip ok">aligned</span>');
    $("policy-chips").innerHTML = chips.join("");
  }

  /* ---------------- GEX variants (PR 9) ---------------- */
  function renderGexVariants(latest, report) {
    const s = tickSignals(latest) || {};
    const auth = s.gex_authoritative || "oi";
    $("gex-auth").textContent = auth ? `auth · ${auth}` : "—";
    const hasPanel = ["oi", "weekly", "volume", "hybrid"].some(
      (v) => num(s[`gex_${v}_net_gex`]) != null || num(s[`gex_${v}_gamma_flip`]) != null);
    const cmp = (report && report.gex_variant_comparison) || {};

    if (!hasPanel && !cmp.n) {
      $("gex-metrics").innerHTML = '<p class="empty">No GEX variant signals yet</p>';
      return;
    }

    const disagree = num(s.gex_disagree_sign);
    const cards = [
      metricCard("Sign disagree", disagree == null ? "—" : (disagree ? "yes" : "no"),
        disagree ? "warn" : "pos"),
      metricCard("Flip spread", fmt(s.gex_disagree_flip_spread, 2),
        num(s.gex_disagree_flip_spread) > 1 ? "warn" : ""),
      metricCard("Wall Δ call", fmt(s.gex_disagree_wall_call, 2)),
      metricCard("Wall Δ put", fmt(s.gex_disagree_wall_put, 2)),
      metricCard("Net GEX range", compact(s.gex_disagree_net_gex_range)),
      metricCard("Variants", s.gex_disagree_n_variants != null
        ? String(s.gex_disagree_n_variants) : "—"),
    ];
    ["oi", "weekly", "volume", "hybrid"].forEach((v) => {
      const flip = s[`gex_${v}_gamma_flip`];
      const net = s[`gex_${v}_net_gex`];
      if (num(flip) == null && num(net) == null) return;
      const q = s[`gex_${v}_quality_score`];
      cards.push(metricCard(
        `${v} flip`,
        `${fmt(flip, 2)}${num(q) != null ? " · q" + fmt(q, 2) : ""}`,
        v === auth ? "info" : ""));
    });
    if (cmp.disagree_sign_rate != null) {
      cards.push(metricCard("Hist. disagree rate", pct(cmp.disagree_sign_rate),
        num(cmp.disagree_sign_rate) > 0.2 ? "warn" : ""));
    }
    if (cmp.variants) {
      Object.entries(cmp.variants).forEach(([v, p]) => {
        if (p && p.corr_vs_pnl != null) {
          cards.push(metricCard(`${v} corr P&L`, fmt(p.corr_vs_pnl, 3),
            num(p.corr_vs_pnl) > 0 ? "pos" : "neg"));
        }
      });
    }
    $("gex-metrics").innerHTML = cards.join("");
  }

  /* ---------------- dealer dynamics ---------------- */
  function renderDynamics(latest) {
    const s = tickSignals(latest) || {};
    const keys = [
      ["expected_move_consumed", "Move consumed", (v) => pct(v)],
      ["flip_chase", "Flip chase", (v) => fmt(v, 3)],
      ["flip_velocity", "Flip velocity", (v) => fmt(v, 3)],
      ["gex_velocity_bn", "GEX velocity", (v) => fmt(v, 3)],
      ["wall_rupture", "Wall rupture", (v) => fmt(v, 2)],
      ["call_wall_velocity", "Call wall vel", (v) => fmt(v, 3)],
      ["put_wall_velocity", "Put wall vel", (v) => fmt(v, 3)],
      ["straddle_ramp", "Straddle ramp", (v) => fmt(v, 3)],
      ["chan_bb_squeeze", "BB squeeze", (v) => fmt(v, 1)],
      ["chan_donchian_breakout_up", "Donchian up", (v) => fmt(v, 1)],
      ["chan_donchian_breakout_down", "Donchian dn", (v) => fmt(v, 1)],
    ];
    const cards = [];
    keys.forEach(([k, label, fmtFn]) => {
      if (num(s[k]) == null && s[k] !== 0) return;
      const warn = (k === "wall_rupture" && num(s[k]) > 0)
        || (k === "expected_move_consumed" && num(s[k]) > 0.7);
      cards.push(metricCard(label, fmtFn(s[k]), warn ? "warn" : ""));
    });
    $("dynamics-metrics").innerHTML = cards.length
      ? cards.join("")
      : '<p class="empty">No dynamics signals on latest tick</p>';
  }

  /* ---------------- V2 forecast bundle ---------------- */
  function forecastFromSignals(tick, live) {
    const s = mergeV2Signals(tick, live);
    const liveFc = (live && live.forecast) || {};
    const p = { ...liveFc };
    for (const [k, v] of Object.entries(s)) {
      if (k.startsWith("v2_fc_") && k !== "v2_fc_mode" && k !== "v2_fc_model_version") {
        p[k.slice(6)] = v;
      }
    }
    if (!Object.keys(p).length) return null;
    return {
      mode: s.v2_fc_mode || liveFc.mode || "signals",
      model_group_version: s.v2_fc_model_version || liveFc.model_version || "",
      uncertainty: p.uncertainty,
      predictions: p,
    };
  }

  function renderForecast(predPayload, tick, live) {
    const row = predPayload && predPayload.prediction;
    let p = (row && row.predictions) || {};
    let sub = row && (row.mode || row.model_group_version);
    let unc = row && row.uncertainty;
    if (!row || !Object.keys(p).length) {
      const fallback = forecastFromSignals(tick, live);
      if (fallback) {
        p = fallback.predictions;
        sub = fallback.mode || fallback.model_group_version || "signals";
        unc = fallback.uncertainty;
      } else {
        $("forecast-sub").textContent = predPayload && predPayload.note
          ? "unavailable" : "—";
        $("forecast-metrics").innerHTML =
          '<p class="empty">No PredictionBundle for this snapshot</p>';
        return;
      }
    }
    $("forecast-sub").textContent = sub || "bundle";
    const cards = [
      metricCard("P(up) 30m", pct(p.p_up_30m)),
      metricCard("P(up) close", pct(p.p_up_close)),
      metricCard("E[r] 30m", fmt(p.expected_return_30m, 4)),
      metricCard("q10 / q50 / q90",
        `${fmt(p.return_q10_30m, 4)} / ${fmt(p.return_q50_30m, 4)} / ${fmt(p.return_q90_30m, 4)}`),
      metricCard("Range survive 30m", pct(p.p_range_survive_30m)),
      metricCard("Realized move 30m", fmt(p.expected_realized_move_30m, 4)),
      metricCard("Touch call wall", pct(p.p_touch_call_wall_30m)),
      metricCard("Touch put wall", pct(p.p_touch_put_wall_30m)),
      metricCard("Uncertainty", fmt(p.uncertainty != null ? p.uncertainty : unc, 2),
        num(p.uncertainty != null ? p.uncertainty : unc) > 0.5 ? "warn" : ""),
      metricCard("Data quality", pct(p.data_quality)),
      metricCard("Coverage", pct(p.feature_coverage)),
    ];
    $("forecast-metrics").innerHTML = cards.join("");
  }

  /* ---------------- signal correlations (V2 keys from report) ---------------- */
  const SIGCORR_PREFIXES = ["sig:policy_", "sig:v2_", "sig:phys_", "sig:gex_",
    "sig:expected_move", "sig:flip_", "sig:wall_", "sig:chan_"];
  function renderSigCorr(report) {
    const corr = (report && report.component_correlations) || {};
    const entries = Object.entries(corr)
      .filter(([k]) => SIGCORR_PREFIXES.some((p) => k.startsWith(p)))
      .map(([k, v]) => [k.replace(/^sig:/, ""), num(v && v.r != null ? v.r : v)])
      .filter(([, r]) => r != null)
      .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
      .slice(0, 14);
    if (!entries.length) {
      $("sigcorr-rows").innerHTML = '<p class="empty">No V2 signal correlations yet</p>';
      return;
    }
    $("sigcorr-rows").innerHTML = `<div class="val-corr">${entries.map(([k, r]) =>
      `<div class="val-corr-row"><span>${esc(k)}</span>
        <span class="${r >= 0 ? "pos" : "neg"}">${r >= 0 ? "+" : ""}${r.toFixed(3)}</span></div>`
    ).join("")}</div>`;
  }

  /* ---------------- regime confidence bars ---------------- */
  function renderRegime(live) {
    const conf = (live.why && live.why.regime_confidences) || {};
    const entries = Object.entries(conf).sort((a, b) => b[1] - a[1]);
    if (!entries.length) { $("regime-bars").innerHTML = '<p class="empty">No regime data</p>'; return; }
    const dom = (live.doing && live.doing.dominant_regime) || "";
    $("regime-bars").innerHTML = entries.map(([k, v]) => {
      const val = num(v) || 0;
      const isDom = k === dom;
      const color = val >= 70 ? "green" : val >= 55 ? "amber" : "red";
      return `<div class="trow">
        <span class="lbl">${esc(k)}${isDom ? " ●" : ""}</span>
        <span class="track"><span style="width:${Math.min(100, val)}%;background:var(--${color === "green" ? "green" : color === "amber" ? "amber" : "red"})"></span></span>
        <span class="num">${val.toFixed(0)}%</span>
      </div>`;
    }).join("");
  }

  /* ---------------- reason sentence ---------------- */
  function renderReason(live, latest) {
    const d = live.doing || {}, w = live.why || {}, t = latest || {};
    let html = "";
    if (!live.ts || (live.status && live.status !== "live")) {
      const lead = (live.status === "feed_not_ready" || live.status === "feed_error") ? "No market feed." : "Standing by.";
      $("reason").innerHTML = `<b>${lead}</b> ${esc(live.note || "No live tick yet — the pipeline is idle.")}`;
      return;
    }
    if (d.stand_down) {
      html = `<b>Standing down.</b> ${esc((w.stand_down_reason || "regime veto").replace(/_/g, " "))}.`;
      if (arr(w.dealer_vetoes).length) html += ` Dealer flags: ${esc(arr(w.dealer_vetoes).join(", "))}.`;
    } else if (d.decision === "TRADE" && d.gate_pass) {
      html = `<b>Trade armed.</b> ${esc(d.structure || "")} ${esc(d.direction || "")} via <b>${esc(d.permitted_engine || "engine")}</b> in a ${esc(d.dominant_regime || "")} regime. `;
      html += `Gate scored <b>${fmt(d.gate_score, 1)}</b>`;
      if (w.capture) html += `, targeting <b>${esc(w.capture)}</b>`;
      if (w.strike_rule) html += ` with strike rule ${esc(w.strike_rule)}`;
      html += ". ";
      if (num(t.ev) != null) html += `Expected value <b>${money(t.ev, 0)}</b> against ${money(t.max_loss, 0)} max risk (${fmt(t.ev_per_risk, 2)}× EV/risk).`;
    } else {
      const reason = w.no_trade_reason || (arr(w.gate_failed).length ? "gate failed: " + arr(w.gate_failed).join(", ")
        : arr(w.selector_vetoes).length ? "selector vetoed: " + arr(w.selector_vetoes).join(", ")
        : "conditions not met");
      html = `<b>No trade.</b> ${esc(String(reason).replace(/_/g, " "))}. `;
      html += `Regime ${esc(d.dominant_regime || "?")}, gate ${fmt(d.gate_score, 1)}.`;
      if (w.intent_note) html += ` ${esc(w.intent_note)}`;
    }
    $("reason").innerHTML = html || "Awaiting decision engine…";
  }

  /* ---------------- why panel ---------------- */
  function renderWhy(live) {
    const w = live.why || {}, d = live.doing || {};
    const cell = arr(w.matrix_cell).filter(Boolean).join(" × ") || "—";
    const cards = [
      metricCard("Matrix cell", esc(cell)),
      metricCard("Engine", esc(d.permitted_engine || "—")),
      metricCard("Information gain", fmt(w.global_information_gain, 0)),
      metricCard("Capture", esc(w.capture || "—")),
      metricCard("Strike rule", esc(w.strike_rule || "—")),
    ];
    $("why-metrics").innerHTML = cards.join("");

    const chips = [];
    const push = (items, cls, prefix) => arr(items).forEach((x) =>
      chips.push(`<span class="chip ${cls}">${prefix}${esc(String(x).replace(/_/g, " "))}</span>`));
    push(w.dealer_vetoes, "veto", "dealer: ");
    push(w.gate_failed, "veto", "gate: ");
    push(w.selector_vetoes, "veto", "");
    push(w.risk_vetoes, "veto", "");
    push(w.intent_vetoes, "veto", "");
    if (!chips.length) chips.push('<span class="chip ok">no active vetoes</span>');
    $("why-chips").innerHTML = chips.join("");
  }

  /* ---------------- volatility term structure ---------------- */
  function renderVol(live) {
    const inp = live.inputs || {};
    const rows = [];
    const maxVix = Math.max(20, num(inp.vix9d) || 0, num(inp.vix) || 0, num(inp.vix3m) || 0) * 1.15;
    const volRow = (lbl, v, color) => {
      const n = num(v);
      const w = n != null ? Math.min(100, (n / maxVix) * 100) : 0;
      return `<div class="trow"><span class="lbl">${lbl}</span>
        <span class="track"><span style="width:${w}%;background:var(--${color})"></span></span>
        <span class="num">${n != null ? n.toFixed(2) : "—"}</span></div>`;
    };
    rows.push(volRow("VIX9D", inp.vix9d, "amber"));
    rows.push(volRow("VIX", inp.vix, "blue"));
    rows.push(volRow("VIX3M", inp.vix3m, "violet"));

    // term structure state: backwardation (9D>VIX) = stress
    const v9 = num(inp.vix9d), v = num(inp.vix);
    if (v9 != null && v != null) {
      const back = v9 > v;
      rows.push(`<div class="trow"><span class="lbl">Structure</span>
        <span class="track" style="background:transparent"></span>
        <span class="num" style="width:auto;color:var(--${back ? "red" : "green"})">${back ? "backwardation" : "contango"}</span></div>`);
    }
    // VVIX vs baseline
    const vv = num(inp.vvix), vvb = num(inp.vvix_baseline);
    if (vv != null) {
      const hot = vvb != null && vv > vvb;
      rows.push(`<div class="trow"><span class="lbl">VVIX</span>
        <span class="track"><span style="width:${Math.min(100, (vv / 140) * 100)}%;background:var(--${hot ? "red" : "cyan"})"></span></span>
        <span class="num">${vv.toFixed(0)}</span></div>`);
    }
    $("vol-rows").innerHTML = rows.join("");
  }

  /* ---------------- technicals + dealer positioning ---------------- */
  function renderTech(live) {
    const inp = live.inputs || {};
    const spot = num(inp.spot);
    const zg = num(inp.zero_gamma_dist_pct);
    const gexPos = num(inp.net_gex) >= 0;
    const rsi = num(inp.rsi);
    const rsiCls = rsi == null ? "" : rsi > 70 ? "neg" : rsi < 30 ? "pos" : "";
    const adx = num(inp.adx);
    const cards = [
      metricCard("Net GEX", compact(inp.net_gex), gexPos ? "info" : "warn"),
      metricCard("GEX rank", inp.gex_pct_rank != null ? (num(inp.gex_pct_rank) * (num(inp.gex_pct_rank) <= 1 ? 100 : 1)).toFixed(0) + "%ile" : "—"),
      metricCard("Gamma flip", fmt(inp.gamma_flip, 2)),
      metricCard("Zero-γ dist", zg != null ? (zg * 100).toFixed(2) + "%" : "—", zg != null && Math.abs(zg) < 0.002 ? "warn" : ""),
      metricCard("Call wall", fmt(inp.call_wall, 0), "neg"),
      metricCard("Put wall", fmt(inp.put_wall, 0), "pos"),
      metricCard("VWAP", fmt(inp.vwap, 2), "info"),
      metricCard("Exp. range", fmt(inp.expected_range, 2)),
      metricCard("ADX", fmt(adx, 1), adx != null && adx > 25 ? "warn" : ""),
      metricCard("RSI", fmt(rsi, 1), rsiCls),
      metricCard("BB width", fmt(inp.bb_width, 3)),
      metricCard("CVD slope", fmt(inp.cvd_slope, 3), num(inp.cvd_slope) >= 0 ? "pos" : "neg"),
    ];
    $("tech-metrics").innerHTML = cards.join("");
  }

  /* ---------------- RAS (regime alignment) helpers ---------------- */
  // Color bands per the activation spec: green > -20, amber -50..-20, red <= -50.
  function rasCls(v) {
    const n = num(v);
    if (n == null) return "";
    return n > -20 ? "ok" : n > -50 ? "warn" : "bad";
  }

  function rasEma(ctx) {
    if (!ctx) return null;
    const e = num(ctx.ras_ema_score);
    return e != null ? e : num(ctx.ras_score);
  }

  function rasCompRow(c) {
    const r = num(c.raw);
    const cl = r == null ? "" : r < 0 ? "neg" : r > 0 ? "pos" : "";
    return `<div class="ras-comp"><span class="mono ${cl}">${sign(r, 2)}</span>
      <span class="ras-comp-name">${esc(String(c.name || "").replace(/_/g, " "))}</span>
      <small>${esc(c.note || "")}</small></div>`;
  }

  // Full RAS health block for an open position card: score + action badge,
  // top negative components inline, complete breakdown behind <details>.
  function rasBlock(ctx) {
    const ema = rasEma(ctx);
    if (ema == null) return "";
    const cls = rasCls(ema);
    const action = String(ctx.ras_action || "ok");
    const comps = arr(ctx.ras_components);
    const negs = comps.filter((c) => num(c.raw) != null && c.raw < -0.01)
      .sort((a, b) => a.raw - b.raw).slice(0, 3);
    return `<div class="ras-block ${cls}">
      <div class="ras-head">
        <span class="ras-title">Regime alignment</span>
        <span class="ras-score mono ${cls}">${sign(ema, 1)}</span>
        <span class="ras-badge ${action}">${esc(action)}</span>
      </div>
      ${negs.map(rasCompRow).join("")}
      ${comps.length ? `<details class="ras-details"><summary>all ${comps.length} components</summary>${comps.map(rasCompRow).join("")}</details>` : ""}
    </div>`;
  }

  // Compact one-line RAS chip for trade-journal rows.
  function rasInline(ctx) {
    const ema = rasEma(ctx);
    if (ema == null) return "";
    const action = String(ctx.ras_action || "ok");
    return `<span class="ras-badge ${action}">RAS ${sign(ema, 1)} · ${esc(action)}</span>`;
  }

  // Closed-trade RAS summary: final score at exit, worst score seen, last action.
  function rasExitLine(ctx) {
    if (!ctx) return "";
    const atExit = num(ctx.ras_at_exit), worst = num(ctx.ras_worst);
    if (atExit == null && worst == null) return "";
    const bits = [];
    if (atExit != null) bits.push(`at exit ${sign(atExit, 1)}`);
    if (worst != null) bits.push(`worst ${sign(worst, 1)}`);
    if (ctx.ras_last_action && ctx.ras_last_action !== "ok") bits.push(esc(ctx.ras_last_action));
    return `<div class="tj-sub mono ras-exit-line ${rasCls(worst != null ? worst : atExit)}">RAS ${bits.join(" · ")}</div>`;
  }

  /* ---------------- open position(s) ---------------- */
  function renderOpenPositions(livePaper) {
    const panel = $("open-positions-panel");
    const open = (livePaper && livePaper.open) || [];
    if (!open.length) {
      panel.classList.add("hidden");
      return;
    }
    panel.classList.remove("hidden");
    $("open-positions-count").textContent = open.length > 1 ? `${open.length} open` : "1 open";
    $("open-positions-list").innerHTML = open.map((p) => {
      const pnl = num(p.unrealized_pnl_dollars);
      const pnlCls = pnl > 0 ? "pos" : pnl < 0 ? "neg" : "";
      const pctMax = p.pct_of_max_profit != null ? pct(p.pct_of_max_profit) : "—";
      return `<div class="op-card">
        <div class="op-head">
          <div>
            <div class="op-strikes">${esc(p.strikes)}</div>
            <div class="op-family">${esc(p.family)} · x${esc(p.contracts)}</div>
          </div>
          <div class="op-pnl ${pnlCls}">${pnl >= 0 ? "+" : ""}${money(pnl, 2)}</div>
        </div>
        <div class="op-metrics">
          <div><span class="k">Entry credit</span><span class="v">${fmt(p.entry_credit, 2)}</span></div>
          <div><span class="k">Held</span><span class="v">${fmt(p.hold_min, 0)}m</span></div>
          <div><span class="k">% of max profit</span><span class="v ${pnlCls}">${pctMax}</span></div>
          <div><span class="k">Max profit</span><span class="v pos">${fmt(p.max_profit_ps, 2)}</span></div>
          <div><span class="k">Max loss</span><span class="v neg">${fmt(p.max_loss_ps, 2)}</span></div>
          <div><span class="k">Opened</span><span class="v">${etTime(p.opened_at)}</span></div>
        </div>
        ${rasBlock(p.entry_ctx || {})}
      </div>`;
    }).join("");
  }

  /* ---------------- paper account ---------------- */
  function renderPaper(paper) {
    if (!paper || paper.trades == null) { $("paper-metrics").innerHTML = '<div class="metric"><span class="k">status</span><span class="v sm">no data</span></div>'; return; }
    const pnlCls = num(paper.total_pnl) > 0 ? "pos" : num(paper.total_pnl) < 0 ? "neg" : "";
    $("paper-metrics").innerHTML = [
      metricCard("Equity", money(paper.equity, 0), "info"),
      metricCard("Total P&L", money(paper.total_pnl, 0), pnlCls),
      metricCard("Win rate", pct(paper.win_rate)),
      metricCard("Profit factor", paper.profit_factor != null ? fmt(paper.profit_factor, 2) : "—"),
      metricCard("Closed trades", paper.trades),
      metricCard("Best exit", topReason(paper.by_exit_reason)),
    ].join("");
  }
  function topReason(m) {
    if (!m) return "—";
    const e = Object.entries(m).sort((a, b) => b[1] - a[1])[0];
    return e ? `${e[0]} (${e[1]})` : "—";
  }

  /* ---------------- system edge ---------------- */
  function renderEdge(report) {
    const eff = (report && report.gate_effectiveness) || {};
    const taken = eff.trades_taken || {}, blocked = eff.blocked_by_gate || {};
    $("edge-metrics").innerHTML = [
      metricCard("Gate verdict", esc(eff.verdict || "insufficient data"),
        eff.verdict && /work|good|effective/i.test(eff.verdict) ? "pos" : ""),
      metricCard("Trades taken", taken.n != null ? `${taken.n} · μ ${money(taken.mean, 0)}` : "—",
        num(taken.mean) > 0 ? "pos" : ""),
      metricCard("Blocked by gate", blocked.n != null ? `${blocked.n} · μ ${money(blocked.mean, 0)}` : "—",
        num(blocked.mean) < 0 ? "pos" : ""),
    ].join("");
  }

  /* ---------------- decision funnel ---------------- */
  const CREDIT_KEYS = new Set([
    "put_credit", "call_credit", "iron_condor", "iron_fly", "broken_wing",
    "PCS", "CCS", "IC", "IF",
  ]);
  function fnHistCol(title, obj, { creditTint = false, warnKeys = null, fmtRow = null } = {}) {
    const entries = Object.entries(obj || {}).sort((a, b) => b[1] - a[1]).slice(0, 10);
    const rows = entries.length
      ? (() => {
          const max = entries[0][1] || 1;
          return entries.map(([k, v]) => {
            const cls = creditTint && CREDIT_KEYS.has(k) ? "credit"
              : (warnKeys && warnKeys.has(k) ? "warn" : "");
            const label = fmtRow ? fmtRow(k) : k;
            return `<div class="fn-row ${cls}">
              <span class="lbl" title="${esc(k)}">${esc(label)}</span>
              <span class="track"><span style="width:${Math.max(4, (v / max) * 100)}%"></span></span>
              <span class="num">${v}</span>
            </div>`;
          }).join("");
        })()
      : '<div class="fn-empty">none</div>';
    return `<div class="fn-col"><div class="fn-title">${esc(title)}</div>${rows}</div>`;
  }
  function renderFunnel(report, ticks) {
    const f = report && report.decision_funnel;
    if (!f || !f.n) {
      $("funnel-sub").textContent = "—";
      $("funnel-metrics").innerHTML = "";
      $("funnel-hists").innerHTML = '<p class="empty">No funnel data yet</p>';
      return;
    }
    $("funnel-sub").textContent = `${f.n} ticks · ${f.sessions} session${f.sessions === 1 ? "" : "s"}`;

    const cm = f.class_mix || {};
    const credit = cm.credit || {}, debit = cm.debit || {}, stand = cm.stand_down || {};
    const flips = (f.premium_flips || {}).n;
    const gex = f.gex_rank || {};
    const warmFrac = gex.frac_at_warmup_neutral;

    // Session-local policy disagreement roll-up from journaled ticks (PR 10).
    let polN = 0, polDisagree = 0, polFallback = 0;
    const polPairs = {};
    (ticks || []).forEach((t) => {
      const s = tickSignals(t);
      if (!s || !s.policy_mode) return;
      polN += 1;
      if (num(s.policy_disagreement) === 1) {
        polDisagree += 1;
        const key = `${s.legacy_policy_structure || "?"}→${s.v2_policy_structure || "?"}`;
        polPairs[key] = (polPairs[key] || 0) + 1;
      }
      if (num(s.policy_fallback_used) === 1) polFallback += 1;
    });

    $("funnel-metrics").innerHTML = [
      metricCard("Credit routed · traded", `${credit.n || 0} · ${credit.traded || 0}`,
        (credit.traded || 0) > 0 ? "pos" : "warn"),
      metricCard("Debit routed · traded", `${debit.n || 0} · ${debit.traded || 0}`),
      metricCard("Stand-down ticks", `${stand.n || 0}`),
      metricCard("Credit→debit flips", flips != null ? String(flips) : "—",
        num(flips) > 0 ? "warn" : ""),
      metricCard("GEX rank in warm-up", warmFrac != null ? pct(warmFrac) : "—",
        num(warmFrac) > 0.2 ? "warn" : ""),
      metricCard("GEX rank < gate floor", gex.frac_below_gate_floor != null
        ? `${pct(gex.frac_below_gate_floor)} of ticks` : "—"),
      metricCard("Policy disagree (session)",
        polN ? `${polDisagree}/${polN}` : "—",
        polDisagree > 0 ? "warn" : ""),
      metricCard("Policy fallback (session)",
        polN ? `${polFallback}/${polN}` : "—",
        polFallback > 0 ? "warn" : ""),
    ].join("");

    const mixCounts = Object.fromEntries(
      Object.entries(f.structure_mix || {}).map(([k, s]) => [k, (s && s.n) || 0]));
    $("funnel-hists").innerHTML = [
      fnHistCol("Routed by regime (Track B)", f.routed_structures, { creditTint: true }),
      fnHistCol("Final structure · traded/total", mixCounts, {
        creditTint: true,
        fmtRow: (k) => {
          const s = (f.structure_mix || {})[k] || {};
          return `${k} ${s.traded || 0}/${s.n || 0}`;
        },
      }),
      fnHistCol("Which layer said no", f.no_trade_reasons),
      fnHistCol("Gate failures", f.gate_failures,
        { warnKeys: new Set(["GEX_WEAK", "TRENDING"]) }),
      fnHistCol("Dealer vetoes", f.regime_vetoes),
      fnHistCol("Selector vetoes", f.selector_vetoes),
      fnHistCol("Policy disagreements", polPairs, { creditTint: false }),
    ].join("");
  }

  /* ---------------- live-readiness checklist ---------------- */
  function fmtNum(x) {
    return Number.isInteger(x) ? String(x) : fmt(x, 3);
  }
  function fmtActual(v, target) {
    if (v == null) return "—";
    if (typeof v === "number") {
      // A fractional value whose target is stated in "%" (e.g. max_drawdown_pct)
      // reads better as a percentage than a bare decimal.
      if (target && /%/.test(target)) return (v * 100).toFixed(1) + "%";
      return fmtNum(v);
    }
    if (typeof v === "object") {
      return Object.entries(v)
        .map(([k, x]) => `${k.replace(/_/g, " ")}: ${typeof x === "number" ? fmtNum(x) : esc(String(x))}`)
        .join(" · ");
    }
    return esc(String(v));
  }

  function renderReadiness(data) {
    const badge = $("readiness-badge");
    if (!data || !data.checks) {
      badge.textContent = "—";
      badge.className = "";
      $("readiness-checks").innerHTML = '<p class="empty">No readiness data yet</p>';
      return;
    }
    badge.textContent = data.ready ? "READY" : "NOT READY";
    badge.className = "readiness-badge " + (data.ready ? "ready" : "not-ready");

    $("readiness-checks").innerHTML = `<div class="rc-grid">${data.checks.map((c) => `
      <div class="rc-row">
        <span class="rc-icon ${c.ok ? "ok" : "no"}">${c.ok ? "✓" : "✕"}</span>
        <div class="rc-body">
          <div class="rc-label">${esc(c.label)}</div>
          <div class="rc-target">target: ${esc(c.target)}</div>
          <div class="rc-actual">${fmtActual(c.actual, c.target)}</div>
        </div>
      </div>`).join("")}</div>`;
  }

  /* ---------------- session log ---------------- */
  function renderTimeline(data) {
    $("log-date").textContent = data.session_date || "";
    const ticks = (data.ticks || []).slice().reverse();
    if (!ticks.length) { $("timeline").innerHTML = '<p class="empty">No evaluations yet today</p>'; return; }
    $("timeline").innerHTML = ticks.map((t) => {
      const trade = t.decision === "TRADE";
      const fam = t.selected_family || "—";
      const s = tickSignals(t) || {};
      const extras = [];
      if (s.policy_structure || s.legacy_policy_structure) {
        const leg = s.legacy_policy_structure || s.policy_structure;
        const v2 = s.v2_policy_structure;
        extras.push(v2 && v2 !== leg
          ? `pol ${leg}≠${v2}`
          : `pol ${leg || "—"}`);
      }
      if (s.phys_density_mode) extras.push(s.phys_density_mode);
      if (num(s.gex_disagree_sign) === 1) extras.push("gex≠");
      if (num(s.policy_fallback_used) === 1) extras.push("fallback");
      const extraHtml = extras.length
        ? ` <small class="tl-v2">${esc(extras.join(" · "))}</small>` : "";
      const detail = trade
        ? `${esc(fam)} <small>gate ${fmt(t.gate_score, 0)} · EV ${money(t.ev, 0)}</small>${extraHtml}`
        : `<small>${esc((t.no_trade_reason || t.gex_regime || "no trade").replace(/_/g, " "))}</small>${extraHtml}`;
      const tint = t.regime_direction === "call" ? " tl-call"
                 : t.regime_direction === "put" ? " tl-put" : "";
      return `<div class="tl-item${trade ? " is-trade" : ""}${tint}">
        <span class="t">${etTime(t.ts)}</span>
        <span class="m">${detail} <small>· ${fmt(t.spot, 2)}</small></span>
        <span class="d ${trade ? "trade" : "no"}">${trade ? "TRADE" : "—"}</span>
      </div>`;
    }).join("");
  }

  /* ============================================================
     REGIME FIELD — continuous bias/gamma readings per journal tick
     ============================================================ */
  function tickSignals(t) {
    let s = t && t.signals_json;
    if (typeof s === "string") {           // older backends serve the raw JSON string
      try { s = JSON.parse(s); } catch (_) { s = null; }
    }
    return s && typeof s === "object" ? s : null;
  }

  function mergeV2Signals(tick, live) {
    const fromLive = (live && live.v2_signals) || {};
    const fromTick = tickSignals(tick) || {};
    return { ...fromLive, ...fromTick };
  }

  function lastTickWithV2(ticks) {
    for (let i = ticks.length - 1; i >= 0; i--) {
      const s = tickSignals(ticks[i]);
      if (!s) continue;
      if (s.phys_v2_mean != null || s.v2_fc_p_up_30m != null
          || s.v2_top_candidate_id != null || s.v2_rank_disagreement != null
          || s.phys_density_mode != null) {
        return ticks[i];
      }
    }
    return ticks.length ? ticks[ticks.length - 1] : null;
  }

  // Direction bias in [-1, +1] (+1 bull). Prefers the continuous matrix bias
  // value journaled in signals_json; falls back to the resolved direction word.
  function tickBias(t) {
    const s = tickSignals(t);
    const bv = s ? num(s.regime_bias_value) : null;
    if (bv != null) return Math.max(-1, Math.min(1, (bv - 50) / 50));
    if (t.regime_direction === "call") return 0.6;
    if (t.regime_direction === "put") return -0.6;
    return 0;
  }

  // Dominant-regime confidence in [0, 1]; neutral default when not journaled.
  function tickConf(t) {
    const s = tickSignals(t);
    const c = s ? num(s.regime_dominant_conf) : null;
    return c != null ? Math.max(0, Math.min(1, c / 100)) : 0.6;
  }

  // Gamma favorability in [-1, +1] (+1 = long gamma, well above the flip).
  function gammaFavor(netGex, zgPct) {
    const gexSign = num(netGex) != null ? (netGex >= 0 ? 1 : -1) : 0;
    const prox = num(zgPct) != null ? Math.tanh(zgPct / 0.004) : 0;
    return Math.max(-1, Math.min(1, 0.55 * prox + 0.45 * gexSign));
  }

  function tickGamma(t) { return gammaFavor(t.net_gex, t.zero_gamma_dist_pct); }

  // Smooth, non-discretized background shading behind the price line: per-tick
  // bias values are EMA-smoothed, then each segment is filled with a horizontal
  // gradient between neighboring colors so regime transitions blend like a
  // gauge field instead of snapping in vertical blocks.
  function drawRegimeZones(ctx, ticks, X, padT, plotH) {
    const rows = ticks
      .map((t) => ({ ts: new Date(t.ts).getTime(), b: tickBias(t), c: tickConf(t) }))
      .filter((r) => isFinite(r.ts));
    if (rows.length < 2) return;

    // EMA over ticks (~1/min): reacts inside a few minutes, ignores one-tick noise
    const alpha = 0.22;
    let ema = rows[0].b * rows[0].c;
    const smooth = rows.map((r, i) => {
      const v = r.b * (0.35 + 0.65 * r.c);   // confidence scales intensity
      ema = i === 0 ? v : alpha * v + (1 - alpha) * ema;
      return { ts: r.ts, v: ema };
    });

    const color = (v) => {
      const a = Math.min(0.16, Math.abs(v) * 0.22);
      if (a < 0.008) return "rgba(0,0,0,0)";
      return v > 0 ? `rgba(46,199,133,${a.toFixed(3)})` : `rgba(255,84,112,${a.toFixed(3)})`;
    };

    for (let i = 1; i < smooth.length; i++) {
      const x0 = X(smooth[i - 1].ts), x1 = X(smooth[i].ts);
      if (x1 <= x0) continue;
      const g = ctx.createLinearGradient(x0, 0, x1, 0);
      g.addColorStop(0, color(smooth[i - 1].v));
      g.addColorStop(1, color(smooth[i].v));
      ctx.fillStyle = g;
      ctx.fillRect(x0, padT, x1 - x0, plotH);
    }
  }

  /* ============================================================
     FOUR-WAY MATRIX — 2x2 quadrant: direction bias x gamma favorability.
     Dot = current state, fading trail = recent ticks, background
     intensity keyed to VIX.
     ============================================================ */
  function drawQuadrant() {
    const cv = $("quad");
    if (!cv || !lastChartData) return;
    const { ticks, live } = lastChartData;
    const dpr = window.devicePixelRatio || 1;
    const W = cv.clientWidth, H = cv.clientHeight;
    if (!W || !H) return;
    cv.width = W * dpr; cv.height = H * dpr;
    const ctx = cv.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);

    const pad = 8;
    const cx = W / 2, cy = H / 2;
    const px = (b) => cx + b * (W / 2 - pad);          // bias -1..+1 → x
    const py = (g) => cy - g * (H / 2 - pad);          // gamma -1..+1 → y (up = favorable)

    // current state from the live payload, falling back to the newest tick
    const doing = live && live.doing ? live.doing : {};
    const inputs = live && live.inputs ? live.inputs : {};
    const latest = ticks && ticks.length ? ticks[ticks.length - 1] : null;
    let bias = num(doing.bias_value) != null
      ? Math.max(-1, Math.min(1, (doing.bias_value - 50) / 50))
      : (latest ? tickBias(latest) : 0);
    let gamma = (num(inputs.net_gex) != null || num(inputs.zero_gamma_dist_pct) != null)
      ? gammaFavor(inputs.net_gex, inputs.zero_gamma_dist_pct)
      : (latest ? tickGamma(latest) : 0);

    // VIX drives background intensity: calm ≈ faint, stressed ≈ saturated
    const vix = num(inputs.vix) != null ? inputs.vix : (latest ? num(latest.vix) : null);
    const heat = vix == null ? 0.10 : Math.max(0.06, Math.min(0.26, (vix - 11) / 60));

    // quadrant tints: bull side green, bear side red; short-gamma half darker
    const tints = [
      { x0: cx, y0: pad, c: `rgba(46,199,133,${heat})` },              // bull + long γ
      { x0: pad, y0: pad, c: `rgba(255,84,112,${(heat * 0.75).toFixed(3)})` },   // bear + long γ
      { x0: cx, y0: cy, c: `rgba(46,199,133,${(heat * 0.55).toFixed(3)})` },     // bull + short γ
      { x0: pad, y0: cy, c: `rgba(255,84,112,${heat})` },              // bear + short γ
    ];
    tints.forEach((q) => {
      const g = ctx.createRadialGradient(
        q.x0 === pad ? pad : W - pad, q.y0 === pad ? pad : H - pad, 8,
        cx, cy, Math.max(W, H) / 1.4);
      g.addColorStop(0, q.c);
      g.addColorStop(1, "rgba(0,0,0,0)");
      ctx.fillStyle = g;
      ctx.fillRect(q.x0, q.y0, cx - pad, cy - pad);
    });

    // axes
    ctx.strokeStyle = "rgba(133,149,176,0.35)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(pad, cy); ctx.lineTo(W - pad, cy);
    ctx.moveTo(cx, pad); ctx.lineTo(cx, H - pad);
    ctx.stroke();

    // corner labels
    ctx.font = "600 9px 'SF Mono', ui-monospace, monospace";
    ctx.fillStyle = "#8595b0";
    ctx.textAlign = "left";
    ctx.fillText("BEAR · LONG γ", pad + 4, pad + 12);
    ctx.fillText("BEAR · SHORT γ", pad + 4, H - pad - 5);
    ctx.textAlign = "right";
    ctx.fillText("BULL · LONG γ", W - pad - 4, pad + 12);
    ctx.fillText("BULL · SHORT γ", W - pad - 4, H - pad - 5);

    // trail: last 30 ticks, oldest faintest
    const trail = (ticks || []).slice(-30);
    trail.forEach((t, i) => {
      const a = 0.06 + 0.5 * (i / Math.max(1, trail.length - 1));
      ctx.fillStyle = `rgba(74,168,255,${a.toFixed(3)})`;
      ctx.beginPath();
      ctx.arc(px(tickBias(t)), py(tickGamma(t)), 2.2, 0, Math.PI * 2);
      ctx.fill();
    });

    // fast-composite marker: hollow ring at the raw fast (1m/5m/15m) bias.
    // When it sits far to one side of the solid dot, the short timeframes are
    // leading the blend — the V-turn signature.
    const fastRaw = num(doing.bias_fast);
    const fastBias = fastRaw != null ? Math.max(-1, Math.min(1, (fastRaw - 50) / 50)) : null;
    if (fastBias != null) {
      ctx.strokeStyle = "#ffb648";
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.arc(px(fastBias), py(gamma), 4.5, 0, Math.PI * 2);
      ctx.stroke();
    }

    // current-state dot (blended bias — what entries actually use)
    const dotX = px(bias), dotY = py(gamma);
    ctx.fillStyle = "#4aa8ff";
    ctx.strokeStyle = "#e6edf7";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.arc(dotX, dotY, 5.5, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();

    // header label: plain-language read of the current quadrant
    const lbl = $("quad-label");
    if (lbl) {
      const side = bias > 0.12 ? "bull" : bias < -0.12 ? "bear" : "neutral";
      const gq = gamma > 0.12 ? "favorable γ" : gamma < -0.12 ? "hostile γ" : "flat γ";
      const vtxt = vix != null ? ` · VIX ${vix.toFixed(1)}` : "";
      const ftxt = fastRaw != null ? ` · fast ${fastRaw.toFixed(0)}` : "";
      lbl.textContent = `${side} · ${gq}${vtxt}${ftxt}`;
    }
  }

  /* ============================================================
     CHART — spot + levels + forward projection + trade markers
     ============================================================ */
  function drawChart() {
    const cv = $("chart");
    if (!cv || !lastChartData) return;
    const { ticks, live, market } = lastChartData;
    const inp = live.inputs || {};

    const dpr = window.devicePixelRatio || 1;
    const W = cv.clientWidth, H = cv.clientHeight;
    if (!W || !H) return;
    cv.width = W * dpr; cv.height = H * dpr;
    const ctx = cv.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);

    const padL = 8, padR = 62, padT = 12, padB = 22;
    const plotW = W - padL - padR, plotH = H - padT - padB;

    const pts = ticks.map((t) => ({ ts: new Date(t.ts).getTime(), spot: num(t.spot), decision: t.decision }))
      .filter((p) => p.spot != null && isFinite(p.ts));

    if (pts.length < 1) {
      ctx.fillStyle = "#5b6a86";
      ctx.font = "13px ui-monospace, monospace";
      ctx.textAlign = "center";
      ctx.fillText("Waiting for intraday ticks…", W / 2, H / 2);
      return;
    }

    const spot = num(inp.spot) != null ? num(inp.spot) : pts[pts.length - 1].spot;
    const vwap = num(inp.vwap);
    const callWall = num(inp.call_wall);
    const putWall = num(inp.put_wall);
    const gammaFlip = num(inp.gamma_flip);

    // projection band half-width
    let band = num(inp.expected_range);
    if (!band || band <= 0) {
      const be = num(inp.straddle_breakeven);
      if (be) band = Math.abs(be - spot);
    }
    if (!band || band <= 0) band = spot * 0.004;
    band = Math.min(band, spot * 0.05);

    // time domain
    const t0 = pts[0].ts;
    const tLast = pts[pts.length - 1].ts;
    let tEnd = tLast;
    if (market && market.is_open && market.next_close) {
      const nc = new Date(market.next_close).getTime();
      if (nc > tLast) tEnd = nc;
    }
    if (tEnd <= tLast) tEnd = tLast + Math.max((tLast - t0) * 0.25, 20 * 60 * 1000);
    const projFrac = 0.72; // where "now" sits horizontally (at zoom 1)
    // base map: historical [t0..tLast] -> [padL .. padL+plotW*projFrac],
    // projection -> remainder; then the zoom/pan screen-space transform.
    const xNowBase = padL + plotW * projFrac;
    const Xb = (ts) => {
      if (ts <= tLast) {
        const f = tLast > t0 ? (ts - t0) / (tLast - t0) : 1;
        return padL + f * (xNowBase - padL);
      }
      const f = tEnd > tLast ? (ts - tLast) / (tEnd - tLast) : 0;
      return xNowBase + f * (padL + plotW - xNowBase);
    };
    clampChartView();
    const zv = chartView.z, panv = chartView.pan;
    const X = (ts) => padL + (Xb(ts) - padL) * zv - panv;
    const xNow = padL + (xNowBase - padL) * zv - panv;
    // inverse: screen x -> timestamp (for axis labels under zoom)
    const Tof = (x) => {
      const xb = (x - padL + panv) / zv + padL;
      if (xb <= xNowBase) {
        return t0 + ((xb - padL) / Math.max(xNowBase - padL, 1e-9)) * (tLast - t0);
      }
      return tLast + ((xb - xNowBase) / Math.max(padL + plotW - xNowBase, 1e-9)) * (tEnd - tLast);
    };

    // price domain
    const lows = [spot - band], highs = [spot + band];
    pts.forEach((p) => { lows.push(p.spot); highs.push(p.spot); });
    [vwap, callWall, putWall, gammaFlip].forEach((v) => { if (v != null) { lows.push(v); highs.push(v); } });
    let lo = Math.min(...lows), hi = Math.max(...highs);
    const pad = (hi - lo) * 0.08 || 1;
    lo -= pad; hi += pad;
    const Y = (p) => padT + (1 - (p - lo) / (hi - lo)) * plotH;

    // --- grid + price axis ---
    ctx.font = "10px ui-monospace, monospace";
    ctx.textAlign = "left";
    const rows = 5;
    for (let i = 0; i <= rows; i++) {
      const p = lo + (i / rows) * (hi - lo);
      const y = Y(p);
      ctx.strokeStyle = "rgba(255,255,255,0.045)";
      ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(padL + plotW, y); ctx.stroke();
      ctx.fillStyle = "#5b6a86";
      ctx.fillText(p.toFixed(1), padL + plotW + 6, y + 3);
    }

    // --- time axis labels (fixed screen slots, times from the inverse map) ---
    ctx.textAlign = "center";
    ctx.fillStyle = "#5b6a86";
    for (let i = 0; i <= 4; i++) {
      const x = padL + (i / 4) * plotW;
      const ts = Math.max(t0, Math.min(tEnd, Tof(x)));
      ctx.fillText(new Date(ts).toLocaleTimeString("en-US", { timeZone: "America/New_York", hour: "numeric", minute: "2-digit" }), x, H - 6);
    }
    // explicit zoom level indicator (always visible)
    ctx.textAlign = "right";
    ctx.fillStyle = "#8595b0";
    ctx.fillText(zv > 1.001 ? `${zv.toFixed(1)}×` : "1× full session", padL + plotW, padT + 9);

    // clip everything time-positioned to the plot area while zoomed/panned
    ctx.save();
    ctx.beginPath();
    ctx.rect(padL, padT, plotW, plotH);
    ctx.clip();

    // --- regime gauge field (drawn first: everything else sits on top) ---
    drawRegimeZones(ctx, ticks, X, padT, plotH);

    // --- "now" divider ---
    ctx.strokeStyle = "rgba(157,123,255,0.35)";
    ctx.setLineDash([3, 4]);
    ctx.beginPath(); ctx.moveTo(xNow, padT); ctx.lineTo(xNow, padT + plotH); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = "#9d7bff";
    ctx.textAlign = "center";
    ctx.fillText("now", xNow, padT + 9);

    // --- horizontal levels ---
    const level = (v, color, label, dash) => {
      if (v == null) return;
      const y = Y(v);
      ctx.strokeStyle = color;
      ctx.lineWidth = 1.25;
      ctx.setLineDash(dash ? [5, 4] : []);
      ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(padL + plotW, y); ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = color;
      ctx.textAlign = "left";
      ctx.font = "9px ui-monospace, monospace";
      ctx.fillText(label, padL + 2, y - 3);
    };
    level(callWall, "#ff5470", "CALL WALL", false);
    level(putWall, "#2ec785", "PUT WALL", false);
    level(gammaFlip, "#ffb648", "γ-FLIP", true);
    level(vwap, "#4aa8ff", "VWAP", true);

    // --- projection cone (violet) from now to end ---
    const yUp = Y(spot + band), yDn = Y(spot - band), yMid = Y(spot);
    const xEnd = X(tEnd);
    const grad = ctx.createLinearGradient(xNow, 0, xEnd, 0);
    grad.addColorStop(0, "rgba(157,123,255,0.02)");
    grad.addColorStop(1, "rgba(157,123,255,0.20)");
    ctx.fillStyle = grad;
    ctx.beginPath();
    ctx.moveTo(xNow, yMid);
    ctx.lineTo(xEnd, yUp);
    ctx.lineTo(xEnd, yDn);
    ctx.closePath();
    ctx.fill();
    ctx.strokeStyle = "rgba(157,123,255,0.5)";
    ctx.setLineDash([4, 4]);
    ctx.beginPath(); ctx.moveTo(xNow, yMid); ctx.lineTo(xEnd, yUp); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(xNow, yMid); ctx.lineTo(xEnd, yDn); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = "#9d7bff";
    ctx.textAlign = "right";
    ctx.font = "9px ui-monospace, monospace";
    ctx.fillText("+" + band.toFixed(1), xEnd - 2, yUp + 10);
    ctx.fillText("-" + band.toFixed(1), xEnd - 2, yDn - 3);

    // --- spot area + line ---
    const areaGrad = ctx.createLinearGradient(0, padT, 0, padT + plotH);
    areaGrad.addColorStop(0, "rgba(230,237,247,0.14)");
    areaGrad.addColorStop(1, "rgba(230,237,247,0)");
    ctx.beginPath();
    pts.forEach((p, i) => { const x = X(p.ts), y = Y(p.spot); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
    ctx.lineTo(X(pts[pts.length - 1].ts), padT + plotH);
    ctx.lineTo(X(pts[0].ts), padT + plotH);
    ctx.closePath();
    ctx.fillStyle = areaGrad;
    ctx.fill();

    ctx.beginPath();
    pts.forEach((p, i) => { const x = X(p.ts), y = Y(p.spot); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
    ctx.strokeStyle = "#e6edf7";
    ctx.lineWidth = 1.75;
    ctx.stroke();

    // --- trade-event markers (rebuild tooltip hit-boxes every draw) ---
    chartHits = [];
    const spotAt = (ts) => {
      let best = pts[0];
      for (const p of pts) if (Math.abs(p.ts - ts) < Math.abs(best.ts - ts)) best = p;
      return best.spot;
    };
    const tri = (x, y, r, up) => {
      ctx.beginPath();
      ctx.moveTo(x, y + (up ? -r : r));
      ctx.lineTo(x - r, y + (up ? r * 0.8 : -r * 0.8));
      ctx.lineTo(x + r, y + (up ? r * 0.8 : -r * 0.8));
      ctx.closePath();
    };
    const diamond = (x, y, r) => {
      ctx.beginPath();
      ctx.moveTo(x, y - r); ctx.lineTo(x + r, y); ctx.lineTo(x, y + r); ctx.lineTo(x - r, y);
      ctx.closePath();
    };
    const mark = (x, y, label) => {
      if (x >= padL - 6 && x <= padL + plotW + 6) chartHits.push({ x, y, label });
    };

    // TRADE signals from the journal ticks: ▲ call / ▼ put / ◆ neutral
    const sigTicks = ticks.filter((t) => t.decision === "TRADE" && num(t.spot) != null);
    sigTicks.forEach((t) => {
      const ts = new Date(t.ts).getTime();
      const x = X(ts), y = Y(num(t.spot));
      const dir = t.regime_direction;
      ctx.strokeStyle = "#0a0e14"; ctx.lineWidth = 1.25;
      if (dir === "call") {
        tri(x, y - 9, 5, true); ctx.fillStyle = "#2ec785";
      } else if (dir === "put") {
        tri(x, y + 9, 5, false); ctx.fillStyle = "#ff5470";
      } else {
        diamond(x, y - 9, 5); ctx.fillStyle = "#9d7bff";
      }
      ctx.fill(); ctx.stroke();
      const fam = t.selected_family || "signal";
      const strikes = [t.short_strikes, t.long_strikes]
        .map((s) => Array.isArray(s) ? s.join("/") : "").filter(Boolean).join(" | ");
      mark(x, y, `SIGNAL ${fam}${strikes ? " " + strikes : ""}${dir && dir !== "none" ? " (" + dir + ")" : ""} · ${new Date(ts).toLocaleTimeString("en-US", { timeZone: "America/New_York", hour: "numeric", minute: "2-digit" })}`);
    });

    // paper entries/exits from /api/trades (open + closed)
    const trades = (lastChartData.trades || {});
    const paperEvents = [];
    (trades.open || []).forEach((p) => {
      paperEvents.push({ ts: new Date(p.opened_at).getTime(), kind: "entry",
        label: `ENTRY ${p.family} ${p.strikes} ×${p.contracts} (open)` });
    });
    (trades.closed || []).forEach((p) => {
      const o = new Date(p.opened_at).getTime(), c = new Date(p.closed_at).getTime();
      paperEvents.push({ ts: o, kind: "entry",
        label: `ENTRY ${p.family} ${p.strikes} ×${p.contracts}` });
      const pnl = num(p.pnl_dollars);
      paperEvents.push({ ts: c, kind: "exit", pnl,
        label: `EXIT ${p.family} ${p.exit_reason} ${pnl != null ? (pnl >= 0 ? "+$" : "-$") + Math.abs(pnl).toFixed(2) : ""}` });
    });
    paperEvents.forEach((e) => {
      if (!isFinite(e.ts) || e.ts < t0 - 60e3 || e.ts > tLast + 60e3) return;
      const x = X(e.ts), y = Y(spotAt(e.ts));
      if (e.kind === "entry") {
        ctx.beginPath(); ctx.arc(x, y, 5.5, 0, Math.PI * 2);
        ctx.strokeStyle = "#34d5e0"; ctx.lineWidth = 2; ctx.stroke();
      } else {
        ctx.strokeStyle = e.pnl != null && e.pnl < 0 ? "#ff5470" : "#ffb648";
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(x - 4, y - 4); ctx.lineTo(x + 4, y + 4);
        ctx.moveTo(x + 4, y - 4); ctx.lineTo(x - 4, y + 4);
        ctx.stroke();
      }
      mark(x, y, e.label);
    });

    // --- last price dot + tag ---
    const lp = pts[pts.length - 1];
    const lx = X(lp.ts), ly = Y(lp.spot);
    ctx.beginPath(); ctx.arc(lx, ly, 4.5, 0, Math.PI * 2);
    ctx.fillStyle = "#4aa8ff"; ctx.fill();
    ctx.strokeStyle = "#e6edf7"; ctx.lineWidth = 1.5; ctx.stroke();

    ctx.restore();                       // end plot clip; margin tag stays visible

    ctx.fillStyle = "#0a0e14";
    const tag = spot.toFixed(2);
    ctx.font = "bold 11px ui-monospace, monospace";
    const tw = ctx.measureText(tag).width + 10;
    ctx.fillStyle = "#4aa8ff";
    roundRect(ctx, padL + plotW + 4, ly - 8, tw, 16, 3); ctx.fill();
    ctx.fillStyle = "#04121f"; ctx.textAlign = "center";
    ctx.fillText(tag, padL + plotW + 4 + tw / 2, ly + 3);
  }

  function roundRect(ctx, x, y, w, h, r) {
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.arcTo(x + w, y, x + w, y + h, r);
    ctx.arcTo(x + w, y + h, x, y + h, r);
    ctx.arcTo(x, y + h, x, y, r);
    ctx.arcTo(x, y, x + w, y, r);
    ctx.closePath();
  }

  /* ---------------- staleness note ---------------- */
  function staleNote(live, market) {
    const host = $("signal-panel");
    let note = host.querySelector(".stale-note");
    let msg = "";
    const age = live.ts ? (Date.now() - new Date(live.ts).getTime()) / 1000 : Infinity;
    if (age > 180) {
      // No fresh heartbeat in >3 min — the pipeline process itself is likely down.
      msg = live.ts
        ? "Pipeline offline — no update in " + Math.floor(age / 60) + " min (check zerodte-shadow service)."
        : "Pipeline offline — no data received yet (check zerodte-shadow service).";
    } else if (live.status && live.status !== "live" && live.note) {
      // Fresh heartbeat with a reason (feed down / market closed).
      msg = live.note;
    }
    if (msg) {
      if (!note) { note = document.createElement("div"); note.className = "stale-note"; host.insertBefore(note, host.firstChild.nextSibling); }
      note.textContent = msg;
    } else if (note) { note.remove(); }
  }

  /* ---------------- predictive power (calibration readouts) ---------------- */
  function renderPredict(report) {
    const cal = (report && report.calibration) || {};
    const d = (cal.directional && cal.directional.overall) || {};
    const pp = cal.prob_profit || {};
    const ev = cal.ev || {};
    const cards = [];
    if (d.n) {
      const hitCls = num(d.hit_rate) >= 0.52 ? "pos" : num(d.hit_rate) < 0.5 ? "neg" : "warn";
      cards.push(metricCard("Dir. hit rate", `${pct(d.hit_rate)} (n=${d.n})`, hitCls));
      cards.push(metricCard("Signed move", fmt(d.avg_fwd_move_pct, 3) + "%",
                            num(d.avg_fwd_move_pct) > 0 ? "pos" : "neg"));
    } else {
      cards.push(metricCard("Dir. hit rate", "no sample"));
    }
    if (pp.n) {
      cards.push(metricCard("Brier skill", `${fmt(pp.brier_skill, 2)} (n=${pp.n})`,
                            num(pp.brier_skill) > 0 ? "pos" : "neg"));
      cards.push(metricCard("Base rate", pct(pp.base_rate)));
    }
    if (ev.n) {
      const bias = num(ev.mean_ev_error);
      cards.push(metricCard("EV bias", (bias >= 0 ? "+" : "") + fmt(bias, 3),
                            Math.abs(bias) <= 0.10 ? "pos" : "warn"));
      cards.push(metricCard("EV MAE", fmt(ev.mae_ev_error, 3)));
    }
    $("predict-metrics").innerHTML = cards.length
      ? cards.join("")
      : '<p class="empty">No settled data yet — prediction is scored at settlement</p>';
  }

  /* ---------------- trade journal tab ---------------- */
  let activeTab = "legacy";

  function switchTab(tab) {
    activeTab = tab;
    $("tab-legacy").classList.toggle("active", tab === "legacy");
    $("tab-v2").classList.toggle("active", tab === "v2");
    $("tab-journal").classList.toggle("active", tab === "journal");
    $("tab-validation").classList.toggle("active", tab === "validation");
    $("tab-learning").classList.toggle("active", tab === "learning");
    $("view-legacy").classList.toggle("hidden", tab !== "legacy");
    $("view-v2").classList.toggle("hidden", tab !== "v2");
    $("view-journal").classList.toggle("hidden", tab !== "journal");
    $("view-validation").classList.toggle("hidden", tab !== "validation");
    $("view-learning").classList.toggle("hidden", tab !== "learning");
    if (tab === "journal") refreshJournal();
    if (tab === "validation") refreshValidation();
    if (tab === "learning") refreshLearning();
  }

  function entryLogicLine(ctx) {
    if (!ctx) return "—";
    const parts = [];
    if (ctx.cell) parts.push(esc(ctx.cell.join(" × ")));
    if (ctx.conviction && ctx.conviction !== "NONE") parts.push(esc(ctx.conviction));
    if (ctx.capture) parts.push(esc(ctx.capture));
    const nums = [];
    if (num(ctx.gate_score) != null) nums.push("gate " + fmt(ctx.gate_score, 1));
    if (num(ctx.ev) != null) nums.push("EV $" + fmt(ctx.ev, 2));
    if (num(ctx.ev_per_risk) != null) nums.push(fmt(ctx.ev_per_risk, 2) + "/risk");
    if (num(ctx.prob_profit) != null) nums.push("PoP " + pct(ctx.prob_profit));
    if (num(ctx.size_mult) != null) nums.push("×" + fmt(ctx.size_mult, 2));
    if (num(ctx.equity_at_entry) != null) nums.push("eq $" + fmt(ctx.equity_at_entry, 0));
    return `<span class="tj-why">${parts.join(" · ")}</span>` +
           (nums.length ? `<span class="tj-nums">${nums.join(" · ")}</span>` : "");
  }

  function exitLogicLine(t) {
    const map = {
      stop:   "stop-loss: loss reached its fraction of max loss",
      target: "profit target: captured its fraction of max profit",
      trail:  "trailing stop: gave back too much of peak profit",
      eod:    "end of day: forced flat before the close",
      ras_invalidate: "regime alignment: the regime moved against the position's thesis",
    };
    return map[t.exit_reason] || esc(t.exit_reason || "—");
  }

  function renderJournal(data) {
    const open = data.open || [];
    $("tj-open-count").textContent = String(open.length);
    if (!open.length) {
      $("tj-open").innerHTML = '<p class="empty">No open positions</p>';
    } else {
      $("tj-open").innerHTML = open.map((p) => {
        const pnl = num(p.unrealized_pnl_dollars);
        const cls = pnl > 0 ? "pos" : pnl < 0 ? "neg" : "";
        return `<div class="tj-open-row">
          <div class="tj-head">
            <b>${esc(p.family)}</b> <span class="mono">${esc(p.strikes)}</span>
            <span>×${p.contracts}</span>
            <span class="mono ${cls}">${pnl != null ? (pnl >= 0 ? "+$" : "-$") + Math.abs(pnl).toFixed(2) : "—"}</span>
            <span class="tj-dim">${fmt(p.hold_min, 0)}m held</span>
            ${rasInline(p.entry_ctx || {})}
          </div>
          <div class="tj-sub">${entryLogicLine(p.entry_ctx)}</div>
          ${rasBlock(p.entry_ctx || {})}
        </div>`;
      }).join("");
    }

    const closed = data.closed || [];
    $("tj-closed-count").textContent = String(closed.length);
    $("tj-empty").classList.toggle("hidden", closed.length > 0);
    $("tj-table").classList.toggle("hidden", closed.length === 0);
    $("tj-body").innerHTML = closed.map((t) => {
      const pnl = num(t.pnl_dollars);
      const cls = pnl > 0 ? "pos" : pnl < 0 ? "neg" : "";
      const opened = (t.opened_at || "").slice(11, 16);
      const reasonCls = t.exit_reason === "target" || t.exit_reason === "trail" ? "good"
                      : t.exit_reason === "stop" ? "bad"
                      : t.exit_reason === "ras_invalidate" ? "warn" : "";
      return `<tr>
        <td class="mono">${esc((t.opened_at || "").slice(0, 10))} ${opened}</td>
        <td><b>${esc(t.family)}</b> <span class="mono tj-dim">${esc(t.strikes)}</span>
            <div class="tj-sub">${entryLogicLine(t.entry_ctx)}</div></td>
        <td class="mono">×${t.contracts}</td>
        <td class="mono">${fmt(t.entry_credit, 2)}</td>
        <td class="mono">${fmt(t.exit_value, 2)}</td>
        <td class="mono">${fmt(t.hold_min, 0)}m</td>
        <td class="mono ${cls}">${pnl != null ? (pnl >= 0 ? "+$" : "-$") + Math.abs(pnl).toFixed(2) : "—"}</td>
        <td><span class="tj-reason ${reasonCls}">${esc(t.exit_reason || "—")}</span>
            <div class="tj-sub">${exitLogicLine(t)}</div>
            ${rasExitLine(t.entry_ctx)}</td>
        <td class="mono">$${fmt(t.equity_after, 2)}</td>
      </tr>`;
    }).join("");
  }

  function drawEquityCurve(closed) {
    const panel = $("tj-equity-panel");
    const eq = closed.slice().reverse()
      .map((t) => num(t.equity_after)).filter((v) => v != null);
    if (eq.length < 2) { panel.classList.add("hidden"); return; }
    panel.classList.remove("hidden");
    $("tj-equity-now").textContent = "$" + eq[eq.length - 1].toFixed(2);

    const canvas = $("tj-equity");
    const wrap = canvas.parentElement;
    const dpr = window.devicePixelRatio || 1;
    const W = wrap.clientWidth || 600, H = wrap.clientHeight || 160;
    canvas.width = W * dpr; canvas.height = H * dpr;
    canvas.style.width = W + "px"; canvas.style.height = H + "px";
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);

    const lo = Math.min(...eq), hi = Math.max(...eq);
    const pad = 12, span = (hi - lo) || 1;
    const x = (i) => pad + (W - 2 * pad) * (eq.length === 1 ? 0 : i / (eq.length - 1));
    const y = (v) => H - pad - (H - 2 * pad) * ((v - lo) / span);

    ctx.strokeStyle = "rgba(230,237,247,0.15)";
    ctx.setLineDash([4, 4]);
    ctx.beginPath(); ctx.moveTo(pad, y(eq[0])); ctx.lineTo(W - pad, y(eq[0])); ctx.stroke();
    ctx.setLineDash([]);

    const up = eq[eq.length - 1] >= eq[0];
    ctx.strokeStyle = up ? "#2ec785" : "#ff5470";
    ctx.lineWidth = 2;
    ctx.lineJoin = "round";
    ctx.beginPath();
    eq.forEach((v, i) => (i ? ctx.lineTo(x(i), y(v)) : ctx.moveTo(x(i), y(v))));
    ctx.stroke();
  }

  async function refreshJournal() {
    try {
      const data = await api("/api/trades?limit=200");
      renderJournal(data);
      drawEquityCurve(data.closed || []);
    } catch (e) {
      if (e.message !== "Unauthorized") console.warn("journal", e);
    }
  }

  /* ---------------- validation tab ---------------- */
  let valFilter = "";
  let valSelectedId = null;
  let valReports = [];

  function valStatus(rep) {
    const flags = rep.flags || [];
    if (flags.some((f) => f && f.severity === "alert")) return "alert";
    if (flags.some((f) => f && f.severity === "warn")) return "warn";
    return "ok";
  }

  const VAL_TYPE_LABEL = { daily: "Daily", weekly: "Weekly", feature_impact: "Feature" };

  function renderValList() {
    $("val-count").textContent = String(valReports.length);
    if (!valReports.length) {
      const emptyMsg = {
        "": "No validation reports yet — the scheduled pipeline will populate this after its first run",
        daily: "No daily reports yet",
        weekly: "No weekly reports yet",
        feature_impact: "No feature-impact reports yet — these are generated manually via scripts/feature_impact.py",
      }[valFilter] || "No validation reports yet";
      $("val-list").innerHTML = `<p class="empty">${emptyMsg}</p>`;
      return;
    }
    $("val-list").innerHTML = valReports.map((r) => {
      const st = valStatus(r);
      const sel = r.id === valSelectedId ? " selected" : "";
      return `<button class="val-row${sel}" type="button" data-id="${r.id}">
        <span class="val-dot ${st}"></span>
        <span class="mono val-date">${esc(r.report_date || "—")}</span>
        <span class="val-type ${esc(r.report_type)}">${esc(VAL_TYPE_LABEL[r.report_type] || r.report_type)}</span>
        <span class="val-sum">${esc((r.summary || "").slice(0, 90))}</span>
      </button>`;
    }).join("");
    document.querySelectorAll("#val-list .val-row").forEach((el) => {
      el.addEventListener("click", () => {
        valSelectedId = +el.dataset.id;
        renderValList();
        const rep = valReports.find((r) => r.id === valSelectedId);
        if (rep) renderValDetail(rep);
      });
    });
  }

  function valDelta(v, invert) {
    const n = num(v);
    if (n == null || n === 0) return "";
    const good = invert ? n < 0 : n > 0;
    return `<span class="val-delta ${good ? "pos" : "neg"}">${sign(n, 3)}</span>`;
  }

  function renderValDetail(rep) {
    $("val-detail-date").textContent = `${rep.report_date || "—"} · ${rep.report_type || ""}`;
    const m = rep.metrics || {};
    const jm = m.journal || {};
    const wf = m.walk_forward || null;
    const deltas = m.deltas || {};
    const parts = [];

    parts.push(`<p class="val-summary">${esc(rep.summary || "—")}</p>`);

    const flags = rep.flags || [];
    if (flags.length) {
      parts.push('<div class="chips">' + flags.map((f) => {
        const sev = (f && f.severity) || "info";
        const cls = sev === "alert" ? "veto" : sev === "warn" ? "warn" : "";
        const label = typeof f === "string" ? f : `${f.flag}: ${f.detail}`;
        return `<span class="chip ${cls}" title="${esc(label)}">${esc(typeof f === "string" ? f : f.flag)}</span>`;
      }).join("") + "</div>");
    }

    if (rep.report_type === "feature_impact") {
      parts.push(renderFeatureImpactDetail(m));
    } else {
      const gate = jm.gate_effectiveness || {};
      const taken = (gate.trades_taken || {}).mean;
      const blocked = (gate.blocked_by_gate || {}).mean;
      const gateEdge = (taken != null && blocked != null) ? taken - blocked : null;
      const cards = [
        metricCard("Win rate", pct(jm.win_rate) + valDelta(deltas.win_rate)),
        metricCard("Mean P&L / trade", sign(jm.mean_pnl_per_trade, 4) + valDelta(deltas.mean_pnl_per_trade)),
        metricCard("Settled trades", jm.n_settled_trades != null ? String(jm.n_settled_trades) : "—"),
        metricCard("WF Sharpe", wf ? sign(wf.mean_sharpe, 2) + valDelta(deltas.mean_sharpe) : "—"),
        metricCard("WF folds profitable", wf
          ? `${wf.n_profitable}/${wf.n_valid_folds != null ? wf.n_valid_folds : wf.n_folds}`
          : "—"),
        metricCard("Gate edge", gateEdge != null ? sign(gateEdge, 4) : "—",
                   gateEdge != null ? (gateEdge >= 0 ? "pos" : "neg") : ""),
        metricCard("Brier skill", sign(jm.brier_skill, 3) + valDelta(deltas.brier_skill),
                   num(jm.brier_skill) != null ? (jm.brier_skill >= 0 ? "pos" : "neg") : ""),
        metricCard("EV bias", sign(jm.ev_bias, 4)),
        metricCard("Directional hit", pct(jm.directional_hit_rate)),
        metricCard("Distinct regimes", ((jm.regime_diversity || {}).distinct != null)
                   ? String(jm.regime_diversity.distinct) : "—"),
      ];
      parts.push(`<div class="metrics">${cards.join("")}</div>`);

      const ras = jm.ras || {};
      if (ras.n) {
        const acts = Object.entries(ras.actions || {})
          .map(([k, v]) => `${esc(k)} ×${v}`).join(" · ");
        parts.push(`<p class="val-note">RAS: ${ras.n} evaluations, mean score ${fmt(ras.score_mean, 1)} — ${acts}</p>`);
      }

      if (m.per_regime && Object.keys(m.per_regime).length) {
        const rows = Object.entries(m.per_regime).map(([regime, b]) => {
          const t = b.taken || {}, bl = b.blocked || {};
          return `<tr><td>${esc(regime)}</td>
            <td class="mono">${t.n || 0}</td>
            <td class="mono ${num(t.mean_pnl) > 0 ? "pos" : num(t.mean_pnl) < 0 ? "neg" : ""}">${t.mean_pnl != null ? sign(t.mean_pnl, 4) : "—"}</td>
            <td class="mono">${t.win_rate != null ? pct(t.win_rate, 0) : "—"}</td>
            <td class="mono">${bl.n || 0}</td>
            <td class="mono">${bl.mean_pnl != null ? sign(bl.mean_pnl, 4) : "—"}</td></tr>`;
        }).join("");
        parts.push(`<h3 class="val-h3">Per-regime performance</h3>
          <table class="val-table"><thead><tr>
            <th>Regime</th><th>Taken n</th><th>Mean P&amp;L</th><th>Win%</th>
            <th>Blocked n</th><th>Blocked P&amp;L</th>
          </tr></thead><tbody>${rows}</tbody></table>`);
      }

      if (Array.isArray(m.recommendations) && m.recommendations.length) {
        parts.push('<h3 class="val-h3">Recommendations</h3><ul class="val-recs">'
          + m.recommendations.map((r) => `<li>${esc(r)}</li>`).join("") + "</ul>");
      }

      if (m.feature_contributions && typeof m.feature_contributions === "object") {
        const entries = Object.entries(m.feature_contributions)
          .filter(([k, v]) => k !== "n" && k !== "note" && typeof v === "number")
          .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1])).slice(0, 12);
        if (entries.length) {
          parts.push('<h3 class="val-h3">Feature contributions (r vs realized P&amp;L)</h3>'
            + '<div class="val-corr">' + entries.map(([k, v]) =>
              `<div class="val-corr-row"><span class="mono">${esc(k)}</span>
               <span class="mono ${v >= 0 ? "pos" : "neg"}">${sign(v, 3)}</span></div>`).join("") + "</div>");
        }
      }
    }

    if (rep.notes) parts.push(`<p class="val-note">${esc(rep.notes)}</p>`);
    if (rep.generated_at) {
      parts.push(`<p class="val-note">generated ${esc(rep.generated_at.slice(0, 19).replace("T", " "))} UTC</p>`);
    }
    $("val-detail").innerHTML = parts.join("");
  }

  function renderFeatureImpactDetail(m) {
    const parts = [];
    if (m.feature) parts.push(`<p class="val-note">Feature: <b class="mono">${esc(m.feature)}</b></p>`);
    if (m.recommendation) {
      const cls = /strong positive|positive/i.test(m.recommendation) ? "pos"
                : /negative/i.test(m.recommendation) ? "neg" : "";
      parts.push(`<p class="val-reco ${cls}">Recommendation: ${esc(m.recommendation)}</p>`);
    }
    const d = m.deltas || {};
    const cards = Object.entries(d)
      .filter(([, v]) => typeof v === "number")
      .map(([k, v]) => {
        const invert = /drawdown/.test(k);
        const good = invert ? v <= 0 : v >= 0;
        return metricCard("Δ " + k.replace(/_/g, " "), sign(v, 4), v === 0 ? "" : good ? "pos" : "neg");
      });
    if (cards.length) parts.push(`<div class="metrics">${cards.join("")}</div>`);

    const side = (label, s) => {
      if (!s) return "";
      return `<div class="val-side"><h3 class="val-h3">${esc(label)}</h3><div class="metrics">`
        + metricCard("Sharpe", sign((s.backtest || {}).sharpe, 2))
        + metricCard("Win rate", pct((s.backtest || {}).win_rate))
        + metricCard("Total P&L", sign((s.backtest || {}).total_pnl, 4))
        + metricCard("Max DD", fmt((s.backtest || {}).max_drawdown, 4))
        + metricCard("WF Sharpe", s.walk_forward ? sign(s.walk_forward.mean_sharpe, 2) : "—")
        + metricCard("WF profitable", s.walk_forward
          ? `${s.walk_forward.n_profitable}/${s.walk_forward.n_valid_folds != null ? s.walk_forward.n_valid_folds : s.walk_forward.n_folds}`
          : "—")
        + "</div></div>";
    };
    parts.push('<div class="val-sides">' + side("Baseline", m.baseline) + side("With feature", m.variant) + "</div>");
    return parts.join("");
  }

  async function refreshValidation() {
    try {
      const q = valFilter ? `?report_type=${valFilter}&limit=100` : "?limit=100";
      const data = await api("/api/validation" + q);
      valReports = data.reports || [];
      // default selection: most recent daily report, else the most recent
      if (valSelectedId == null || !valReports.some((r) => r.id === valSelectedId)) {
        const daily = valReports.find((r) => r.report_type === "daily");
        valSelectedId = (daily || valReports[0] || {}).id ?? null;
      }
      renderValList();
      const rep = valReports.find((r) => r.id === valSelectedId);
      if (rep) renderValDetail(rep);
      else {
        $("val-detail-date").textContent = "—";
        $("val-detail").innerHTML = '<p class="empty">Select a report</p>';
      }
    } catch (e) {
      if (e.message !== "Unauthorized") console.warn("validation", e);
    }
  }

  function initValidationControls() {
    document.querySelectorAll("#val-filters .val-filter").forEach((el) => {
      el.addEventListener("click", () => {
        valFilter = el.dataset.type || "";
        document.querySelectorAll("#val-filters .val-filter")
          .forEach((b) => b.classList.toggle("active", b === el));
        valSelectedId = null;
        refreshValidation();
      });
    });
  }

  // Command-center awareness: light the Validation tab dot when the most
  // recent report carries an alert-severity degradation flag.
  function renderValidationBadge(latest) {
    const reports = (latest && latest.reports) || [];
    const hasAlert = reports.length > 0 && valStatus(reports[0]) === "alert";
    $("tab-validation-dot").classList.toggle("hidden", !hasAlert);
  }

  /* ---------------- learning tab (adaptive learning engine) ---------------- */
  const SEV_CLS = { alert: "veto", warn: "warn", info: "" };

  function renderLearningDiagnostics(runs) {
    const latest = runs.find((r) => (r.diagnostics || []).length) || runs[0];
    const diags = (latest && latest.diagnostics) || [];
    $("lrn-diag-sub").textContent = latest
      ? `run ${String(latest.run_id || "").slice(0, 8)} · ${latest.mode || ""}`
      : "—";
    if (!diags.length) {
      $("lrn-diagnostics").innerHTML =
        '<p class="empty">Latest learning run found no diagnosable failure</p>';
      return;
    }
    $("lrn-diagnostics").innerHTML = diags.map((d) => `
      <div class="lrn-card">
        <div class="lrn-card-head">
          <span class="chip ${SEV_CLS[d.severity] || ""}">${esc(d.severity || "info")}</span>
          <b class="mono">${esc(d.issue || "?")}</b>
          <span class="tj-dim">conf ${pct(d.confidence)}</span>
        </div>
        <div class="tj-sub">${esc(d.likely_cause || "")}</div>
        <div class="tj-sub lrn-reco">→ ${esc(d.recommendation || "")}</div>
      </div>`).join("");
  }

  function renderLearningPromotions(promos) {
    $("lrn-promo-sub").textContent = String(promos.length);
    if (!promos.length) {
      $("lrn-promotions").innerHTML = '<p class="empty">No promotion decisions yet</p>';
      return;
    }
    $("lrn-promotions").innerHTML = promos.slice(0, 8).map((p) => {
      const rules = ((p.decision || {}).rules) || [];
      const stCls = p.status === "pending_review" ? "warn"
                  : p.status === "approved" ? "ok" : "veto";
      const pending = p.status === "pending_review";
      return `<div class="lrn-card ${pending ? "lrn-pending" : ""}">
        <div class="lrn-card-head">
          <b class="mono">${esc(String(p.config_id || "").slice(0, 8))}</b>
          <span class="chip ${stCls}">${esc(p.status || "?")}</span>
          <span class="tj-dim mono">${esc((p.created_at || "").slice(0, 16).replace("T", " "))}</span>
        </div>
        <div class="lrn-rules">${rules.map((r) =>
          `<span class="lrn-rule ${r.passed ? "ok" : "bad"}" title="${esc(r.detail || "")}">${r.passed ? "✓" : "✗"} ${esc(r.name)}</span>`).join("")}</div>
        ${pending ? `<div class="tj-sub lrn-reco">awaiting human review — <span class="mono">python3 -m adaptive_learning.promoter --approve ${esc(String(p.config_id || "").slice(0, 8))}</span></div>` : ""}
      </div>`;
    }).join("");
  }

  function renderLearningDrift(reports) {
    $("lrn-drift-sub").textContent = String(reports.length);
    if (!reports.length) {
      $("lrn-drift").innerHTML = '<p class="empty">No drift snapshots yet</p>';
      return;
    }
    $("lrn-drift").innerHTML = reports.slice(0, 6).map((r) => {
      const drifts = ((r.metrics || {}).drifts) || [];
      const chips = drifts.length
        ? drifts.map((d) => `<span class="chip warn" title="${esc(d.detail || "")}">${esc(d.kind)}: ${esc(d.metric)} ${sign(d.rel_change, 2)}</span>`).join("")
        : '<span class="chip">no drift</span>';
      return `<div class="lrn-card">
        <div class="lrn-card-head"><b class="mono">${esc(r.report_date || "?")}</b></div>
        <div class="chips">${chips}</div>
      </div>`;
    }).join("");
  }

  function overridesLine(ov) {
    const entries = Object.entries(ov || {});
    if (!entries.length) return '<span class="tj-dim">defaults</span>';
    return entries.map(([k, v]) =>
      `<span class="mono lrn-ov">${esc(k)}=${esc(String(v))}</span>`).join(" ");
  }

  function renderLearningCandidates(data) {
    const champ = data.champion;
    const cands = data.candidates || [];
    $("lrn-cand-sub").textContent = String(cands.length);
    $("lrn-champion").innerHTML = champ
      ? `<div class="lrn-card lrn-champ">
          <div class="lrn-card-head">
            <span class="chip ok">CHAMPION</span>
            <b class="mono">${esc(String(champ.config_id || "").slice(0, 8))}</b>
            <span class="tj-dim">${esc(champ.label || "")}</span>
          </div>
          <div class="tj-sub">${overridesLine(champ.overrides)}</div>
        </div>`
      : '<p class="empty">No champion config — dataclass defaults live</p>';
    if (!cands.length) {
      $("lrn-candidates").innerHTML = '<p class="empty">No candidate configs yet</p>';
      return;
    }
    const rows = cands.slice(0, 10).map((c) => {
      const m = c.metrics || {};
      const stCls = c.status === "pending_review" ? "warn"
                  : c.status === "promoted" ? "good"
                  : c.status === "rejected" ? "bad" : "";
      return `<tr>
        <td class="mono">${esc(String(c.config_id || "").slice(0, 8))}</td>
        <td>${esc(c.label || "")}<div class="tj-sub">${overridesLine(c.overrides)}</div></td>
        <td class="mono">${m.search_score != null ? sign(m.search_score, 3) : "—"}</td>
        <td class="mono">${m.holdout_score != null ? sign(m.holdout_score, 3) : "—"}</td>
        <td class="mono">${m.champion_score != null ? sign(m.champion_score, 3) : "—"}</td>
        <td><span class="tj-reason ${stCls}">${esc(c.status || "?")}</span></td>
      </tr>`;
    }).join("");
    $("lrn-candidates").innerHTML = `<table class="val-table"><thead><tr>
      <th>Config</th><th>Label / overrides</th><th>Score</th><th>Holdout</th>
      <th>Champion</th><th>Status</th></tr></thead><tbody>${rows}</tbody></table>`;
  }

  function renderLearningRuns(runs) {
    $("lrn-runs-sub").textContent = String(runs.length);
    if (!runs.length) {
      $("lrn-runs").innerHTML = '<p class="empty">No learning runs yet</p>';
      return;
    }
    const rows = runs.slice(0, 10).map((r) => {
      const outCls = r.outcome === "promotion_recommended" ? "good"
                   : r.outcome === "rejected" ? "bad" : "";
      return `<tr>
        <td class="mono">${esc((r.started_at || "").slice(0, 16).replace("T", " "))}</td>
        <td>${esc(r.mode || "?")}</td>
        <td class="mono">${(r.diagnostics || []).length}</td>
        <td class="mono">${r.n_trials != null ? r.n_trials : "—"}</td>
        <td class="mono">${r.best_score != null ? sign(r.best_score, 3) : "—"}</td>
        <td class="mono">${r.holdout_score != null ? sign(r.holdout_score, 3) : "—"}</td>
        <td><span class="tj-reason ${outCls}">${esc(r.outcome || "?")}</span></td>
      </tr>`;
    }).join("");
    $("lrn-runs").innerHTML = `<table class="val-table"><thead><tr>
      <th>Started</th><th>Mode</th><th>Diag</th><th>Trials</th><th>Best</th>
      <th>Holdout</th><th>Outcome</th></tr></thead><tbody>${rows}</tbody></table>`;
  }

  function renderLearningFeatures(features) {
    $("lrn-feat-sub").textContent = String(features.length);
    if (!features.length) {
      $("lrn-features").innerHTML = '<p class="empty">No feature scores yet</p>';
      return;
    }
    const order = { production: 0, candidate: 1, experimental: 2, observation: 3 };
    const sorted = features.slice().sort((a, b) =>
      (order[a.status] ?? 9) - (order[b.status] ?? 9)
      || Math.abs(b.pearson || 0) - Math.abs(a.pearson || 0));
    const rows = sorted.slice(0, 20).map((f) => {
      const stCls = f.status === "candidate" || f.status === "production" ? "good"
                  : f.status === "experimental" ? "warn" : "";
      return `<tr>
        <td class="mono">${esc(f.feature)}</td>
        <td class="mono ${num(f.pearson) > 0 ? "pos" : num(f.pearson) < 0 ? "neg" : ""}">${f.pearson != null ? sign(f.pearson, 3) : "—"}</td>
        <td class="mono">${f.spearman != null ? sign(f.spearman, 3) : "—"}</td>
        <td class="mono">${f.mutual_info != null ? fmt(f.mutual_info, 3) : "—"}</td>
        <td class="mono">${f.perm_importance != null ? sign(f.perm_importance, 3) : "—"}</td>
        <td class="mono">${f.stability != null ? fmt(f.stability, 2) : "—"}</td>
        <td><span class="tj-reason ${stCls}">${esc(f.status || "observation")}</span></td>
      </tr>`;
    }).join("");
    $("lrn-features").innerHTML = `<table class="val-table"><thead><tr>
      <th>Feature</th><th>Pearson</th><th>Spearman</th><th>MI</th>
      <th>Perm</th><th>Stability</th><th>Lifecycle</th></tr></thead>
      <tbody>${rows}</tbody></table>`;
  }

  async function refreshLearning() {
    try {
      const [learning, cands, promos, feats, drift] = await Promise.all([
        api("/api/learning?limit=50").catch(() => ({})),
        api("/api/candidates?limit=50").catch(() => ({})),
        api("/api/promotions?limit=50").catch(() => ({})),
        api("/api/feature-scores").catch(() => ({})),
        api("/api/drift?limit=10").catch(() => ({})),
      ]);
      renderLearningDiagnostics(learning.runs || []);
      renderLearningRuns(learning.runs || []);
      renderLearningCandidates(cands || {});
      renderLearningPromotions(promos.promotions || []);
      renderLearningFeatures(feats.features || []);
      renderLearningDrift(drift.reports || []);
    } catch (e) {
      if (e.message !== "Unauthorized") console.warn("learning", e);
    }
  }

  // Light the Learning tab dot when a candidate awaits human review.
  function renderLearningBadge(promos) {
    const pending = ((promos && promos.promotions) || [])
      .some((p) => p.status === "pending_review");
    $("tab-learning-dot").classList.toggle("hidden", !pending);
  }

  /* ---------------- refresh loop ---------------- */
  async function refresh() {
    if (activeTab === "journal") refreshJournal();
    if (activeTab === "validation") refreshValidation();
    if (activeTab === "learning") refreshLearning();
    try {
      let [live, market, history, report, paper, readiness, trades, valLatest, pendingPromos] = await Promise.all([
        api("/api/live"),
        api("/api/market-status"),
        api("/api/ticks?limit=200"),
        api("/api/report").catch(() => ({})),
        api("/api/paper").catch(() => ({})),
        api("/api/readiness").catch(() => ({})),
        api("/api/trades?limit=100").catch(() => ({})),
        api("/api/validation?limit=1").catch(() => ({})),
        api("/api/promotions?status=pending_review&limit=1").catch(() => ({})),
      ]);
      renderValidationBadge(valLatest);
      renderLearningBadge(pendingPromos);
      const ticks = history.ticks || [];
      const latest = ticks.length ? ticks[ticks.length - 1] : null;
      // Playbook should show the live candidate: prefer the newest tick, but if it
      // carries no structure, fall back to the most recent tick that proposed one.
      let candidate = latest;
      if (!candidate || !candidate.selected_family) {
        for (let i = ticks.length - 1; i >= 0; i--) {
          if (ticks[i].selected_family) { candidate = ticks[i]; break; }
        }
      }

      renderMarketPill(market);
      renderTopbar(live, market, ticks);
      renderSignal(live);
      renderOpenPositions(live.paper);
      renderPlaybook(candidate, live);
      renderParallel(live, latest);
      renderPolicy(latest);
      const v2Tick = lastTickWithV2(ticks) || latest;
      renderPhysDensity(v2Tick, live);
      renderRanker(v2Tick, live);
      renderV2Playbook(candidate, live);
      renderRegime(live);
      renderReason(live, candidate);
      renderWhy(live);
      renderVol(live);
      renderTech(live);
      renderGexVariants(latest, report);
      renderDynamics(latest);
      // Before any trade has closed, /api/paper's equity is null (it's derived
      // from the last CLOSED trade's balance in SQL); the live broker snapshot
      // embedded in /api/live already has the correct starting/current equity.
      if (paper.equity == null && live.paper && live.paper.equity != null) {
        paper = { ...paper, equity: live.paper.equity };
      }
      renderPaper(paper);
      renderEdge(report);
      renderFunnel(report, ticks);
      renderPredict(report);
      renderSigCorr(report);
      renderReadiness(readiness);
      renderTimeline(history);
      staleNote(live, market);

      // PredictionBundle: prefer store lookup, fall back to journaled v2_fc_*.
      const forecastTick = v2Tick || latest;
      if (forecastTick && forecastTick.snapshot_id) {
        api("/api/predictions?snapshot_id=" + encodeURIComponent(forecastTick.snapshot_id))
          .then((payload) => renderForecast(payload, forecastTick, live))
          .catch(() => renderForecast(null, forecastTick, live));
      } else {
        renderForecast(null, forecastTick, live);
      }

      lastChartData = { ticks, live, market, trades };
      drawChart();
      drawQuadrant();
    } catch (e) {
      if (e.message !== "Unauthorized") console.warn("refresh", e);
    }
  }

  /* ---------------- full-screen mode ---------------- */
  function chartIsFullscreen() {
    const panel = $("chart-panel");
    return document.fullscreenElement === panel || panel.classList.contains("fs-fallback");
  }

  function toggleChartFullscreen() {
    const panel = $("chart-panel");
    if (chartIsFullscreen()) {
      if (document.fullscreenElement) document.exitFullscreen().catch(() => {});
      panel.classList.remove("fs-fallback");
      requestAnimationFrame(drawChart);
      return;
    }
    if (panel.requestFullscreen) {
      // fixed-overlay fallback if the browser refuses (e.g. iframe sandbox)
      panel.requestFullscreen().catch(() => {
        panel.classList.add("fs-fallback");
        requestAnimationFrame(drawChart);
      });
    } else {
      panel.classList.add("fs-fallback");   // iOS Safari: no element fullscreen
      requestAnimationFrame(drawChart);
    }
  }

  /* ---------------- zoom presets ---------------- */
  function setZoomPreset(z) {
    // keep the view centered while jumping between preset magnifications
    const center = CH_PADL + chartPlotW() / 2;
    if (z <= 1) { chartView.z = 1; chartView.pan = 0; drawChart(); }
    else chartZoomAt(center, z / chartView.z);
    updateZoomButtons();
  }

  function updateZoomButtons() {
    [["zoom-1x", 1], ["zoom-5x", 5], ["zoom-15x", 15]].forEach(([id, z]) => {
      const b = $(id);
      if (b) b.classList.toggle("active", Math.abs(chartView.z - z) < 0.25);
    });
  }

  /* ---------------- chart zoom / pan / tooltip wiring ---------------- */
  function initChartControls() {
    const cv = $("chart");
    const tip = $("chart-tip");
    if (!cv) return;

    $("zoom-in").addEventListener("click", () => { chartZoomAt(CH_PADL + chartPlotW() / 2, 1.5); updateZoomButtons(); });
    $("zoom-out").addEventListener("click", () => { chartZoomAt(CH_PADL + chartPlotW() / 2, 1 / 1.5); updateZoomButtons(); });
    $("zoom-reset").addEventListener("click", () => {
      chartView.z = 1; chartView.pan = 0; drawChart(); updateZoomButtons();
    });
    $("zoom-1x").addEventListener("click", () => setZoomPreset(1));
    $("zoom-5x").addEventListener("click", () => setZoomPreset(5));
    $("zoom-15x").addEventListener("click", () => setZoomPreset(15));
    $("chart-fs").addEventListener("click", toggleChartFullscreen);

    document.addEventListener("fullscreenchange", () => requestAnimationFrame(drawChart));
    document.addEventListener("keydown", (e) => {
      if (/^(INPUT|TEXTAREA|SELECT)$/.test((e.target || {}).tagName || "")) return;
      if (e.key === "f" || e.key === "F") toggleChartFullscreen();
      else if (e.key === "Escape" && $("chart-panel").classList.contains("fs-fallback")) {
        $("chart-panel").classList.remove("fs-fallback");
        requestAnimationFrame(drawChart);
      }
    });

    cv.addEventListener("wheel", (e) => {
      e.preventDefault();
      const x = e.offsetX;
      chartZoomAt(x, e.deltaY < 0 ? 1.2 : 1 / 1.2);
      updateZoomButtons();
    }, { passive: false });

    cv.addEventListener("dblclick", (e) => {
      chartZoomAt(e.offsetX, 2);              // double-click: zoom in at cursor
      updateZoomButtons();
    });

    // drag to pan (only meaningful when zoomed)
    let drag = null;
    cv.addEventListener("mousedown", (e) => {
      if (chartView.z <= 1) return;
      drag = { x0: e.clientX, pan0: chartView.pan };
      cv.style.cursor = "grabbing";
    });
    window.addEventListener("mousemove", (e) => {
      if (!drag) return;
      chartView.pan = drag.pan0 - (e.clientX - drag.x0);
      clampChartView();
      drawChart();
    });
    window.addEventListener("mouseup", () => { drag = null; cv.style.cursor = ""; });

    // touch: one finger pans, two fingers pinch-zoom
    let touch = null;
    const tdist = (t) => Math.hypot(t[0].clientX - t[1].clientX, t[0].clientY - t[1].clientY);
    cv.addEventListener("touchstart", (e) => {
      if (e.touches.length === 2) {
        touch = { mode: "pinch", d0: tdist(e.touches), z0: chartView.z,
                  cx: (e.touches[0].clientX + e.touches[1].clientX) / 2 - cv.getBoundingClientRect().left };
      } else if (e.touches.length === 1 && chartView.z > 1) {
        touch = { mode: "pan", x0: e.touches[0].clientX, pan0: chartView.pan };
      }
    }, { passive: true });
    cv.addEventListener("touchmove", (e) => {
      if (!touch) return;
      if (touch.mode === "pinch" && e.touches.length === 2) {
        e.preventDefault();
        const f = tdist(e.touches) / touch.d0;
        const target = Math.max(1, Math.min(40, touch.z0 * f));
        chartZoomAt(touch.cx, target / chartView.z);
      } else if (touch.mode === "pan" && e.touches.length === 1) {
        e.preventDefault();
        chartView.pan = touch.pan0 - (e.touches[0].clientX - touch.x0);
        clampChartView();
        drawChart();
      }
    }, { passive: false });
    cv.addEventListener("touchend", () => { touch = null; });

    // marker tooltips
    const showTip = (hit, xCss, yCss) => {
      tip.textContent = hit.label;
      tip.classList.remove("hidden");
      const wrap = cv.parentElement;
      const maxX = wrap.clientWidth - tip.offsetWidth - 4;
      tip.style.left = Math.max(4, Math.min(maxX, xCss + 10)) + "px";
      tip.style.top = Math.max(4, yCss - 30) + "px";
    };
    cv.addEventListener("mousemove", (e) => {
      if (drag) return;
      const x = e.offsetX, y = e.offsetY;
      let best = null, bd = 121;                  // 11px radius
      for (const h of chartHits) {
        const d = (h.x - x) ** 2 + (h.y - y) ** 2;
        if (d < bd) { bd = d; best = h; }
      }
      if (best) { showTip(best, best.x, best.y); cv.style.cursor = "pointer"; }
      else { tip.classList.add("hidden"); if (!drag) cv.style.cursor = chartView.z > 1 ? "grab" : ""; }
    });
    cv.addEventListener("mouseleave", () => tip.classList.add("hidden"));
    cv.addEventListener("touchstart", (e) => {          // tap a marker on mobile
      if (e.touches.length !== 1) return;
      const r = cv.getBoundingClientRect();
      const x = e.touches[0].clientX - r.left, y = e.touches[0].clientY - r.top;
      let best = null, bd = 400;                  // 20px touch radius
      for (const h of chartHits) {
        const d = (h.x - x) ** 2 + (h.y - y) ** 2;
        if (d < bd) { bd = d; best = h; }
      }
      if (best) { showTip(best, best.x, best.y); setTimeout(() => tip.classList.add("hidden"), 2500); }
    }, { passive: true });
  }

  /* ---------------- boot ---------------- */
  function boot() {
    showApp();
    $("tab-legacy").addEventListener("click", () => switchTab("legacy"));
    $("tab-v2").addEventListener("click", () => switchTab("v2"));
    $("tab-journal").addEventListener("click", () => switchTab("journal"));
    $("tab-validation").addEventListener("click", () => switchTab("validation"));
    $("tab-learning").addEventListener("click", () => switchTab("learning"));
    initValidationControls();
    initChartControls();
    loadMarketStatus();
    refresh();
    countdownTimer = setInterval(tickCountdown, 1000);
    refreshTimer = setInterval(refresh, REFRESH_MS);
  }

  let resizeRAF = null;
  window.addEventListener("resize", () => {
    if (resizeRAF) cancelAnimationFrame(resizeRAF);
    resizeRAF = requestAnimationFrame(() => { drawChart(); drawQuadrant(); });
  });

  async function init() {
    $("token-save").addEventListener("click", () => {
      const v = $("token-input").value.trim();
      if (v) { sessionStorage.setItem(TOKEN_KEY, v); boot(); }
    });
    $("token-input").addEventListener("keydown", (e) => { if (e.key === "Enter") $("token-save").click(); });

    // Vercel proxy: /api/health works without a client token (token injected server-side).
    try {
      const probe = await fetch("/api/health");
      if (probe.ok) { boot(); return; }
    } catch (_) { /* offline or misconfigured */ }

    if (!getToken()) { showAuth(); return; }
    boot();
  }

  init();
})();
