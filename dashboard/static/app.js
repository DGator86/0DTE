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
  let lastChartData = null; // {ticks, live, market} kept for resize redraws

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
    $("playbook-fam").textContent = t.selected_family || (live.doing && live.doing.structure) || "—";

    // The candidate journaled on a NO_TRADE tick is the measurement loop's
    // would-be pick (kept so settlement can score its hypothetical P&L).
    // It can be a degenerate penny structure — flag it so it never reads
    // as a trade recommendation.
    const isDiagnostic = !!t.selected_family && t.decision !== "TRADE" && !num(t.was_traded);
    $("playbook-diag").hidden = !isDiagnostic;

    const evCls = num(t.ev) > 0 ? "pos" : num(t.ev) < 0 ? "neg" : "";
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
    ];
    $("playbook-metrics").innerHTML = cards.join("");
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
      const detail = trade
        ? `${esc(fam)} <small>gate ${fmt(t.gate_score, 0)} · EV ${money(t.ev, 0)}</small>`
        : `<small>${esc((t.no_trade_reason || t.gex_regime || "no trade").replace(/_/g, " "))}</small>`;
      return `<div class="tl-item${trade ? " is-trade" : ""}">
        <span class="t">${etTime(t.ts)}</span>
        <span class="m">${detail} <small>· ${fmt(t.spot, 2)}</small></span>
        <span class="d ${trade ? "trade" : "no"}">${trade ? "TRADE" : "—"}</span>
      </div>`;
    }).join("");
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
    const projFrac = 0.72; // where "now" sits horizontally
    // map: historical [t0..tLast] -> [padL .. padL+plotW*projFrac], projection -> remainder
    const xNow = padL + plotW * projFrac;
    const X = (ts) => {
      if (ts <= tLast) {
        const f = tLast > t0 ? (ts - t0) / (tLast - t0) : 1;
        return padL + f * (xNow - padL);
      }
      const f = tEnd > tLast ? (ts - tLast) / (tEnd - tLast) : 0;
      return xNow + f * (padL + plotW - xNow);
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

    // --- time axis labels ---
    ctx.textAlign = "center";
    ctx.fillStyle = "#5b6a86";
    for (let i = 0; i <= 4; i++) {
      const ts = t0 + (i / 4) * (tEnd - t0);
      const x = X(Math.min(ts, tEnd));
      ctx.fillText(new Date(ts).toLocaleTimeString("en-US", { timeZone: "America/New_York", hour: "numeric", minute: "2-digit" }), x, H - 6);
    }

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
    const xEnd = padL + plotW;
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

    // --- trade markers ---
    pts.forEach((p) => {
      if (p.decision !== "TRADE") return;
      const x = X(p.ts), y = Y(p.spot);
      ctx.beginPath(); ctx.arc(x, y, 4, 0, Math.PI * 2);
      ctx.fillStyle = "#2ec785"; ctx.fill();
      ctx.strokeStyle = "#0a0e14"; ctx.lineWidth = 1.5; ctx.stroke();
    });

    // --- last price dot + tag ---
    const lp = pts[pts.length - 1];
    const lx = X(lp.ts), ly = Y(lp.spot);
    ctx.beginPath(); ctx.arc(lx, ly, 4.5, 0, Math.PI * 2);
    ctx.fillStyle = "#4aa8ff"; ctx.fill();
    ctx.strokeStyle = "#e6edf7"; ctx.lineWidth = 1.5; ctx.stroke();

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

  /* ---------------- trade journal tab ---------------- */
  let activeTab = "command";

  function switchTab(tab) {
    activeTab = tab;
    $("tab-command").classList.toggle("active", tab === "command");
    $("tab-journal").classList.toggle("active", tab === "journal");
    $("view-command").classList.toggle("hidden", tab !== "command");
    $("view-journal").classList.toggle("hidden", tab !== "journal");
    if (tab === "journal") refreshJournal();
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
          </div>
          <div class="tj-sub">${entryLogicLine(p.entry_ctx)}</div>
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
                      : t.exit_reason === "stop" ? "bad" : "";
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
            <div class="tj-sub">${exitLogicLine(t)}</div></td>
        <td class="mono">$${fmt(t.equity_after, 2)}</td>
      </tr>`;
    }).join("");
  }

  async function refreshJournal() {
    try {
      renderJournal(await api("/api/trades?limit=200"));
    } catch (e) {
      if (e.message !== "Unauthorized") console.warn("journal", e);
    }
  }

  /* ---------------- refresh loop ---------------- */
  async function refresh() {
    if (activeTab === "journal") refreshJournal();
    try {
      let [live, market, history, report, paper, readiness] = await Promise.all([
        api("/api/live"),
        api("/api/market-status"),
        api("/api/ticks?limit=200"),
        api("/api/report").catch(() => ({})),
        api("/api/paper").catch(() => ({})),
        api("/api/readiness").catch(() => ({})),
      ]);
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
      renderRegime(live);
      renderReason(live, candidate);
      renderWhy(live);
      renderVol(live);
      renderTech(live);
      // Before any trade has closed, /api/paper's equity is null (it's derived
      // from the last CLOSED trade's balance in SQL); the live broker snapshot
      // embedded in /api/live already has the correct starting/current equity.
      if (paper.equity == null && live.paper && live.paper.equity != null) {
        paper = { ...paper, equity: live.paper.equity };
      }
      renderPaper(paper);
      renderEdge(report);
      renderReadiness(readiness);
      renderTimeline(history);
      staleNote(live, market);

      lastChartData = { ticks, live, market };
      drawChart();
    } catch (e) {
      if (e.message !== "Unauthorized") console.warn("refresh", e);
    }
  }

  /* ---------------- boot ---------------- */
  function boot() {
    showApp();
    $("tab-command").addEventListener("click", () => switchTab("command"));
    $("tab-journal").addEventListener("click", () => switchTab("journal"));
    loadMarketStatus();
    refresh();
    countdownTimer = setInterval(tickCountdown, 1000);
    refreshTimer = setInterval(refresh, REFRESH_MS);
  }

  let resizeRAF = null;
  window.addEventListener("resize", () => {
    if (resizeRAF) cancelAnimationFrame(resizeRAF);
    resizeRAF = requestAnimationFrame(drawChart);
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
