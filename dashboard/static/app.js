(function () {
  "use strict";

  const TOKEN_KEY = "zerodte_dashboard_token";
  let marketAnchor = null;
  let countdownTimer = null;
  let refreshTimer = null;

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

  function showAuth() {
    document.getElementById("auth-gate").classList.remove("hidden");
    document.getElementById("app").classList.add("hidden");
  }

  function showApp() {
    document.getElementById("auth-gate").classList.add("hidden");
    document.getElementById("app").classList.remove("hidden");
  }

  function fmtDuration(totalSec) {
    const s = Math.max(0, Math.floor(totalSec));
    const d = Math.floor(s / 86400);
    const h = Math.floor((s % 86400) / 3600);
    const m = Math.floor((s % 3600) / 60);
    const sec = s % 60;
    const parts = [];
    if (d) parts.push(d + "d");
    if (h || d) parts.push(h + "h");
    parts.push(m + "m");
    parts.push(String(sec).padStart(2, "0") + "s");
    return parts.join(" ");
  }

  function fmtEt(iso) {
    if (!iso) return "—";
    const d = new Date(iso);
    return d.toLocaleString("en-US", {
      timeZone: "America/New_York",
      weekday: "short",
      month: "short",
      day: "numeric",
      hour: "numeric",
      minute: "2-digit",
      timeZoneName: "short",
    });
  }

  function updateMarketBanner(status) {
    const banner = document.getElementById("market-banner");
    const label = document.getElementById("market-label");
    const countdown = document.getElementById("market-countdown");
    const subtitle = document.getElementById("market-subtitle");

    banner.classList.toggle("open", status.is_open);
    banner.classList.toggle("closed", !status.is_open);
    label.textContent = status.is_open ? status.label_open : status.label_closed;

    marketAnchor = {
      is_open: status.is_open,
      targetIso: status.is_open ? status.next_close : status.next_open,
      fetchedAt: Date.now(),
      secondsRemaining: status.is_open
        ? status.seconds_until_close
        : status.seconds_until_open,
      session_type: status.session_type,
      next_open: status.next_open,
    };

    tickCountdown();

    if (status.is_open) {
      subtitle.textContent =
        (status.session_type === "early_close" ? "Early close" : "Regular session") +
        " · NYSE";
    } else {
      subtitle.textContent = "Next session: " + fmtEt(status.next_open);
    }
  }

  function tickCountdown() {
    if (!marketAnchor) return;
    const elapsed = (Date.now() - marketAnchor.fetchedAt) / 1000;
    const remaining = Math.max(0, (marketAnchor.secondsRemaining || 0) - elapsed);
    const el = document.getElementById("market-countdown");
    if (marketAnchor.is_open) {
      el.textContent = "Closes in " + fmtDuration(remaining);
    } else {
      el.textContent = "Opens in " + fmtDuration(remaining);
    }
    if (remaining <= 0 && countdownTimer) {
      loadMarketStatus();
    }
  }

  async function loadMarketStatus() {
    try {
      const status = await api("/api/market-status");
      updateMarketBanner(status);
    } catch (e) {
      console.warn("market-status", e);
    }
  }

  function kvHtml(pairs) {
    return (
      "<dl class='kv'>" +
      pairs
        .map(
          ([k, v]) =>
            "<dt>" + escapeHtml(k) + "</dt><dd>" + escapeHtml(String(v ?? "—")) + "</dd>"
        )
        .join("") +
      "</dl>"
    );
  }

  function escapeHtml(s) {
    return s
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function badge(pass) {
    if (pass === true) return "<span class='badge pass'>PASS</span>";
    if (pass === false) return "<span class='badge fail'>FAIL</span>";
    return "<span class='badge neutral'>—</span>";
  }

  function staleness(ts) {
    if (!ts) return "";
    const age = (Date.now() - new Date(ts).getTime()) / 1000;
    if (age < 120) return "";
    const mins = Math.floor(age / 60);
    return "<p class='stale'>Last updated " + mins + " min ago — pipeline may be idle off-hours</p>";
  }

  function renderNow(live, market) {
    const d = live.doing || {};
    const open = market && market.is_open;
    let html = staleness(live.ts);
    if (!open && live.ts) {
      html += "<p class='stale'>Market closed — showing last tick snapshot</p>";
    }
    html += "<div class='card'><h2>Current action</h2>" + kvHtml([
      ["Regime", d.dominant_regime],
      ["Engine", d.permitted_engine],
      ["Structure", d.structure],
      ["Direction", d.direction],
      ["Conviction", d.conviction],
      ["Gate", badge(d.gate_pass)],
      ["Gate score", d.gate_score != null ? d.gate_score.toFixed(1) : "—"],
      ["Decision", d.decision],
      ["Size mult", d.final_size_mult != null ? "×" + d.final_size_mult.toFixed(2) : "—"],
      ["Stand down", d.stand_down ? "yes" : "no"],
    ]) + "</div>";

    html += "<div class='card'><h2>Feed</h2>" + kvHtml([
      ["Data source", live.feed_source || "—"],
      ["Chain data", live.chain_available ? "available" : "unavailable"],
      ["Tick time", live.ts ? fmtEt(live.ts) : "—"],
    ]) + "</div>";

    const paper = live.paper || {};
    if (paper.trades != null) {
      html += "<div class='card'><h2>Paper account (simulated)</h2>" + kvHtml([
        ["Equity", paper.equity != null ? "$" + paper.equity : "—"],
        ["Closed trades", paper.trades],
        ["Win rate", paper.win_rate != null ? (paper.win_rate * 100).toFixed(1) + "%" : "—"],
        ["Total P&L", paper.total_pnl != null ? "$" + paper.total_pnl : "—"],
      ]) + "<p style='color:var(--muted);font-size:0.8rem;margin:0.5rem 0 0'>Simulated fills only — not actionable</p></div>";
    }
    document.getElementById("panel-now").innerHTML = html;
  }

  function renderInputs(live) {
    const inp = live.inputs || {};
    const pairs = Object.entries(inp).map(([k, v]) => [
      k.replace(/_/g, " "),
      typeof v === "number" ? (Math.abs(v) > 1e6 ? v.toExponential(2) : Number(v.toFixed(4))) : v,
    ]);
    let html = "<div class='card'><h2>Market snapshot</h2>";
    if (!pairs.length) {
      html += "<p class='empty'>No market inputs yet</p>";
    } else {
      html += kvHtml(pairs);
    }
    html += "</div><div class='card'><h2>Data feed</h2>" + kvHtml([
      ["Provider", live.feed_source || "—"],
      ["Option chain", live.chain_available ? "yes" : "no"],
    ]) + "</div>";
    document.getElementById("panel-inputs").innerHTML = html;
  }

  function renderWhy(live) {
    const w = live.why || {};
    const conf = w.regime_confidences || {};
    const confPairs = Object.entries(conf)
      .sort((a, b) => b[1] - a[1])
      .map(([k, v]) => [k, v + "%"]);

    let html = "<div class='card'><h2>Regime reasoning</h2>" + kvHtml([
      ["Matrix cell", (w.matrix_cell || []).join(" × ")],
      ["Information gain", w.global_information_gain],
      ["Stand-down reason", w.stand_down_reason || "—"],
      ["Intent note", w.intent_note || "—"],
      ["Capture", w.capture || "—"],
      ["Strike rule", w.strike_rule || "—"],
      ["No-trade reason", w.no_trade_reason || "—"],
    ]) + "</div>";

    if (confPairs.length) {
      html += "<div class='card'><h2>Regime confidences</h2>" + kvHtml(confPairs) + "</div>";
    }

    const chips = (label, items) => {
      if (!items || !items.length) return "";
      return (
        "<div class='card'><h2>" + label + "</h2><div class='list-chips'>" +
        items.map((x) => "<span class='chip'>" + escapeHtml(String(x)) + "</span>").join("") +
        "</div></div>"
      );
    };
    html += chips("Dealer vetoes", w.dealer_vetoes);
    html += chips("Gate failures", w.gate_failed);
    html += chips("Selector vetoes", w.selector_vetoes);
    html += chips("Risk vetoes", w.risk_vetoes);

    document.getElementById("panel-why").innerHTML = html || "<p class='empty'>No reasoning data yet</p>";
  }

  function renderHistory(data) {
    const ticks = data.ticks || [];
    if (!ticks.length) {
      document.getElementById("panel-history").innerHTML =
        "<p class='empty'>No ticks for " + escapeHtml(data.session_date || "today") + "</p>";
      return;
    }
    let html = "<div class='card'><h2>" + escapeHtml(data.session_date) + "</h2>";
    ticks.slice().reverse().forEach((t) => {
      const time = t.ts ? new Date(t.ts).toLocaleTimeString("en-US", { timeZone: "America/New_York" }) : "—";
      const trade = t.decision === "TRADE";
      html +=
        "<div class='timeline-item" + (trade ? " trade" : "") + "' data-id='" + t.id + "'>" +
        "<div class='timeline-time'>" + escapeHtml(time) + "</div>" +
        "<div class='timeline-summary'>" +
        escapeHtml([t.gex_regime, t.selected_family || "—", t.decision, "gate=" + (t.gate_pass ? "PASS" : "FAIL")].join(" · ")) +
        "</div>" +
        "<div class='timeline-detail'>" +
        kvHtml([
          ["Spot", t.spot],
          ["Gate score", t.gate_score],
          ["EV", t.ev],
          ["No-trade reason", t.no_trade_reason || "—"],
          ["Gate failed", Array.isArray(t.gate_failed) ? t.gate_failed.join(", ") : t.gate_failed],
          ["Vetoes", Array.isArray(t.veto_reasons) ? t.veto_reasons.join(", ") : t.veto_reasons],
        ]) +
        "</div></div>";
    });
    html += "</div>";
    document.getElementById("panel-history").innerHTML = html;

    document.querySelectorAll(".timeline-item").forEach((el) => {
      el.addEventListener("click", () => el.classList.toggle("expanded"));
    });
  }

  function renderStats(report, paper) {
    const eff = report.gate_effectiveness || {};
    const taken = eff.trades_taken || {};
    const blocked = eff.blocked_by_gate || {};
    let html = "<div class='card'><h2>Gate effectiveness</h2>" + kvHtml([
      ["Trades taken (n)", taken.n],
      ["Taken mean P&L", taken.mean],
      ["Blocked by gate (n)", blocked.n],
      ["Blocked mean P&L", blocked.mean],
      ["Verdict", eff.verdict],
    ]) + "</div>";

    const corr = report.component_correlations || {};
    const corrPairs = Object.entries(corr).filter(([k]) => k !== "n" && k !== "note");
    if (corrPairs.length) {
      html += "<div class='card'><h2>Score correlations</h2>" + kvHtml(corrPairs) + "</div>";
    }

    if (paper && paper.trades != null) {
      html += "<div class='card'><h2>Paper trading (simulated)</h2>" + kvHtml([
        ["Trades", paper.trades],
        ["Win rate", (paper.win_rate * 100).toFixed(1) + "%"],
        ["Total P&L", "$" + paper.total_pnl],
        ["Equity", "$" + paper.equity],
        ["Max drawdown note", "see full report CLI"],
      ]) + "</div>";
    }
    document.getElementById("panel-stats").innerHTML = html;
  }

  async function refresh() {
    try {
      const [live, market, history, report, paper] = await Promise.all([
        api("/api/live"),
        api("/api/market-status"),
        api("/api/ticks?limit=100"),
        api("/api/report"),
        api("/api/paper"),
      ]);
      updateMarketBanner(market);
      renderNow(live, market);
      renderInputs(live);
      renderWhy(live);
      renderHistory(history);
      renderStats(report, paper);
    } catch (e) {
      if (e.message !== "Unauthorized") console.warn("refresh", e);
    }
  }

  function setupTabs() {
    document.querySelectorAll(".tab").forEach((btn) => {
      btn.addEventListener("click", () => {
        document.querySelectorAll(".tab").forEach((b) => b.classList.remove("active"));
        document.querySelectorAll(".panel").forEach((p) => p.classList.remove("active"));
        btn.classList.add("active");
        document.getElementById("panel-" + btn.dataset.tab).classList.add("active");
      });
    });
  }

  async function init() {
    setupTabs();
    document.getElementById("token-save").addEventListener("click", () => {
      const v = document.getElementById("token-input").value.trim();
      if (v) {
        sessionStorage.setItem(TOKEN_KEY, v);
        boot();
      }
    });

    // Vercel proxy: /api/health works without a client token (token is server-side).
    try {
      const probe = await fetch("/api/health");
      if (probe.ok) {
        boot();
        return;
      }
    } catch (_) {
      /* offline or misconfigured */
    }

    if (!getToken()) {
      showAuth();
      return;
    }
    boot();
  }

  function boot() {
    showApp();
    loadMarketStatus();
    refresh();
    countdownTimer = setInterval(tickCountdown, 1000);
    refreshTimer = setInterval(refresh, 15000);
  }

  init();
})();
