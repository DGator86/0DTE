"""
notifier.py
===========
Human-readable trade ticket for 0DTE TRADE signals.

Backends (all optional except stdout):
  stdout    — always fires; a formatted ticket block to the terminal.
  file      — NOTIFY_FILE env var; appends one JSON line per ticket.
  email     — NOTIFY_SMTP_HOST / NOTIFY_SMTP_PORT / NOTIFY_SMTP_USER /
              NOTIFY_SMTP_PASS / NOTIFY_EMAIL_TO  (all from env, none hardcoded).
  ntfy push — NOTIFY_NTFY_TOPIC env var; POST to https://ntfy.sh/{topic}.
              Optional NOTIFY_NTFY_TOKEN for private topics.

SECURITY: no credentials are hardcoded. Set environment variables before use.

NOT financial advice.
"""
from __future__ import annotations

import dataclasses
import json
import logging
import math
import os
import smtplib
import ssl
import urllib.request
from dataclasses import asdict, dataclass
from email.mime.text import MIMEText
from typing import Optional

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Ticket                                                                        #
# --------------------------------------------------------------------------- #
@dataclass
class Ticket:
    ts: str                        # ISO-8601 timestamp
    session_date: str
    symbol: str

    # regime context
    dominant_regime: str
    exec_regime: str
    context_regime: str
    direction_bias: str
    size_mult: float               # from TradeIntent

    # structure
    family: str
    direction: str                 # "call" | "put" | "both" | "none"

    # legs, sorted by side
    short_calls: list              # list of floats (strikes), may be empty
    long_calls: list
    short_puts: list
    long_puts: list

    # risk metrics
    credit: float                  # positive = credit; negative = debit
    max_loss: float                # always positive
    ev: float
    ev_per_risk: float
    prob_profit: float
    gate_score: float
    theta_per_day: float
    contracts_per_1k: int          # floor(1000 / (max_loss * 100))

    @classmethod
    def from_tick_result(cls, result, symbol: str) -> Optional["Ticket"]:
        """Build from a TickResult. Returns None if the result is not a tradable TRADE."""
        dec = result.decision
        if dec is None or dec.decision != "TRADE" or not dec.gate_pass:
            return None
        c = dec.candidate
        if c is None:
            return None

        short_calls, long_calls, short_puts, long_puts = [], [], [], []
        for leg in c.legs:
            if leg.kind in ("C", "call"):
                (short_calls if leg.qty < 0 else long_calls).append(leg.strike)
            else:
                (short_puts if leg.qty < 0 else long_puts).append(leg.strike)

        ml = c.max_loss if c.max_loss and c.max_loss > 0 else None
        contracts = max(1, int(math.floor(1000.0 / (ml * 100)))) if ml else 1

        intent = result.intent
        regime = result.regime

        return cls(
            ts=result.ts.isoformat(),
            session_date=dec.session_date,
            symbol=symbol,
            dominant_regime=regime.dominant_regime,
            exec_regime=intent.exec_regime,
            context_regime=intent.context_regime,
            direction_bias=intent.direction_bias,
            size_mult=result.final_size_mult,
            family=c.family,
            direction=intent.decision.direction,
            short_calls=sorted(short_calls),
            long_calls=sorted(long_calls),
            short_puts=sorted(short_puts),
            long_puts=sorted(long_puts),
            credit=round(c.credit, 4),
            max_loss=round(c.max_loss, 4),
            ev=round(c.ev, 4),
            ev_per_risk=round(c.ev_per_risk, 4),
            prob_profit=round(c.prob_profit, 4),
            gate_score=round(dec.gate_score, 4),
            theta_per_day=round(c.theta, 4) if c.theta else 0.0,
            contracts_per_1k=contracts,
        )

    @classmethod
    def from_unified_decision(
            cls, result, symbol: str,
            candidate_lookup=None) -> Optional["Ticket"]:
        """
        Build a ticket from the authoritative unified decision.

        Uses authoritative_decision + the selected candidate (looked up from
        paper_intents / candidate_lookup), not the legacy TradeDecision.
        """
        auth = getattr(result, "authoritative_decision", None)
        if not isinstance(auth, dict):
            return None
        if str(auth.get("final_action") or "").upper() != "TRADE":
            return None

        cid = auth.get("selected_candidate_id")
        cand = None
        if candidate_lookup is not None and cid is not None:
            if callable(candidate_lookup):
                cand = candidate_lookup(cid)
            elif isinstance(candidate_lookup, dict):
                cand = candidate_lookup.get(cid)
        if cand is None:
            # Prefer the V3 paper intent candidate when present
            for intent in (getattr(result, "paper_intents", None) or []):
                if str(intent.get("track") or "").lower() != "v3":
                    continue
                if cid is None or str(intent.get("candidate_id") or "") == str(cid):
                    cand = intent.get("candidate")
                    if cand is not None:
                        break
        if cand is None and getattr(result, "decision", None) is not None:
            # Last resort: only if legacy candidate id matches authoritative
            legacy_c = getattr(result.decision, "candidate", None)
            legacy_id = (
                getattr(legacy_c, "candidate_id", None)
                or getattr(legacy_c, "v2_candidate_id", None)
            ) if legacy_c is not None else None
            if legacy_c is not None and (
                    cid is None or str(legacy_id or "") == str(cid)):
                # Only use legacy cand when authority selected it
                if str(getattr(result, "authority_source", "") or "") in (
                        "legacy", "legacy_fallback"):
                    cand = legacy_c
        if cand is None:
            return None

        legs = getattr(cand, "legs", None) or ()
        short_calls, long_calls, short_puts, long_puts = [], [], [], []
        for leg in legs:
            kind = getattr(leg, "kind", None) or (
                leg.get("kind") if isinstance(leg, dict) else None)
            qty = getattr(leg, "qty", None)
            if qty is None and isinstance(leg, dict):
                qty = leg.get("qty")
            strike = getattr(leg, "strike", None)
            if strike is None and isinstance(leg, dict):
                strike = leg.get("strike")
            if kind in ("C", "call"):
                (short_calls if (qty or 0) < 0 else long_calls).append(float(strike))
            else:
                (short_puts if (qty or 0) < 0 else long_puts).append(float(strike))

        ml = float(getattr(cand, "max_loss", 0) or 0)
        contracts = max(1, int(math.floor(1000.0 / (ml * 100)))) if ml > 0 else 1
        intent = result.intent
        regime = result.regime
        size = float(auth.get("size_mult")
                     if auth.get("size_mult") is not None
                     else getattr(result, "final_size_mult", 1.0) or 1.0)
        family = (
            auth.get("structure")
            or getattr(cand, "family", None)
            or "unknown"
        )
        direction = (
            auth.get("direction")
            or getattr(getattr(intent, "decision", None), "direction", None)
            or "both"
        )
        session_date = ""
        if hasattr(result, "ts") and result.ts is not None:
            try:
                session_date = result.ts.date().isoformat()
            except Exception:
                session_date = str(getattr(result.ts, "isoformat", lambda: "")())
        dec = getattr(result, "decision", None)
        if dec is not None and getattr(dec, "session_date", None):
            session_date = str(dec.session_date)

        return cls(
            ts=result.ts.isoformat(),
            session_date=session_date,
            symbol=symbol,
            dominant_regime=getattr(regime, "dominant_regime", "") or "",
            exec_regime=getattr(intent, "exec_regime", "") or "",
            context_regime=getattr(intent, "context_regime", "") or "",
            direction_bias=getattr(intent, "direction_bias", "") or "",
            size_mult=size,
            family=str(family),
            direction=str(direction),
            short_calls=sorted(short_calls),
            long_calls=sorted(long_calls),
            short_puts=sorted(short_puts),
            long_puts=sorted(long_puts),
            credit=round(float(getattr(cand, "credit", 0) or 0), 4),
            max_loss=round(ml, 4),
            ev=round(float(getattr(cand, "ev", 0) or 0), 4),
            ev_per_risk=round(float(getattr(cand, "ev_per_risk", 0) or 0), 4),
            prob_profit=round(float(getattr(cand, "prob_profit", 0) or 0), 4),
            gate_score=0.0,
            theta_per_day=round(float(getattr(cand, "theta", 0) or 0), 4),
            contracts_per_1k=contracts,
        )


# --------------------------------------------------------------------------- #
# Formatter                                                                     #
# --------------------------------------------------------------------------- #
def _leg_line(label: str, strikes: list) -> str:
    if not strikes:
        return ""
    return f"  {label:<18} {' / '.join(f'${s:.1f}' for s in strikes)}\n"


def format_ticket(t: Ticket) -> str:
    side = "CREDIT" if t.credit >= 0 else "DEBIT"
    abs_credit = abs(t.credit)
    lines = [
        "=" * 56,
        f"  0DTE TRADE SIGNAL  {t.ts[11:19]} ET  {t.symbol}",
        "=" * 56,
        f"  Structure:         {t.family.upper()}  ({t.direction})",
    ]
    for lbl, strikes in (
        ("Short calls:", t.short_calls),
        ("Long calls:", t.long_calls),
        ("Short puts:", t.short_puts),
        ("Long puts:", t.long_puts),
    ):
        line = _leg_line(lbl, strikes).rstrip("\n")
        if line:
            lines.append(line)

    lines += [
        "-" * 56,
        f"  {side:<18}     ${abs_credit:.2f} / contract",
        f"  Max loss:          ${t.max_loss:.2f} / contract",
        f"  EV:                ${t.ev:.2f}  (EV/risk {t.ev_per_risk:.3f})",
        f"  P(profit):         {t.prob_profit * 100:.1f}%",
        f"  Theta/day:         ${t.theta_per_day:.2f}",
        "-" * 56,
        f"  Gate score:        {t.gate_score:.3f}",
        f"  Size multiplier:   x{t.size_mult:.2f}",
        f"  Contracts @ $1k:   {t.contracts_per_1k}",
        "-" * 56,
        f"  Regime:  {t.dominant_regime}  |  exec={t.exec_regime}  ctx={t.context_regime}",
        f"  Bias:    {t.direction_bias}",
        "=" * 56,
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Notifier                                                                      #
# --------------------------------------------------------------------------- #
class Notifier:
    """
    Dispatch trade tickets to configured backends.

    All backend configuration is read from environment variables at send() time
    so the process can pick up changes without restarting.

    Dedup: a persistent regime re-emits the same TRADE ticket every tick, which
    used to mean one phone push per minute for hours. A ticket is now sent only
    when its SIGNATURE (family + legs + direction) differs from the last one
    sent, or when the resend cooldown has elapsed (a still-valid signal
    re-pings occasionally rather than never). Override the cooldown with
    NOTIFY_COOLDOWN_MIN; 0 restores the old fire-every-tick behavior.
    """

    def __init__(self) -> None:
        self._last_sig: Optional[tuple] = None
        self._last_sent_ts: Optional[float] = None

    @staticmethod
    def _signature(t: Ticket) -> tuple:
        return (t.family, t.direction, tuple(t.short_calls), tuple(t.long_calls),
                tuple(t.short_puts), tuple(t.long_puts))

    def _should_send(self, ticket: Ticket) -> bool:
        cooldown_min = float(os.environ.get("NOTIFY_COOLDOWN_MIN", "15"))
        if cooldown_min <= 0:
            return True
        sig = self._signature(ticket)
        import time as _time
        now = _time.monotonic()
        if sig != self._last_sig or self._last_sent_ts is None \
                or (now - self._last_sent_ts) >= cooldown_min * 60.0:
            self._last_sig = sig
            self._last_sent_ts = now
            return True
        return False

    def send(self, ticket: Optional[Ticket]) -> None:
        if ticket is None:
            return
        if not self._should_send(ticket):
            return
        text = format_ticket(ticket)
        self._stdout(text)
        self._file(ticket)
        self._email(ticket, text)
        self._ntfy(ticket, text)

    def send_text(self, title: str, body: str, tags: str = "bell") -> None:
        """Push a plain-text notification (used for paper entries/exits). Goes to
        stdout always and to ntfy when NOTIFY_NTFY_TOPIC is set."""
        print(f"{title}: {body}", flush=True)
        topic = os.environ.get("NOTIFY_NTFY_TOPIC", "")
        if not topic:
            return
        token = os.environ.get("NOTIFY_NTFY_TOKEN", "")
        req = urllib.request.Request(
            f"https://ntfy.sh/{topic}", data=body.encode("utf-8"), method="POST")
        req.add_header("Title", title)
        req.add_header("Priority", "default")
        req.add_header("Tags", tags)
        req.add_header("Content-Type", "text/plain")
        if token:
            req.add_header("Authorization", f"Bearer {token}")
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:  # noqa: S310
                if resp.status >= 400:
                    log.warning("notifier: ntfy returned HTTP %d", resp.status)
        except Exception as exc:
            log.warning("notifier: ntfy text backend failed: %s", exc)

    # -- backends ------------------------------------------------------------

    @staticmethod
    def _stdout(text: str) -> None:
        print(text, flush=True)

    @staticmethod
    def _file(ticket: Ticket) -> None:
        path = os.environ.get("NOTIFY_FILE", "")
        if not path:
            return
        try:
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(asdict(ticket)) + "\n")
        except Exception as exc:
            log.warning("notifier: file backend failed: %s", exc)

    @staticmethod
    def _email(ticket: Ticket, text: str) -> None:
        host = os.environ.get("NOTIFY_SMTP_HOST", "")
        to   = os.environ.get("NOTIFY_EMAIL_TO", "")
        if not host or not to:
            return
        port = int(os.environ.get("NOTIFY_SMTP_PORT", "587"))
        user = os.environ.get("NOTIFY_SMTP_USER", "")
        pw   = os.environ.get("NOTIFY_SMTP_PASS", "")
        subject = (
            f"0DTE {ticket.symbol} {ticket.family.upper()} "
            f"{'CR' if ticket.credit >= 0 else 'DB'} "
            f"${abs(ticket.credit):.2f}  {ticket.ts[11:16]} ET"
        )
        msg = MIMEText(text, "plain")
        msg["Subject"] = subject
        msg["From"] = user or "0dte-shadow@localhost"
        msg["To"] = to
        try:
            ctx = ssl.create_default_context()
            with smtplib.SMTP(host, port) as smtp:
                smtp.ehlo()
                smtp.starttls(context=ctx)
                if user and pw:
                    smtp.login(user, pw)
                smtp.sendmail(msg["From"], [to], msg.as_string())
        except Exception as exc:
            log.warning("notifier: email backend failed: %s", exc)

    @staticmethod
    def _ntfy(ticket: Ticket, text: str) -> None:
        topic = os.environ.get("NOTIFY_NTFY_TOPIC", "")
        if not topic:
            return
        token = os.environ.get("NOTIFY_NTFY_TOKEN", "")
        url = f"https://ntfy.sh/{topic}"
        side = "CR" if ticket.credit >= 0 else "DB"
        title = (
            f"0DTE {ticket.symbol} {ticket.family.upper()} "
            f"{side} ${abs(ticket.credit):.2f}"
        )
        req = urllib.request.Request(
            url,
            data=text.encode("utf-8"),
            method="POST",
        )
        req.add_header("Title", title)
        req.add_header("Priority", "high")
        req.add_header("Tags", "chart_with_upwards_trend")
        req.add_header("Content-Type", "text/plain")
        if token:
            req.add_header("Authorization", f"Bearer {token}")
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:  # noqa: S310
                if resp.status >= 400:
                    log.warning("notifier: ntfy returned HTTP %d", resp.status)
        except Exception as exc:
            log.warning("notifier: ntfy backend failed: %s", exc)


# --------------------------------------------------------------------------- #
# Smoke test                                                                    #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import datetime as dt
    from zoneinfo import ZoneInfo

    ET = ZoneInfo("America/New_York")

    t = Ticket(
        ts=dt.datetime(2026, 6, 26, 10, 32, tzinfo=ET).isoformat(),
        session_date="2026-06-26",
        symbol="SPY",
        dominant_regime="compression",
        exec_regime="compression",
        context_regime="compression",
        direction_bias="neutral",
        size_mult=1.0,
        family="iron_condor",
        direction="both",
        short_calls=[602.0], long_calls=[604.0],
        short_puts=[597.0],  long_puts=[595.0],
        credit=1.42,
        max_loss=0.58,
        ev=0.31,
        ev_per_risk=0.534,
        prob_profit=0.72,
        gate_score=0.81,
        theta_per_day=0.18,
        contracts_per_1k=17,
    )
    print(format_ticket(t))
    Notifier().send(t)
