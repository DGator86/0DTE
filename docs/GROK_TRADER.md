# Grok 4.5 autonomous paper trader

This integration adds a fourth, isolated paper account named `grok` to the
existing Legacy/V2/V3 shadow pipeline. It is deliberately incapable of routing
a live broker order.

## Data boundary

Grok can inspect:

- the current raw market snapshot, bars, chain, option rows, and weekly rows;
- decision-blinded Legacy analytical signals;
- decision-blinded V2 forecasts, distributions, and ranking diagnostics;
- decision-blinded V3 scenario and utility diagnostics;
- the Grok paper account, its own positions, and its persisted intraday memory.

The evidence serializer removes final policy/action fields including engine
trade decisions, selected candidates, selected structures, selected strikes,
paper intents, policy source, and final size. The local test suite contains a
leakage assertion.

Large inputs are not copied into every prompt. Grok receives a terminal summary
and can page through raw rows or request a chain slice with local function calls.
No hosted web or X-search tool is enabled.

## Authority

Grok may propose these same-day, defined-risk families:

- put credit spread;
- call credit spread;
- iron condor;
- long call spread;
- long put spread.

The deterministic firewall independently checks the symbol, expiration, entry
window, chain availability, quote age when available, leg pattern, current
strikes, liquidity, limit price, defined maximum loss, risk fraction, current
position count, and cost state. The existing `PaperBroker` then reprices the
approved candidate, simulates slippage, sizes the position, marks it, and keeps
its stop/target/trailing/RAS/EOD protections active. The firewall also validates
the model's mandatory exit time, and the coordinator closes the Grok position
deterministically when that time arrives.

Grok may also close its own paper position. It cannot alter Legacy/V2/V3
positions, increase an existing position, change risk limits, or access a live
execution credential.

## Safe rollout

The committed example configuration is off by default:

```bash
GROK_ENABLED=0
GROK_ORDER_SUBMISSION_ENABLED=0
GROK_PAPER_ONLY=1
```

Recommended activation sequence:

1. Add `XAI_API_KEY` to `/etc/zerodte/zerodte.env`.
2. Set `GROK_ENABLED=1` and leave submission disabled for read-only shadow
   decisions.
3. Review `journalctl -u zerodte-shadow` and the `grok_audit.sqlite` records.
4. Set `GROK_ORDER_SUBMISSION_ENABLED=1` to permit validated **paper** entries.

The model uses `grok-4.5` with `reasoning.effort=high`. The default cadence is
five minutes when flat and 90 seconds when a Grok position is open, with faster
event-triggered reviews for regime changes, gamma-flip crossings, and material
spot moves.

## Cost controls

Defaults:

```bash
GROK_DAILY_SOFT_CAP_USD=5.00
GROK_DAILY_HARD_CAP_USD=8.00
GROK_MONTHLY_HARD_CAP_USD=170.00
GROK_MAX_DAILY_CYCLES=100
```

At the soft cap, new entries are disabled but the model may still reduce risk.
At a hard daily/monthly/cycle lockout, model calls stop and the deterministic
paper-broker protections remain active.

Each cycle records trigger, model, response ID, latency, input/cached/output/
reasoning tokens, estimated cost, status, and action result. Pricing defaults to
$2/M input and $6/M output and can be updated through environment variables.

## Restart behavior

The legacy broker persists closed trades but holds open positions in memory.
This integration persists the Grok track's open position in
`grok_audit.sqlite`, restores it only during the same trading session, and
re-registers it with the existing position monitor after a service restart.
Stale prior-session 0DTE positions are never restored.
