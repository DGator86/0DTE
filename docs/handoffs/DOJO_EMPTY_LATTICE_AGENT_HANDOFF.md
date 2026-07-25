# Handoff: Land Dojo empty-lattice fix + make Dojo run for real

## Goal

Dojo should **keep running on schedule**, but **refuse the expensive universe
lattice** when there is no recorded tape (no more ~74‑minute no-ops). When tape
exists, it should **generate and score**. Also get market recording working so
tape accumulates, and ship the dashboard display fix.

## Context (read this)

- **0DTE** (`DGator86/0DTE`) is being retired; **Dojo / AI ownership is
  SPY-DER** (`DGator86/SPY-DER`).
- Saturday `spy-der-dojo-weekly` burned ~74 min: 192 universes, ~40k snapshots,
  **0 scored**, empty report. Cause: lattice ran with zero sessions;
  `generate_result()` outcomes were discarded; synthetic `realized_pnl` was
  null.
- Fix is **already written and unit-tested**, shipped as a patch on **0DTE
  `main`** (merged PR #156). It is **not yet on SPY-DER `main` or the VPS**.
- Live VPS (via `https://0-dte-kappa.vercel.app/api/system`):
  `market=never_seen`, `feed=no_recordings`, SPY-DER deploy still `8fd7267`.
  Old empty Dojo report still served.
- Previous cloud agent **could not push to SPY-DER** (403). You need write
  access to SPY-DER.

## Patch location

On `DGator86/0DTE` `main`:

- `patches/spy-der-dojo-skip-empty-lattice.patch`
- `patches/README-spy-der-dojo-fix.md`

Behavior after apply:

| Situation | Universe phase |
|---|---|
| No recorded tape (default) | `skipped`, flag `universe_skipped_no_tape`, **no lattice** |
| `--force-universe` | Runs lattice and scores (terminal P&L) |
| Tape present | Runs + scores |

---

## Task 1 — Apply patch to SPY-DER and merge to main (do this first)

```bash
git clone https://github.com/DGator86/SPY-DER.git && cd SPY-DER
git clone --depth 1 https://github.com/DGator86/0DTE.git /tmp/0DTE

git checkout main && git pull origin main
git checkout -b cursor/dojo-skip-empty-lattice-d52d

git am /tmp/0DTE/patches/spy-der-dojo-skip-empty-lattice.patch
# If am fails:
#   git apply /tmp/0DTE/patches/spy-der-dojo-skip-empty-lattice.patch
#   git add -A
#   git commit -m "Refuse empty-tape universe lattice; wire synthetic outcomes for scoring"

python3 -m pip install -e '.[dev]' -q
python3 -m pytest tests/unit/test_dojo_ownership.py \
  tests/unit/test_dojo_phase4.py \
  tests/unit/test_synthetic.py \
  tests/unit/test_ai_market_hours_gate.py -q --no-cov

git push -u origin cursor/dojo-skip-empty-lattice-d52d
# Open PR → merge to main
```

**Verify on GitHub:** `src/spy_der/dojo/config.py` contains `force_universe`;
`runner.py` contains `refusing universe lattice` / `universe_skipped_no_tape`.

---

## Task 2 — Confirm VPS pulled the fix

SPY-DER self-updates via `spy-der-update.timer` (~every 2 min). After main has
the commit:

```bash
# On VPS (SSH or GitHub Actions "VPS Ops" if available)
sudo systemctl start spy-der-update.service
# wait, then:
git -C /opt/spy-der rev-parse --short HEAD   # should NOT be 8fd7267
rg -n "force_universe|universe_skipped_no_tape" /opt/spy-der/src/spy_der/dojo/
```

Optional smoke (should finish in seconds, not an hour):

```bash
sudo -u spy-der bash -c 'set -a; . /etc/spy-der/spy-der.env; set +a; \
  /opt/spy-der/venv/bin/spy-der dojo \
    --reports-dir /var/lib/spy-der/reports/dojo \
    --configs-dir /var/lib/spy-der/configs \
    --experience-dir /var/lib/spy-der/inbox/experience \
    --full-lattice --days 8 --generations 2 --trials 25'
# Expect: universe skipped / universe_skipped_no_tape while tape is empty
# Confirm latest.json flags include universe_skipped_no_tape
```

**Do not** leave `--force-universe` on the systemd weekly unit.

---

## Task 3 — Keep Dojo timers enabled

```bash
systemctl list-timers 'spy-der-dojo-*'
# Ensure daily / recent / weekly timers are enabled.
# If weekly was disabled as a stopgap:
sudo systemctl enable --now spy-der-dojo-weekly.timer
```

Desired end state: timers fire; empty tape → fast skip; with tape → real
scored run.

---

## Task 4 — Fix market recording (otherwise Dojo will always skip lattice)

Live: `spy-der-market` has **never** published a heartbeat; feed has **no
recordings**.

```bash
systemctl status spy-der-market.service spy-der-engine.service spy-der-agent.service --no-pager
journalctl -u spy-der-market -n 100 --no-pager
# Credentials live in /etc/spy-der/spy-der.env (TRADIER_ACCESS_TOKEN / MASSIVE_API_KEY).
# Do not print secrets.
sudo systemctl enable --now spy-der-market.service
```

Success criteria after market open (Mon 09:30 ET):

- `https://0-dte-kappa.vercel.app/api/system` → market `ok`/`late`, feed
  `recording` with ticks > 0
- Files under `/var/lib/spy-der/market/*.jsonl` growing
- Experience inbox / 0DTE→SPY-DER packets accumulating if that path is wired

Until ≥ `min_sessions` (default 3) exist, Dojo correctly refuses the lattice —
that is success, not a bug.

---

## Task 5 — Merge 0DTE dashboard display PR

- Open PR: https://github.com/DGator86/0DTE/pull/155
- Branch: `cursor/dojo-dashboard-display-d52d`
- Makes the Vercel Dojo tab show thin/skipped runs clearly (coverage.cells
  unwrap, insufficient_data banner). Merge + let Vercel redeploy.

---

## Task 6 — Acceptance checklist

- [ ] SPY-DER `main` has empty-tape lattice gate + outcome wiring
- [ ] VPS `/opt/spy-der` on that commit
- [ ] Manual dojo without tape exits quickly with `universe_skipped_no_tape`
- [ ] `spy-der-dojo-{daily,recent,weekly}.timer` enabled
- [ ] `spy-der-market` healthy; recordings appear on a trading day
- [ ] After ≥3 sessions, a dojo run shows `n_scored_universes > 0` (or clear
      scored authorities), not a silent generate-only no-op
- [ ] 0DTE PR #155 merged; Dojo tab on `0-dte-kappa.vercel.app` shows the
      report/skip clearly

---

## Out of scope / do not

- Do not add Dojo/Grok logic back into 0DTE (`zerodte/**`, `integrations/**`
  are delete-at-cutover).
- Do not enable `--force-universe` on production timers.
- Do not disable weekly timer permanently once the patch is live.
- Do not print `/etc/spy-der/spy-der.env` secrets.

## Key URLs

- Patch PR (merged): https://github.com/DGator86/0DTE/pull/156
- Dashboard PR (open): https://github.com/DGator86/0DTE/pull/155
- Live system: https://0-dte-kappa.vercel.app/api/system
- Live dojo: https://0-dte-kappa.vercel.app/api/dojo
- SPY-DER ops doc (after patch): `docs/ops/dojo.md` (Preconditions section)
- This handoff: `docs/handoffs/DOJO_EMPTY_LATTICE_AGENT_HANDOFF.md`
