# SPY-DER Dojo empty-lattice fix

This agent can write to `DGator86/0DTE` but **not** `DGator86/SPY-DER`.
The Dojo fix lives in SPY-DER. Apply the patch there, then let
`spy-der-update.timer` pull it onto the VPS (or redeploy).

## What it fixes

Saturday’s weekly Dojo ran ~74 minutes, generated ~40k synthetic snapshots,
scored **zero**, and wrote an empty report because:

1. Universe lattice ran even with `recorded.status == insufficient_data`
2. `generate_result()` outcomes were discarded before scoring
3. Synthetic outcomes shipped `realized_pnl=None`, so the evaluator matched nothing

## Apply on SPY-DER

```bash
cd /path/to/SPY-DER
git checkout -b cursor/dojo-skip-empty-lattice-d52d
git am ../0DTE/patches/spy-der-dojo-skip-empty-lattice.patch
# or: git apply ../0DTE/patches/spy-der-dojo-skip-empty-lattice.patch && git commit
python3 -m pytest tests/unit/test_dojo_ownership.py tests/unit/test_dojo_phase4.py \
  tests/unit/test_synthetic.py -q --no-cov
git push -u origin HEAD
```

## Behavior after merge

| Situation | Universe phase |
|---|---|
| No recorded tape (default) | `skipped` + flag `universe_skipped_no_tape` — **no lattice** |
| `--force-universe` | Runs lattice and **scores** it (terminal P&L at settlement) |
| Tape present | Runs + scores as before |

Immediate VPS stopgap (until the patch lands):

```bash
sudo systemctl disable --now spy-der-dojo-weekly.timer
```
