# forex_rl_v4 — Build Progress Tracker

Tracks implementation status against `forex_rl_spec.md` (v5). Update the
checkboxes as you go; re-run the verification commands at the bottom whenever
you want a fresh read on where things stand.

**Last verified:** 2026-08-24, against this fixed version of the code
(torch 2.13 CPU, synthetic data for the automated checks — real MT5 data
untested since it requires a live Windows terminal; see Gap 5 below).

## TL;DR

- The full pipeline (data → env → regime → GNN+PPO → eval) runs end-to-end
  without crashing, on synthetic data. 16/16 unit tests pass.
- All 5 gaps from the previous review are fixed and covered by new/updated
  tests, except the real-data connection itself, which is code-complete but
  needs a live MT5 terminal to actually exercise (see Gap 5).
- `data_pipeline.py` previously had **no `__main__` block at all** — running
  it directly did nothing, by design, not a bug in the strict sense, but
  clearly not what anyone expects. Fixed: it now prints a coverage report.

## Suggested Build Order (§22) — status

- [x] **1. `validate_data_coverage()` (§2)** — runs, reports effective start
      year and coverage warnings. `python data_pipeline.py` now surfaces this
      directly instead of doing nothing.
- [x] **2. Env + safety layer + action masking (§8, §9, §10)** — implemented
      and unit-tested (9/9 masking tests pass, including `MAX_HOLD_BARS`
      enforcement, fixed this round).
- [x] **3. Reward function unit tests (§7)** — differential Sharpe, close-out
      penalty, and turnover all match a hand-rolled reference (3/3 tests pass).
- [x] **4. Regime pipeline on a single fold (§11)** — min-data guard,
      SVD/PCA sign-fix, and train-only fit/transform split all verified
      (4/4 leakage tests pass).
- [x] **5. Time-awareness features (§3)** — `bars_remaining_norm` and
      `risk_budget_used` reach both the policy and value heads.
- [x] **6. Baseline PPO loop, single fold, γ=1 (§6, §13)** — ran end-to-end
      via `train.py --smoke-test` after all fixes; checkpoint + regime
      pickle saved together, test metrics computed.
- [x] **7. Auxiliary loss with warm-up (§5)** — `aux_weight_schedule` ramps
      0.01 → 0.10 over 50 epochs and is wired into `ppo_update`.
- [ ] **8. Fold-parallel harness + run manifest (§13, §14, §16)** — harness
      and manifest logging both work, but only ever run for 1 fold / 1 seed /
      2 iterations so far (smoke test). Not yet run across the full 8 folds
      × 3 seeds, and not yet run on real data.
- [ ] **9. Aggregate + evaluate: deflated Sharpe, bootstrap CI, residual
      exposure, fold-correlation caveat (§15)** — implemented in `eval.py`,
      but nothing to evaluate yet since only one partial smoke-test fold has
      been run.

## Fixed this round

1. **`MAX_HOLD_BARS` now enforced.** Added `safety.max_hold_mask()` — a pair
   held ≥ `MAX_HOLD_BARS` may only choose the flat bucket, on top of (not
   instead of) exposure caps. Wired into `apply_safety_layer` and
   `env.get_action_mask()`. New priority order: exposure caps → max hold →
   vol circuit breaker → VaR cap → forced flatten. 4 new tests, including
   one that catches an off-by-semantics bug in my first pass (an explicit
   `max_hold_bars=None` was silently falling back to the config default
   instead of actually disabling the check).

2. **Lookback no longer resets at fold boundaries.** Added
   `data_pipeline.contiguous_day_starts()`, a global-index version of the
   old per-split contiguity check. `fold_runner.py` now builds each
   `TradingEnv` on the *full* edge-feature array with only `valid_day_starts`
   restricted to that split's date range, so the 64-bar lookback for early
   val/test days can reach back into real prior-day bars instead of
   zero-padding, matching spec §10's stated intent. This also removed the
   dead `if False else` line that was in the old fold-slicing code.

3. **tz warning gone.** `load_fold()` now normalizes both sides of the date
   comparison to tz-naive UTC before comparing, instead of casting a
   tz-aware array straight to `datetime64[ns]`. Verified by re-running the
   smoke test with `-W error::UserWarning` (would have crashed if the
   warning still fired) — it didn't.

4. **`README.md` and `requirements.txt` added**, matching what the spec's
   file tree and `data_pipeline.py`'s old error message both referenced but
   didn't actually exist.

5. **Real MT5 data pipeline — this is what you asked for this round.**
   `data_pipeline.py` was never a downloader — it only reads CSVs already on
   disk, which is why `python data_pipeline.py` printed nothing before (no
   `__main__` block at all) and can't itself talk to MT5. Added
   `scripts/download_mt5_data.py`, a real counterpart to
   `scripts/generate_synthetic_data.py`: connects to your running MT5
   terminal via the official `MetaTrader5` package, pulls H1 bars for all 28
   pairs, and writes them in the exact CSV schema the rest of the pipeline
   expects. **Important, and worth reading before your first run:** MT5 bar
   timestamps are in the broker's server time, not UTC — the script
   auto-detects the offset and prints it, but auto-detection is a
   best-effort guess, not a guarantee, so sanity-check it against what your
   broker publishes before trusting a full download. I could not test this
   script end-to-end myself — the `MetaTrader5` package is Windows-only and
   needs a live logged-in terminal, neither of which exists in this sandbox
   — so treat your first run as the real test of it, and tell me what breaks.

## Cross-check against your original concerns

**Addressed in this rebuild:**
- *Leverage wasn't working, only hedging* → real position sizing via
  `ACTION_BUCKETS × BASE_LOT × vol_scale`, capped by `MAX_PAIR_EXPOSURE` /
  `MAX_ABS_EXPOSURE` (§8/§9).
- *Reward signal correctness* → differential Sharpe ratio (epsilon-guarded)
  plus turnover and close-out penalties, tested against a hand-rolled
  reference.
- *RL needs a horizon parameter* → `bars_remaining_norm` / `risk_budget_used`
  are explicit state inputs (§3), plus `MAX_HOLD_BARS` for shorter-than-
  episode closes, now actually enforced (see Fixed #1 above).

**Still open:**
- *GNN doesn't take in indicators / interest in FFT features* → still true.
  `EDGE_FEATURES` is still 4 raw channels (log return, HL range, volume z,
  spread z). Not touched this round — flag if you want this prioritized next.

## Not yet run

- **Real MT5 data end-to-end** — the download script is code-complete (Fixed
  #5) but unverified against a live terminal; run it and report back.
- The full 8-fold × 3-seed run (spec's own estimate: ~1-2 days on one GPU).
- `eval.py`'s deflated Sharpe / block-bootstrap / stress test — implemented
  and individually sound, but nothing meaningful to run them on yet.

## How to re-run this check

```bash
pip install -r requirements.txt
pytest tests/ -v
python data_pipeline.py        # coverage sanity check on whatever data is present
python train.py --smoke-test
```

To get real data first (on the Windows machine with MT5 installed and
logged in):
```bash
python scripts/download_mt5_data.py --years 6
```
