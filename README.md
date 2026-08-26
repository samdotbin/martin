# forex_rl_v4

Multi-pair FX RL trading system: a GNN encoder over 28 pairs / 8 currencies,
PPO training, walk-forward folds, and a safety layer that runs identically
at train and inference time. Full design in `forex_rl_spec.md` (v5). Current
implementation status: `PROGRESS.md`.

## Setup

```bash
pip install -r requirements.txt
```

`torch`/`numpy`/`pandas`/`scikit-learn`/`scipy` run anywhere. `MetaTrader5`
only installs on Windows and is only needed for real-data downloads below.

## Data

Every pair needs a CSV at `data/raw/{PAIR}.csv` with columns `timestamp,
open, high, low, close, volume, spread` — one row per H1 bar. Nothing else
in the pipeline fetches data for you; populate this directory first with
one of:

**Real data**, via a running MT5 terminal:
```bash
python scripts/download_mt5_data.py --years 6
```
Requires the MT5 terminal application running on this machine and logged
into a broker account that has all 28 pairs. It does **not** need to be
running from inside the terminal's own folder — the Python package talks to
the running terminal process, not to files on disk. Read the script's
docstring before your first real run: MT5 bar timestamps are in the
broker's **server time**, not UTC, and the whole pipeline's date logic
(fold splits, day-of-week checks, the lookback window's day contiguity)
assumes true UTC. The script auto-detects the offset and prints it — verify
that against what your broker publishes, and override with
`--server-utc-offset-hours` if it's wrong.

**Synthetic data**, for a smoke test with no broker connection:
```bash
python scripts/generate_synthetic_data.py --years 6
```

Either way, sanity-check what landed:
```bash
python data_pipeline.py
```
This reports each pair's coverage, the effective start year, and any
day-of-week or pip/point unit-mismatch warnings (§2) — it does not fetch or
modify data.

## Running it

```bash
pytest tests/ -v            # unit tests: reward, masking, regime leakage
python train.py --smoke-test   # 1 fold, 1 seed, 2 PPO iterations — checks the pipeline runs end to end
python train.py                # full run: all N_FOLDS folds, all seeds (config.py) — this is a long run, see forex_rl_spec.md §16 for the time estimate
```

## Project layout

| File | Covers |
|---|---|
| `config.py` | all tunables — pairs, currencies, action buckets, safety limits, fold dates |
| `data_pipeline.py` | loads CSVs already on disk, coverage validation, fold date-range slicing |
| `scripts/download_mt5_data.py` | real data, via a running MT5 terminal |
| `scripts/generate_synthetic_data.py` | synthetic data, for smoke tests |
| `env.py` | the trading environment — state, reward, step mechanics |
| `safety.py` | exposure caps, max-hold, vol circuit breaker, VaR cap, forced flatten |
| `regime.py` | train-only-fit regime/PCA embedding |
| `model.py` | GNN encoder + policy/value heads |
| `rollout.py` | episode collection |
| `ppo_trainer.py` | PPO update step, aux loss |
| `fold_runner.py` | one fold end-to-end: data → regime → train → eval |
| `train.py` | many folds/seeds via `fold_runner.py` |
| `eval.py` | policy evaluation, deflated Sharpe, bootstrap CI, stress tests |
| `utils.py` | seeding, logging, checkpoints, run manifest |
| `tests/` | unit tests |

See `forex_rl_spec.md` for the full design and `PROGRESS.md` for what's
built, tested, and still open.
