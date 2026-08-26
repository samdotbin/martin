"""
scripts/generate_synthetic_data.py — writes synthetic H1 CSVs for all 28 pairs
so the full pipeline (data_pipeline -> env -> regime -> model -> ppo_trainer
-> fold_runner -> train -> eval) can be run and tested end-to-end without
real broker exports. This is NOT a substitute for real data before trusting
any result — see README §Data for how to plug in real MT5/vendor CSVs.

Usage:
    python scripts/generate_synthetic_data.py --years 6
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

import config as cfg


def generate_pair_csv(pair, start, periods, seed, out_dir):
    rng = np.random.default_rng(seed)
    # H1 bars, business-day-ish FX calendar approximated by dropping most weekend hours.
    timestamps = pd.date_range(start=start, periods=periods, freq="h", tz="UTC")
    dow = timestamps.dayofweek
    keep = ~((dow == 5) | ((dow == 6) & (timestamps.hour < 21)))
    timestamps = timestamps[keep]

    n = len(timestamps)
    drift = rng.normal(0, 1e-5, size=n)
    vol_regime = 1.0 + 0.5 * np.sin(np.linspace(0, 40, n)) + rng.normal(0, 0.05, size=n)
    log_returns = rng.normal(drift, 3e-4 * np.abs(vol_regime), size=n)
    price = 1.1 * np.exp(np.cumsum(log_returns))

    high = price * (1 + np.abs(rng.normal(0, 2e-4, size=n)))
    low = price * (1 - np.abs(rng.normal(0, 2e-4, size=n)))
    open_ = np.roll(price, 1)
    open_[0] = price[0]
    volume = rng.integers(50, 5000, size=n)
    spread = rng.uniform(0.5, 3.0, size=n)  # pips, synthetic

    df = pd.DataFrame({
        "timestamp": timestamps, "open": open_, "high": high, "low": low,
        "close": price, "volume": volume, "spread": spread,
    })
    df.to_csv(os.path.join(out_dir, f"{pair}.csv"), index=False)
    return len(df)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", type=int, default=6,
                         help="years of synthetic H1 data per pair (keep small for a fast smoke test)")
    parser.add_argument("--start", type=str, default="2019-01-01")
    parser.add_argument("--out-dir", type=str, default=cfg.DATA_DIR)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    periods = args.years * 365 * 24

    for i, pair in enumerate(cfg.PAIRS):
        n = generate_pair_csv(pair, args.start, periods, seed=i, out_dir=args.out_dir)
        print(f"{pair}: {n} bars -> {args.out_dir}/{pair}.csv")

    print(f"\nSynthetic dataset written for {len(cfg.PAIRS)} pairs, {args.years} years from {args.start}.")
    print("This is synthetic random-walk data for pipeline smoke-testing only — "
          "not a substitute for real market data before trusting any backtest result.")


if __name__ == "__main__":
    main()
