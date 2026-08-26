"""
scripts/benchmark_rollout.py — measures the real speedup from batching
rollout collection across ROLLOUT_N_ENVS parallel lanes vs. the old
single-lane (n_lanes=1) sequential behavior, on this machine.

Run this locally before committing to a long Colab run — the win comes from
collapsing the Python-interpreter-bound per-step forward-call count (model.py's
GNN loops over all 28 pairs in Python on every layer, every call), not from
raw FLOPs, so it does NOT scale linearly with N_ENVS and is worth measuring
directly rather than assuming.

Usage:
    python scripts/benchmark_rollout.py
    python scripts/benchmark_rollout.py --n-envs 64 --episodes 64
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import config as cfg
import env as env_module
import fold_runner
import rollout as rollout_module
import utils
from model import PolicyValueNet


def _build_universe(n_days=40):
    pair_currency_map = fold_runner._pair_currency_map()
    n_pairs = len(cfg.PAIRS)
    n_bars = cfg.LOOKBACK_BARS + n_days * cfg.BARS_PER_EPISODE + 1
    rng = np.random.default_rng(0)
    edge_features = rng.normal(0, 0.001, size=(n_bars, n_pairs, cfg.EDGE_FEATURES)).astype(np.float32)
    timestamps = np.array(
        np.datetime64("2020-01-06") + np.arange(n_bars) * np.timedelta64(1, "h")
    )
    valid_day_starts = np.arange(cfg.LOOKBACK_BARS, cfg.LOOKBACK_BARS + n_days * cfg.BARS_PER_EPISODE,
                                  cfg.BARS_PER_EPISODE, dtype=int)
    return edge_features, timestamps, valid_day_starts, pair_currency_map


def _build_lanes(n_lanes, edge_features, timestamps, valid_day_starts, pair_currency_map):
    lanes = []
    for _ in range(n_lanes):
        lane = env_module.TradingEnv(edge_features, timestamps, valid_day_starts, pair_currency_map, cfg=cfg)
        lane.set_regime(np.zeros(cfg.N_REGIME_COMPONENTS + cfg.N_REGIME_CLUSTERS, dtype=np.float32))
        lanes.append(lane)
    return lanes


def _time_rollout(n_lanes, n_episodes, edge_features, timestamps, valid_day_starts, pair_currency_map):
    utils.set_seed(0)
    lanes = _build_lanes(n_lanes, edge_features, timestamps, valid_day_starts, pair_currency_map)
    model = PolicyValueNet(pair_currency_map).to(cfg.DEVICE)
    start = time.perf_counter()
    rollout_module.collect_rollout(lanes, model, None, n_episodes, cfg)
    return time.perf_counter() - start


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-envs", type=int, default=cfg.ROLLOUT_N_ENVS)
    parser.add_argument("--episodes", type=int, default=64, help="episodes per timed rollout")
    args = parser.parse_args()

    universe = _build_universe()
    print(f"device: {cfg.DEVICE}, episodes: {args.episodes}, "
          f"comparing n_lanes=1 vs n_lanes={args.n_envs}\n")

    t_sequential = _time_rollout(1, args.episodes, *universe)
    print(f"n_lanes=1:            {t_sequential:.1f}s")

    t_vectorized = _time_rollout(args.n_envs, args.episodes, *universe)
    print(f"n_lanes={args.n_envs}:{' ' * max(1, 12 - len(str(args.n_envs)))}{t_vectorized:.1f}s")

    speedup = t_sequential / t_vectorized if t_vectorized > 0 else float("inf")
    print(f"\nspeedup: {speedup:.1f}x")
    print("(expect a solid win from fewer, larger forward calls, not a naive "
          "linear-in-N speedup — CPU FLOPs still scale with batch size)")


if __name__ == "__main__":
    main()
