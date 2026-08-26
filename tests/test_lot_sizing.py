"""
test_lot_sizing.py — env.py's vol-scaled lot sizing (§8) must produce the
same numbers whether computed via the naive per-(pair,bucket) loop or a
vectorized version. Reference implementation is independent of env.py
(mirrors the original per-bucket formula), following the repo's existing
convention (test_reward.py) rather than diffing against pre-refactor code.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import config as cfg
import env as env_module
import fold_runner


def _reference_lots_for_bucket(edge_features, idx, bucket_val, pair_idx, cfg_module):
    window = edge_features[max(0, idx - 20):idx + 1, :, 0]
    if window.shape[0] < 2:
        vol_scale = 1.0
    else:
        per_pair_vol = np.std(window, axis=0) + 1e-9
        median_vol = np.median(per_pair_vol)
        symbol_vol = per_pair_vol[pair_idx]
        vol_scale = float(np.clip(median_vol / symbol_vol, *cfg_module.VOL_SCALE_CLIP))
    return bucket_val * cfg_module.BASE_LOT * vol_scale


def _reference_bucket_lots_grid(edge_features, idx, n_pairs, cfg_module):
    buckets = np.asarray(cfg_module.ACTION_BUCKETS)
    grid = np.zeros((n_pairs, len(buckets)))
    for p in range(n_pairs):
        for b, bv in enumerate(buckets):
            grid[p, b] = _reference_lots_for_bucket(edge_features, idx, bv, p, cfg_module)
    return grid


def _build_env(n_bars=120, seed=0):
    pair_currency_map = fold_runner._pair_currency_map()
    n_pairs = len(cfg.PAIRS)
    rng = np.random.default_rng(seed)
    edge_features = rng.normal(0, 0.001, size=(n_bars, n_pairs, cfg.EDGE_FEATURES)).astype(np.float32)
    timestamps = np.array(
        np.datetime64("2020-01-06") + np.arange(n_bars) * np.timedelta64(1, "h")
    )
    valid_day_starts = np.array([cfg.LOOKBACK_BARS], dtype=int)
    env = env_module.TradingEnv(edge_features, timestamps, valid_day_starts, pair_currency_map, cfg=cfg)
    return env, edge_features, n_pairs


def test_bucket_lots_grid_matches_naive_reference_across_idx_values():
    env, edge_features, n_pairs = _build_env()
    # idx < 20 exercises the short-window branch; idx >= 20 the full-window branch.
    for idx in (0, 1, 19, 20, 21, 90, 119):
        expected = _reference_bucket_lots_grid(edge_features, idx, n_pairs, cfg)
        actual = env._bucket_lots_grid(idx)
        assert np.allclose(actual, expected, atol=1e-9), f"mismatch at idx={idx}"


def test_step_target_lots_matches_naive_reference():
    env, edge_features, n_pairs = _build_env()
    env.reset(day_start_idx=cfg.LOOKBACK_BARS)
    idx = env._current_bar_absolute_idx()
    buckets = np.asarray(cfg.ACTION_BUCKETS)
    rng = np.random.default_rng(3)
    bucket_indices = rng.integers(0, len(buckets), size=n_pairs)

    expected = np.array([
        _reference_lots_for_bucket(edge_features, idx, buckets[bucket_indices[p]], p, cfg)
        for p in range(n_pairs)
    ])
    expected = np.clip(expected, -cfg.MAX_PAIR_EXPOSURE, cfg.MAX_PAIR_EXPOSURE)

    env.step(bucket_indices)
    actual = env.positions  # step() sets self.positions = clipped target_lots

    assert np.allclose(actual, expected, atol=1e-5)


if __name__ == "__main__":
    test_bucket_lots_grid_matches_naive_reference_across_idx_values()
    test_step_target_lots_matches_naive_reference()
    print("test_lot_sizing.py: all tests passed")
