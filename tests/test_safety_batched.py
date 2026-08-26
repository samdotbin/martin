"""
test_safety_batched.py — the lane-batched safety functions (used only by
rollout.py's hot training loop) must be mathematically identical, lane by
lane, to calling the original per-lane functions in a loop. Fuzz-tested
across many random scenarios rather than a handful of hand-picked ones,
since this is safety-critical code with a documented history of subtle
mask-interaction bugs (see safety.py's own comments) (§21).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import config as cfg
import safety


def _simple_pair_map():
    # Same 3-pair/3-currency fixture as test_env_masking.py.
    return [(1, 0), (2, 0), (1, 2)]


def test_exposure_cap_mask_batched_matches_looped_original():
    rng = np.random.default_rng(0)
    pair_map = _simple_pair_map()
    n_pairs, n_buckets, n_currencies = 3, len(cfg.ACTION_BUCKETS), 3
    L = 16

    for trial in range(20):
        positions = rng.uniform(-2.5, 2.5, size=(L, n_pairs))
        bucket_lots = rng.uniform(-3.0, 3.0, size=(L, n_pairs, n_buckets))
        # Make sure the "flat" bucket really is exactly 0.0 in some lanes,
        # since that's the invariant under test.
        zero_idx = list(cfg.ACTION_BUCKETS).index(0)
        bucket_lots[:, :, zero_idx] = 0.0

        batched = safety.exposure_cap_mask_batched(
            positions, pair_map, n_currencies, bucket_lots,
            max_pair_exposure=cfg.MAX_PAIR_EXPOSURE, max_abs_currency_exposure=cfg.MAX_ABS_EXPOSURE,
        )
        for lane in range(L):
            expected = safety.exposure_cap_mask(
                positions[lane], pair_map, n_currencies, bucket_lots[lane],
                max_pair_exposure=cfg.MAX_PAIR_EXPOSURE, max_abs_currency_exposure=cfg.MAX_ABS_EXPOSURE,
            )
            assert np.array_equal(batched[lane], expected), f"trial {trial} lane {lane} mismatch"


def test_max_hold_mask_batched_matches_looped_original():
    rng = np.random.default_rng(1)
    n_pairs, n_buckets = 5, len(cfg.ACTION_BUCKETS)
    zero_idx = list(cfg.ACTION_BUCKETS).index(0)
    L = 10

    for trial in range(10):
        hold_bars = rng.integers(0, cfg.MAX_HOLD_BARS + 3, size=(L, n_pairs))
        batched = safety.max_hold_mask_batched(hold_bars, n_buckets, zero_idx, max_hold_bars=cfg.MAX_HOLD_BARS)
        for lane in range(L):
            expected = safety.max_hold_mask(hold_bars[lane], n_buckets, zero_idx, max_hold_bars=cfg.MAX_HOLD_BARS)
            assert np.array_equal(batched[lane], expected), f"trial {trial} lane {lane} mismatch"


def test_apply_safety_layer_batched_matches_looped_original():
    rng = np.random.default_rng(2)
    pair_map = _simple_pair_map()
    n_pairs, n_buckets, n_currencies = 3, len(cfg.ACTION_BUCKETS), 3
    zero_idx = list(cfg.ACTION_BUCKETS).index(0)
    L = 12

    for trial in range(15):
        positions = rng.uniform(-2.5, 2.5, size=(L, n_pairs))
        bucket_lots = rng.uniform(-3.0, 3.0, size=(L, n_pairs, n_buckets))
        bucket_lots[:, :, zero_idx] = 0.0
        hold_bars = rng.integers(0, cfg.MAX_HOLD_BARS + 3, size=(L, n_pairs))
        bar_idx_per_lane = rng.integers(0, cfg.BARS_PER_EPISODE, size=L)
        # Mix calm and volatile lanes so vol/VaR halts trigger in some but not others.
        trailing_returns_per_lane = [
            rng.normal(0, 0.05 if lane % 3 == 0 else 0.0001, size=cfg.VAR_WINDOW)
            for lane in range(L)
        ]
        longer_run_vol_per_lane = [0.001] * L

        batched = safety.apply_safety_layer_batched(
            bar_idx_per_lane, positions, pair_map, n_currencies,
            trailing_returns_per_lane, longer_run_vol_per_lane,
            bucket_lots, hold_bars, cfg=cfg,
        )

        for lane in range(L):
            expected = safety.apply_safety_layer(
                bar_idx=int(bar_idx_per_lane[lane]),
                current_positions=positions[lane],
                pair_currency_map=pair_map,
                n_currencies=n_currencies,
                trailing_portfolio_returns=trailing_returns_per_lane[lane],
                longer_run_avg_vol=longer_run_vol_per_lane[lane],
                bucket_lots=bucket_lots[lane],
                hold_bars=hold_bars[lane],
                cfg=cfg,
            )
            assert np.array_equal(batched["action_mask"][lane], expected["action_mask"]), \
                f"trial {trial} lane {lane}: action_mask mismatch"
            assert bool(batched["forced_flatten"][lane]) == expected["forced_flatten"]
            assert bool(batched["vol_halt"][lane]) == expected["vol_halt"]
            assert bool(batched["var_halt"][lane]) == expected["var_halt"]
            assert bool(batched["max_hold_halt"][lane]) == expected["max_hold_halt"]
            # Every pair must always keep at least one legal bucket (the
            # invariant apply_safety_layer's own regression tests guard).
            assert batched["action_mask"][lane].any(axis=-1).all()


if __name__ == "__main__":
    test_exposure_cap_mask_batched_matches_looped_original()
    test_max_hold_mask_batched_matches_looped_original()
    test_apply_safety_layer_batched_matches_looped_original()
    print("test_safety_batched.py: all tests passed")
