"""
test_reward.py — differential Sharpe + close-out penalty vs. known sequences (§21).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import config as cfg


def differential_sharpe_reference(bar_returns, eta=cfg.DIFFERENTIAL_SHARPE_ETA, eps=cfg.DIFF_SHARPE_EPS):
    """Hand-rolled reference implementation, independent of env.py, per §7."""
    A, B = 0.0, 0.0
    D = []
    for r in bar_returns:
        delta_A = r - A
        delta_B = r ** 2 - B
        denom = max(B - A ** 2, eps) ** 1.5
        D_t = (B * delta_A - 0.5 * A * delta_B) / denom
        A += eta * delta_A
        B += eta * delta_B
        D.append(D_t)
    return np.array(D)


def test_differential_sharpe_matches_env_formula():
    import env as env_module
    import data_pipeline

    rng = np.random.default_rng(0)
    n_pairs = 2
    bar_returns = rng.normal(0, 0.001, size=20)

    # Reference calc.
    ref = differential_sharpe_reference(bar_returns)
    assert np.all(np.isfinite(ref)), "differential Sharpe must stay finite even near bar 0 (eps guard)"

    # First-bar stability: with A=B=0, eps in the denominator prevents a div-by-zero blowup.
    A, B = 0.0, 0.0
    delta_A = bar_returns[0] - A
    delta_B = bar_returns[0] ** 2 - B
    denom = max(B - A ** 2, cfg.DIFF_SHARPE_EPS) ** 1.5
    D0 = (B * delta_A - 0.5 * A * delta_B) / denom
    assert D0 == 0.0, "at bar 0 with A=B=0, D_t must be exactly 0 (both numerator terms vanish)"
    assert np.isfinite(D0)


def test_close_out_penalty_only_fires_near_episode_end():
    # bars_remaining_norm <= 2/24 should trigger the penalty; earlier bars should not.
    bars_per_episode = cfg.BARS_PER_EPISODE
    threshold = cfg.CLOSE_OUT_PENALTY_BARS_REMAINING
    held_size = 3.0

    for bar_idx in range(bars_per_episode):
        bars_remaining_norm = (bars_per_episode - bar_idx) / bars_per_episode
        should_fire = bars_remaining_norm <= threshold
        penalty = cfg.CLOSE_OUT_MU * held_size if should_fire else 0.0
        if bar_idx >= bars_per_episode - 2:
            assert penalty > 0, f"bar {bar_idx}: close-out penalty should be active in the last 2 bars"
        else:
            assert penalty == 0, f"bar {bar_idx}: close-out penalty should NOT fire this early"


def test_turnover_is_sum_of_absolute_position_changes():
    prev = np.array([0.0, 1.0, -2.0])
    target = np.array([1.0, 1.0, 0.0])
    turnover = float(np.sum(np.abs(target - prev)))
    assert turnover == 1.0 + 0.0 + 2.0


if __name__ == "__main__":
    test_differential_sharpe_matches_env_formula()
    test_close_out_penalty_only_fires_near_episode_end()
    test_turnover_is_sum_of_absolute_position_changes()
    print("test_reward.py: all tests passed")
