"""
test_env_masking.py — action masks + safety layer under edge cases (§21).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import config as cfg
import safety


def _simple_pair_map():
    # 3 pairs over 3 currencies: EURUSD, GBPUSD, EURGBP -> USD=0, EUR=1, GBP=2
    # pair 0: EUR/USD -> (base=EUR=1, quote=USD=0)
    # pair 1: GBP/USD -> (base=GBP=2, quote=USD=0)
    # pair 2: EUR/GBP -> (base=EUR=1, quote=GBP=2)
    return [(1, 0), (2, 0), (1, 2)]


def _unscaled_bucket_lots(n_pairs):
    """bucket_lots grid at vol_scale=1.0 (i.e. plain bucket*BASE_LOT), for tests
    that don't care about vol-scaling specifically."""
    buckets = np.asarray(cfg.ACTION_BUCKETS, dtype=np.float64)
    return np.tile(buckets * cfg.BASE_LOT, (n_pairs, 1))


def test_exposure_cap_masks_breaching_buckets():
    positions = np.zeros(3)
    bucket_lots = _unscaled_bucket_lots(3)
    mask = safety.exposure_cap_mask(
        positions, _simple_pair_map(), n_currencies=3, bucket_lots=bucket_lots,
        max_pair_exposure=cfg.MAX_PAIR_EXPOSURE,
        max_abs_currency_exposure=cfg.MAX_ABS_EXPOSURE,
    )
    zero_idx = list(cfg.ACTION_BUCKETS).index(0)
    assert mask[0, zero_idx], "flat/hold must always remain legal from a flat start"
    # MAX_ABS_EXPOSURE=0.5 < a full +2 lot bucket, so extreme buckets must be masked
    # out for a currency starting at zero exposure.
    max_bucket_idx = list(cfg.ACTION_BUCKETS).index(max(cfg.ACTION_BUCKETS))
    assert not mask[0, max_bucket_idx], "a bucket breaching MAX_ABS_EXPOSURE must be masked out"


def test_vol_circuit_breaker_forces_flat_only():
    rng = np.random.default_rng(0)
    trailing = rng.normal(0, 0.05, size=cfg.VOL_BREAKER_WINDOW)  # artificially huge realized vol
    halted = safety.vol_circuit_breaker(trailing, longer_run_avg_vol=0.001)
    assert halted, "vol circuit breaker should trip when realized vol >> longer-run average"

    calm = rng.normal(0, 0.0001, size=cfg.VOL_BREAKER_WINDOW)
    not_halted = safety.vol_circuit_breaker(calm, longer_run_avg_vol=0.001)
    assert not not_halted, "vol circuit breaker should NOT trip when realized vol is below the longer-run average"


def test_hard_forced_flatten_only_at_last_bar():
    for bar_idx in range(cfg.BARS_PER_EPISODE):
        forced = safety.hard_forced_flatten(bar_idx, cfg.BARS_PER_EPISODE)
        if bar_idx == cfg.BARS_PER_EPISODE - 1:
            assert forced, "forced flatten must trigger on the final bar"
        else:
            assert not forced, f"forced flatten must NOT trigger before the final bar (bar {bar_idx})"


def test_priority_order_forced_flatten_overrides_everything():
    """§9: exposure caps -> vol breaker -> VaR cap -> hard forced flatten.
    Forced flatten is the last, unconditional check — it must win even if
    earlier checks would have allowed more."""
    positions = np.zeros(3)
    result = safety.apply_safety_layer(
        bar_idx=cfg.BARS_PER_EPISODE - 1,  # last bar
        current_positions=positions,
        pair_currency_map=_simple_pair_map(),
        n_currencies=3,
        trailing_portfolio_returns=np.array([0.0001] * 30),  # calm market, no vol/VaR halt
        longer_run_avg_vol=0.01,
    )
    assert result["forced_flatten"] is True
    zero_idx = list(cfg.ACTION_BUCKETS).index(0)
    assert np.all(result["action_mask"][:, zero_idx]), "only flat/hold should remain legal at forced flatten"
    assert not np.any(np.delete(result["action_mask"], zero_idx, axis=1)), \
        "every non-flat bucket must be masked out once forced flatten triggers"


def test_max_hold_mask_restricts_only_held_pairs():
    """§10: a pair held >= MAX_HOLD_BARS may only go flat; pairs under the
    limit are left fully unrestricted by this check (tested in isolation
    from exposure caps, which are covered separately above)."""
    zero_idx = list(cfg.ACTION_BUCKETS).index(0)
    hold_bars = np.array([cfg.MAX_HOLD_BARS, 0, cfg.MAX_HOLD_BARS - 1])
    mask = safety.max_hold_mask(hold_bars, n_buckets=len(cfg.ACTION_BUCKETS),
                                 zero_idx=zero_idx, max_hold_bars=cfg.MAX_HOLD_BARS)
    assert mask[0, zero_idx] and not np.any(np.delete(mask[0], zero_idx)), \
        "pair 0 hit MAX_HOLD_BARS — only flat should be legal"
    assert mask[1].all(), "pair 1 (hold_bars=0) should be fully unrestricted by max-hold"
    assert mask[2].all(), "pair 2 (one bar under the limit) should be fully unrestricted by max-hold"


def test_max_hold_halt_flag_via_apply_safety_layer():
    """Integration check: apply_safety_layer sets max_hold_halt and restricts
    the held pair to flat, on top of (not instead of) exposure caps."""
    positions = np.zeros(3)
    bucket_lots = _unscaled_bucket_lots(3)
    zero_idx = list(cfg.ACTION_BUCKETS).index(0)
    hold_bars = np.array([cfg.MAX_HOLD_BARS, 0, 0])

    result = safety.apply_safety_layer(
        bar_idx=0,
        current_positions=positions,
        pair_currency_map=_simple_pair_map(),
        n_currencies=3,
        trailing_portfolio_returns=np.array([0.0001] * 30),  # calm, no vol/VaR halt
        longer_run_avg_vol=0.01,
        bucket_lots=bucket_lots,
        hold_bars=hold_bars,
    )
    assert result["max_hold_halt"] is True
    mask = result["action_mask"]
    assert mask[0, zero_idx] and not np.any(np.delete(mask[0], zero_idx)), \
        "pair 0 hit MAX_HOLD_BARS — only flat should be legal"


def test_max_hold_disabled_when_explicitly_none():
    """An explicit max_hold_bars=None means disabled, NOT 'use the config
    default' — None is a meaningful disabled state here (config.py's own
    'None to disable' contract), distinct from omitting the argument."""
    hold_bars = np.array([999, 999, 999])
    mask = safety.max_hold_mask(hold_bars, n_buckets=len(cfg.ACTION_BUCKETS),
                                 zero_idx=list(cfg.ACTION_BUCKETS).index(0), max_hold_bars=None)
    assert mask.all(), "max_hold_bars=None must disable the check entirely"


def test_max_hold_uses_config_default_when_omitted():
    hold_bars = np.array([cfg.MAX_HOLD_BARS])
    zero_idx = list(cfg.ACTION_BUCKETS).index(0)
    mask = safety.max_hold_mask(hold_bars, n_buckets=len(cfg.ACTION_BUCKETS), zero_idx=zero_idx)
    assert mask[0, zero_idx] and not np.any(np.delete(mask[0], zero_idx)), \
        "omitting max_hold_bars should fall back to cfg.MAX_HOLD_BARS, not disable the check"


def test_exposure_cap_never_masks_out_flat_even_when_hedging():
    """Regression test for the crash seen in training: pair 0 (EUR/USD) is
    short and effectively hedging EUR exposure created by pairs 1 and 2. If
    pair 0 were forced flat, EUR's net exposure would grow *past* the cap
    (removing a hedge can look like "gaining" exposure under the mask's own
    math). exposure_cap_mask must still keep bucket 0 legal for pair 0 —
    flat can never be the bucket that gets masked out by an exposure cap."""
    # currencies: USD=0, EUR=1, GBP=2 (see _simple_pair_map)
    # pair 0: EUR/USD (base=EUR), pair 1: GBP/USD (base=GBP), pair 2: EUR/GBP (base=EUR, quote=GBP)
    positions = np.array([-1.5, 1.5, 1.5])  # pair 0 short EUR, offsetting pairs 1&2's long EUR
    bucket_lots = _unscaled_bucket_lots(3)
    mask = safety.exposure_cap_mask(
        positions, _simple_pair_map(), n_currencies=3, bucket_lots=bucket_lots,
        max_pair_exposure=cfg.MAX_PAIR_EXPOSURE,
        max_abs_currency_exposure=cfg.MAX_ABS_EXPOSURE,
    )
    zero_idx = list(cfg.ACTION_BUCKETS).index(0)
    assert mask[0, zero_idx], (
        "flat must remain legal for pair 0 even though closing its hedge "
        "would otherwise push EUR exposure further past the cap"
    )


def test_apply_safety_layer_never_returns_an_all_false_row():
    """Regression test: combining exposure caps with max-hold (or any other
    future combination of checks) must never AND a pair's row down to zero
    legal buckets — that produced NaN probs out of the masked softmax and
    crashed training (ValueError: probs ... does not satisfy Simplex())."""
    positions = np.array([-1.5, 1.5, 1.5])
    hold_bars = np.array([cfg.MAX_HOLD_BARS, 0, 0])  # pair 0 also hit max-hold
    result = safety.apply_safety_layer(
        bar_idx=0,
        current_positions=positions,
        pair_currency_map=_simple_pair_map(),
        n_currencies=3,
        trailing_portfolio_returns=np.array([0.0001] * 30),
        longer_run_avg_vol=0.01,
        bucket_lots=_unscaled_bucket_lots(3),
        hold_bars=hold_bars,
    )
    mask = result["action_mask"]
    assert mask.any(axis=1).all(), "every pair must retain at least one legal bucket"
    zero_idx = list(cfg.ACTION_BUCKETS).index(0)
    assert mask[0, zero_idx], "pair 0 should resolve the conflict by falling back to flat"


def test_priority_order_var_breach_masks_to_flat():
    positions = np.zeros(3)
    huge_losses = np.array([-0.5] * cfg.VAR_WINDOW)  # guarantees VaR cap breach
    result = safety.apply_safety_layer(
        bar_idx=0,
        current_positions=positions,
        pair_currency_map=_simple_pair_map(),
        n_currencies=3,
        trailing_portfolio_returns=huge_losses,
        longer_run_avg_vol=0.001,
    )
    assert result["var_halt"] is True
    zero_idx = list(cfg.ACTION_BUCKETS).index(0)
    assert np.all(result["action_mask"][:, zero_idx])


if __name__ == "__main__":
    test_exposure_cap_masks_breaching_buckets()
    test_vol_circuit_breaker_forces_flat_only()
    test_hard_forced_flatten_only_at_last_bar()
    test_max_hold_forces_flat_only_for_held_pairs()
    test_max_hold_disabled_when_none()
    test_priority_order_forced_flatten_overrides_everything()
    test_priority_order_var_breach_masks_to_flat()
    print("test_env_masking.py: all tests passed")
