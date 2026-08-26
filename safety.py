"""
safety.py — vol circuit breaker, VaR cap, exposure caps, hard flatten (§9).

Every check is a pure function of (state, config) -> mask/bool, with no
internal state of its own (§21 implementation hint). This makes each one
trivially unit-testable in isolation, and safe to reuse unchanged in a future
live-trading harness — §9 requires the safety layer to run identically at
train and inference time, which only holds if it's one shared module.

Priority order when checks conflict (§9):
    exposure caps -> max hold -> vol circuit breaker -> VaR cap -> hard forced flatten
Exposure caps and max-hold are both cheap, local, per-pair checks and run
first; the VaR cap is portfolio-wide and looks at everything the earlier
checks already allowed, so it is the final gate before execution (forced
flatten aside, which is unconditional at bar 24).
"""
import numpy as np

import config as _default_config


def exposure_cap_mask(
    current_positions: np.ndarray,      # (n_pairs,) lots, signed
    pair_currency_map,                  # list of (base_idx, quote_idx) per pair
    n_currencies: int,
    bucket_lots: np.ndarray,            # (n_pairs, n_buckets) ALREADY vol-scaled lot size
                                         # each bucket would produce (§8) — this must be the
                                         # same value env.step() will actually execute, or a
                                         # bucket the mask calls "legal" could still breach the
                                         # cap once vol_scale is applied downstream.
    max_pair_exposure=None,
    max_abs_currency_exposure=None,
) -> np.ndarray:
    """
    Returns a (n_pairs, n_buckets) boolean mask: True = bucket is legal.
    A bucket is masked out if taking it would breach the per-pair cap or push
    any currency's net exposure beyond MAX_ABS_EXPOSURE (§8, §9).
    """
    bucket_lots = np.asarray(bucket_lots, dtype=np.float64)
    max_pair_exposure = (
        max_pair_exposure if max_pair_exposure is not None else _default_config.MAX_PAIR_EXPOSURE
    )
    max_abs_currency_exposure = (
        max_abs_currency_exposure
        if max_abs_currency_exposure is not None
        else _default_config.MAX_ABS_EXPOSURE
    )

    n_pairs, n_buckets = bucket_lots.shape
    mask = np.ones((n_pairs, n_buckets), dtype=bool)

    # Net per-currency exposure implied by current positions (before this bar's action).
    currency_exposure = np.zeros(n_currencies)
    for i, (base_idx, quote_idx) in enumerate(pair_currency_map):
        currency_exposure[base_idx] += current_positions[i]
        currency_exposure[quote_idx] -= current_positions[i]

    for p in range(n_pairs):
        base_idx, quote_idx = pair_currency_map[p]
        delta_from_current = current_positions[p]
        for b in range(n_buckets):
            target_lots = bucket_lots[p, b]
            if target_lots == 0.0:
                # Flat/hold-none is always legal, regardless of what other
                # pairs are doing. Without this, going flat can look like it
                # *breaches* the cap: removing pair p's own contribution from
                # currency_exposure can enlarge the *other* currency's net
                # exposure if p's position was partially hedging it (see the
                # new_exposure_base/quote math below). That would let the
                # exposure cap mask out bucket 0 itself — the one bucket every
                # other check in this file (max-hold, vol breaker, VaR cap,
                # forced flatten) assumes it can always fall back to. Losing
                # that guarantee is what let a pair's mask row go all-False
                # once max-hold also forced "flat only" on the same pair,
                # crashing the downstream Categorical (all -inf -> NaN probs).
                continue
            if abs(target_lots) > max_pair_exposure + 1e-9:
                mask[p, b] = False
                continue
            new_exposure_base = currency_exposure[base_idx] - delta_from_current + target_lots
            new_exposure_quote = currency_exposure[quote_idx] + delta_from_current - target_lots
            if (
                abs(new_exposure_base) > max_abs_currency_exposure + 1e-9
                or abs(new_exposure_quote) > max_abs_currency_exposure + 1e-9
            ):
                mask[p, b] = False

    return mask


_UNSET = object()  # distinguishes "caller passed nothing" from "caller passed None"


def max_hold_mask(
    hold_bars: np.ndarray,     # (n_pairs,) consecutive bars each pair has been held
    n_buckets: int,
    zero_idx: int,
    max_hold_bars=_UNSET,
) -> np.ndarray:
    """
    Returns a (n_pairs, n_buckets) boolean mask: a pair held >= MAX_HOLD_BARS
    may only choose the flat bucket this step — forces active management
    instead of parking a position for the whole episode (§10 "max hold:
    optional forced close after N bars").

    max_hold_bars follows config.py's "None to disable" contract: omit the
    argument to use cfg.MAX_HOLD_BARS, or pass max_hold_bars=None explicitly
    to disable the check (all-True mask) regardless of the config default —
    None is a meaningful disabled state here, not "no override given", so it
    is NOT silently replaced by the config default the way an omitted
    argument is.
    """
    if max_hold_bars is _UNSET:
        max_hold_bars = _default_config.MAX_HOLD_BARS
    n_pairs = len(hold_bars)
    mask = np.ones((n_pairs, n_buckets), dtype=bool)
    if not max_hold_bars:
        return mask
    forced = np.asarray(hold_bars) >= max_hold_bars
    mask[forced, :] = False
    mask[forced, zero_idx] = True
    return mask


def vol_circuit_breaker(
    trailing_returns: np.ndarray,   # last VOL_BREAKER_WINDOW bars of portfolio returns
    longer_run_avg_vol: float,      # trailing average realized vol over a longer window
    window=None,
    multiplier=None,
) -> bool:
    """True => halt new entries this bar (§9)."""
    window = window if window is not None else _default_config.VOL_BREAKER_WINDOW
    multiplier = multiplier if multiplier is not None else _default_config.VOL_BREAKER_MULTIPLIER
    recent = np.asarray(trailing_returns)[-window:]
    if len(recent) < window or longer_run_avg_vol <= 0:
        return False
    realized_vol = np.std(recent)
    return bool(realized_vol > multiplier * longer_run_avg_vol)


def historical_var(portfolio_returns_window: np.ndarray, percentile: float = 5.0) -> float:
    """
    Historical VaR: the specified percentile of the trailing window's portfolio
    returns. No Gaussian/Cornish-Fisher assumption — appropriate for fat-tailed
    FX returns (§9). Returns a positive number representing the loss magnitude.
    """
    if len(portfolio_returns_window) == 0:
        return 0.0
    q = np.percentile(portfolio_returns_window, percentile)
    return float(max(-q, 0.0))


def var_cap_breach(
    portfolio_returns_window: np.ndarray, var_cap=None, window=None
) -> bool:
    var_cap = var_cap if var_cap is not None else _default_config.VAR_CAP
    window = window if window is not None else _default_config.VAR_WINDOW
    recent = np.asarray(portfolio_returns_window)[-window:]
    var_est = historical_var(recent)
    return bool(var_est > var_cap)


def hard_forced_flatten(bar_idx: int, bars_per_episode=None) -> bool:
    """
    Independent of anything the policy learned: force all positions flat at
    the last bar, regardless of §3's shaping penalty (§9). This is a backstop,
    not the primary mechanism.
    """
    bars_per_episode = bars_per_episode if bars_per_episode is not None else _default_config.BARS_PER_EPISODE
    return bar_idx >= bars_per_episode - 1


def apply_safety_layer(
    bar_idx: int,
    current_positions: np.ndarray,
    pair_currency_map,
    n_currencies: int,
    trailing_portfolio_returns: np.ndarray,
    longer_run_avg_vol: float,
    bucket_lots: np.ndarray = None,   # (n_pairs, n_buckets), vol-scaled — see exposure_cap_mask
    hold_bars: np.ndarray = None,     # (n_pairs,) consecutive bars each pair has been held
    cfg=_default_config,
) -> dict:
    """
    Runs all checks in priority order and returns:
      {
        "action_mask": (n_pairs, n_buckets) bool,
        "forced_flatten": bool,
        "vol_halt": bool,
        "var_halt": bool,
        "max_hold_halt": bool,
      }
    Priority order (§9): exposure caps -> max hold -> vol circuit breaker ->
    VaR cap -> hard forced flatten. Earlier checks' masks are always applied;
    later checks can only further restrict (never relax) what's legal.
    """
    result = {"forced_flatten": False, "vol_halt": False, "var_halt": False, "max_hold_halt": False}

    if bucket_lots is None:
        # Fallback for callers that don't have vol-scaled lots handy (e.g. a
        # quick unit test): approximate with the unscaled bucket*base_lot grid.
        buckets = np.asarray(cfg.ACTION_BUCKETS, dtype=np.float64)
        n_pairs = len(current_positions)
        bucket_lots = np.tile(buckets * cfg.BASE_LOT, (n_pairs, 1))

    # 1. Exposure caps (cheapest, most local).
    mask = exposure_cap_mask(
        current_positions, pair_currency_map, n_currencies, bucket_lots,
        max_pair_exposure=cfg.MAX_PAIR_EXPOSURE,
        max_abs_currency_exposure=cfg.MAX_ABS_EXPOSURE,
    )

    zero_idx = list(cfg.ACTION_BUCKETS).index(0)

    # 2. Max hold: pairs held >= MAX_HOLD_BARS may only go flat this step.
    if hold_bars is not None:
        n_buckets = mask.shape[1]
        mh_mask = max_hold_mask(hold_bars, n_buckets, zero_idx, max_hold_bars=cfg.MAX_HOLD_BARS)
        if not mh_mask.all():
            result["max_hold_halt"] = True
        mask &= mh_mask

    # 3. Vol circuit breaker: halt NEW entries (mask out non-flat buckets), keep flat legal.
    if vol_circuit_breaker(trailing_portfolio_returns, longer_run_avg_vol,
                            window=cfg.VOL_BREAKER_WINDOW, multiplier=cfg.VOL_BREAKER_MULTIPLIER):
        result["vol_halt"] = True
        mask[:, :] = False
        mask[:, zero_idx] = True  # only "flat/hold" remains legal

    # 4. VaR cap: portfolio-wide, looks at everything already allowed. Breach -> flat only.
    if var_cap_breach(trailing_portfolio_returns, var_cap=cfg.VAR_CAP, window=cfg.VAR_WINDOW):
        result["var_halt"] = True
        mask[:, :] = False
        mask[:, zero_idx] = True

    # 5. Hard forced flatten: unconditional at the last bar, regardless of the above.
    if hard_forced_flatten(bar_idx, cfg.BARS_PER_EPISODE):
        result["forced_flatten"] = True
        mask[:, :] = False
        mask[:, zero_idx] = True

    # 6. Invariant safety net: every pair must have at least one legal bucket.
    # exposure_cap_mask keeps flat legal on its own now (see above), but this
    # is a cheap, permanent backstop against any future check — or any future
    # combination of checks — that could otherwise AND a pair's row down to
    # all-False and hand a NaN-producing masked-softmax row to the policy.
    no_legal_action = ~mask.any(axis=1)
    if no_legal_action.any():
        mask[no_legal_action, :] = False
        mask[no_legal_action, zero_idx] = True

    result["action_mask"] = mask
    return result


# ---------------------------------------------------------------------------
# Lane-batched variants — used only by rollout.py's hot training-collection
# loop, where exposure_cap_mask()'s O(n_pairs * n_buckets) Python loop,
# run once per lane per step, profiled as the dominant non-model cost once
# rollout collection was batched across lanes. Mathematically identical,
# lane by lane, to calling the functions above once per lane in a loop —
# verified by a fuzz-equivalence test (tests/test_safety_batched.py), not
# just asserted. The functions above are left completely untouched and stay
# the ones used everywhere else (eval.py, single-env stepping, all existing
# tests), so nothing about the well-tested single-lane path changes.
# ---------------------------------------------------------------------------
def exposure_cap_mask_batched(
    current_positions,      # (L, n_pairs)
    pair_currency_map,
    n_currencies: int,
    bucket_lots,             # (L, n_pairs, n_buckets)
    max_pair_exposure=None,
    max_abs_currency_exposure=None,
) -> np.ndarray:
    """Lane-batched exposure_cap_mask(): -> (L, n_pairs, n_buckets) bool."""
    bucket_lots = np.asarray(bucket_lots, dtype=np.float64)
    current_positions = np.asarray(current_positions, dtype=np.float64)
    max_pair_exposure = (
        max_pair_exposure if max_pair_exposure is not None else _default_config.MAX_PAIR_EXPOSURE
    )
    max_abs_currency_exposure = (
        max_abs_currency_exposure
        if max_abs_currency_exposure is not None
        else _default_config.MAX_ABS_EXPOSURE
    )

    L, n_pairs, n_buckets = bucket_lots.shape
    base_idx = np.array([m[0] for m in pair_currency_map])
    quote_idx = np.array([m[1] for m in pair_currency_map])

    # Net per-currency exposure implied by current positions, per lane —
    # np.add.at (not fancy-index assignment) because multiple pairs can
    # share a currency and their contributions must accumulate, not overwrite.
    currency_exposure = np.zeros((L, n_currencies))
    lane_rows = np.arange(L)[:, None]
    np.add.at(currency_exposure, (lane_rows, base_idx[None, :]), current_positions)
    np.add.at(currency_exposure, (lane_rows, quote_idx[None, :]), -current_positions)

    cur_exp_base = currency_exposure[:, base_idx]    # (L, n_pairs)
    cur_exp_quote = currency_exposure[:, quote_idx]  # (L, n_pairs)
    delta = current_positions                        # (L, n_pairs) == "delta_from_current" per pair

    new_exposure_base = cur_exp_base[:, :, None] - delta[:, :, None] + bucket_lots
    new_exposure_quote = cur_exp_quote[:, :, None] + delta[:, :, None] - bucket_lots

    breach_pair_cap = np.abs(bucket_lots) > (max_pair_exposure + 1e-9)
    breach_currency = (
        (np.abs(new_exposure_base) > max_abs_currency_exposure + 1e-9)
        | (np.abs(new_exposure_quote) > max_abs_currency_exposure + 1e-9)
    )

    mask = ~(breach_pair_cap | breach_currency)
    # Flat/hold-none is always legal (see exposure_cap_mask's docstring for
    # why — closing a hedge can look like it breaches the cap under this
    # same arithmetic). Applied last, unconditionally, matching the
    # single-lane version's `continue`-before-any-check semantics exactly.
    mask[bucket_lots == 0.0] = True
    return mask


def max_hold_mask_batched(hold_bars, n_buckets: int, zero_idx: int, max_hold_bars=_UNSET) -> np.ndarray:
    """Lane-batched max_hold_mask(): hold_bars (L, n_pairs) -> (L, n_pairs, n_buckets) bool."""
    if max_hold_bars is _UNSET:
        max_hold_bars = _default_config.MAX_HOLD_BARS
    hold_bars = np.asarray(hold_bars)
    L, n_pairs = hold_bars.shape
    mask = np.ones((L, n_pairs, n_buckets), dtype=bool)
    if not max_hold_bars:
        return mask
    forced = hold_bars >= max_hold_bars  # (L, n_pairs)
    mask[forced] = False
    mask[forced, zero_idx] = True
    return mask


def apply_safety_layer_batched(
    bar_idx_per_lane,
    current_positions,          # (L, n_pairs)
    pair_currency_map,
    n_currencies: int,
    trailing_returns_per_lane,  # length-L list of 1D arrays, one per lane
    longer_run_avg_vol_per_lane,  # length-L list of floats
    bucket_lots,                 # (L, n_pairs, n_buckets)
    hold_bars,                   # (L, n_pairs)
    cfg=_default_config,
) -> dict:
    """
    Lane-batched apply_safety_layer(). Batches the two checks whose per-lane
    cost is a proven bottleneck (exposure_cap_mask, max_hold_mask) across all
    active lanes in one vectorized call; the vol circuit breaker, VaR cap,
    and hard forced flatten checks stay a simple per-lane loop since they
    operate on small per-lane windows (<= a few dozen elements — cheap
    regardless) and batching them isn't where the cost was. Same priority
    order as apply_safety_layer(): exposure caps -> max hold -> vol circuit
    breaker -> VaR cap -> hard forced flatten -> invariant backstop.
    """
    bucket_lots = np.asarray(bucket_lots, dtype=np.float64)
    L, n_pairs, n_buckets = bucket_lots.shape
    zero_idx = list(cfg.ACTION_BUCKETS).index(0)

    mask = exposure_cap_mask_batched(
        current_positions, pair_currency_map, n_currencies, bucket_lots,
        max_pair_exposure=cfg.MAX_PAIR_EXPOSURE,
        max_abs_currency_exposure=cfg.MAX_ABS_EXPOSURE,
    )

    max_hold_halt = np.zeros(L, dtype=bool)
    if hold_bars is not None:
        mh_mask = max_hold_mask_batched(hold_bars, n_buckets, zero_idx, max_hold_bars=cfg.MAX_HOLD_BARS)
        max_hold_halt = ~mh_mask.all(axis=(1, 2))
        mask &= mh_mask

    vol_halt = np.zeros(L, dtype=bool)
    var_halt = np.zeros(L, dtype=bool)
    forced_flatten = np.zeros(L, dtype=bool)
    for lane in range(L):
        if vol_circuit_breaker(trailing_returns_per_lane[lane], longer_run_avg_vol_per_lane[lane],
                                window=cfg.VOL_BREAKER_WINDOW, multiplier=cfg.VOL_BREAKER_MULTIPLIER):
            vol_halt[lane] = True
            mask[lane, :, :] = False
            mask[lane, :, zero_idx] = True

        if var_cap_breach(trailing_returns_per_lane[lane], var_cap=cfg.VAR_CAP, window=cfg.VAR_WINDOW):
            var_halt[lane] = True
            mask[lane, :, :] = False
            mask[lane, :, zero_idx] = True

        if hard_forced_flatten(bar_idx_per_lane[lane], cfg.BARS_PER_EPISODE):
            forced_flatten[lane] = True
            mask[lane, :, :] = False
            mask[lane, :, zero_idx] = True

    no_legal_action = ~mask.any(axis=2)  # (L, n_pairs)
    if no_legal_action.any():
        mask[no_legal_action] = False
        mask[no_legal_action, zero_idx] = True

    return {
        "action_mask": mask,
        "forced_flatten": forced_flatten,
        "vol_halt": vol_halt,
        "var_halt": var_halt,
        "max_hold_halt": max_hold_halt,
    }
