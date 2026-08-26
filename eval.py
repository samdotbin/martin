"""
eval.py — deflated Sharpe, bootstrap CIs, stress test, residual-exposure check (§15).

The deflated Sharpe ratio needs the number of implicit trials as an input —
log this count in the run manifest as training goes rather than hardcoding a
guess here after the fact (§21 hint). See utils.RunManifest.trial_count().
"""
import numpy as np
import torch

import config as cfg
import rollout as rollout_module


# ---------------------------------------------------------------------------
# Policy evaluation (greedy rollout, no exploration)
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate_policy(env, model, regime_extractor, n_episodes, cfg_module=cfg, deterministic=True):
    model.eval()
    device = next(model.parameters()).device
    daily_returns = []
    residual_exposures = []
    max_drawdowns = []

    n_available = len(env.valid_day_starts)
    n_episodes = min(n_episodes, n_available) if n_available else 0

    for i in range(max(n_episodes, 1)):
        day_start = env.valid_day_starts[i % max(n_available, 1)] if n_available else None
        state = env.reset(day_start)
        done = False
        last_two_bars_exposure = []
        while not done:
            time_feats = rollout_module.build_time_features(state)
            action_mask_np = env.get_action_mask()

            edge_t = torch.as_tensor(state["edge_history"], dtype=torch.float32, device=device).unsqueeze(0)
            pos_t = torch.as_tensor(state["positions"], dtype=torch.float32, device=device).unsqueeze(0)
            regime_t = torch.as_tensor(state["regime_embedding"], dtype=torch.float32, device=device).unsqueeze(0)
            time_t = torch.as_tensor(time_feats, dtype=torch.float32, device=device).unsqueeze(0)
            brn_t = torch.as_tensor([state["bars_remaining_norm"]], dtype=torch.float32, device=device)
            rbu_t = torch.as_tensor([state["risk_budget_used"]], dtype=torch.float32, device=device)
            mask_t = torch.as_tensor(action_mask_np, dtype=torch.bool, device=device).unsqueeze(0)

            out = model(edge_t, pos_t, regime_t, time_t, brn_t, rbu_t, action_mask=mask_t)
            if deterministic:
                action = out["probs"][0].argmax(dim=-1)
            else:
                action = torch.distributions.Categorical(probs=out["probs"][0]).sample()

            if env._bars_remaining_norm() <= cfg_module.CLOSE_OUT_PENALTY_BARS_REMAINING:
                last_two_bars_exposure.append(np.sum(np.abs(env.positions)))

            next_state, reward, done, info = env.step(action.cpu().numpy())
            state = next_state

        daily_returns.append((env.equity - env.equity_0) / env.equity_0)
        max_drawdowns.append(env._max_drawdown)
        if last_two_bars_exposure:
            residual_exposures.append(np.mean(last_two_bars_exposure))

    daily_returns = np.array(daily_returns)
    sharpe = float(np.mean(daily_returns) / (np.std(daily_returns) + 1e-9) * np.sqrt(252)) if len(daily_returns) else 0.0
    win_rate = float(np.mean(daily_returns > 0)) if len(daily_returns) else 0.0

    residual_exposure_metric = 0.0
    if residual_exposures:
        denom = cfg_module.PAIRS.__len__() * max(cfg_module.ACTION_BUCKETS)
        residual_exposure_metric = float(np.mean(residual_exposures) / denom)

    model.train()
    return {
        "sharpe": sharpe,
        "win_rate": win_rate,
        "max_drawdown": float(np.mean(max_drawdowns)) if max_drawdowns else 0.0,
        "cumulative_return": float(np.prod(1 + daily_returns) - 1) if len(daily_returns) else 0.0,
        "residual_exposure": residual_exposure_metric,
        "daily_returns": daily_returns.tolist(),
    }


# ---------------------------------------------------------------------------
# §15 — deflated Sharpe
# ---------------------------------------------------------------------------
def deflated_sharpe_ratio(sharpe: float, n_obs: int, n_trials: int, skew=0.0, kurt=3.0) -> float:
    """
    Probabilistic Sharpe Ratio deflated for the number of implicit trials
    (Bailey & Lopez de Prado). n_trials = configs x folds x seeds — read from
    RunManifest.trial_count() (§15, §16), not guessed.
    """
    from scipy.stats import norm

    if n_trials <= 1 or n_obs <= 1:
        return sharpe

    # Expected max Sharpe under the null across n_trials independent trials.
    euler_gamma = 0.5772156649
    sr_std = np.sqrt((1 - skew * sharpe + (kurt - 1) / 4 * sharpe ** 2) / (n_obs - 1))
    expected_max_sr = sr_std * (
        (1 - euler_gamma) * norm.ppf(1 - 1.0 / n_trials) + euler_gamma * norm.ppf(1 - 1.0 / (n_trials * np.e))
    )
    psr = norm.cdf((sharpe - expected_max_sr) / (sr_std + 1e-12))
    return float(psr)


# ---------------------------------------------------------------------------
# §15 — bootstrap CIs, with the fold-independence caveat from §13
# ---------------------------------------------------------------------------
def bootstrap_ci(daily_returns: np.ndarray, n_boot=2000, ci=0.90):
    if len(daily_returns) == 0:
        return {"low": 0.0, "high": 0.0, "mean": 0.0}
    rng = np.random.default_rng(0)
    sharpes = []
    for _ in range(n_boot):
        sample = rng.choice(daily_returns, size=len(daily_returns), replace=True)
        s = np.mean(sample) / (np.std(sample) + 1e-9) * np.sqrt(252)
        sharpes.append(s)
    lo = np.percentile(sharpes, (1 - ci) / 2 * 100)
    hi = np.percentile(sharpes, (1 + ci) / 2 * 100)
    return {"low": float(lo), "high": float(hi), "mean": float(np.mean(sharpes))}


def block_bootstrap_folds(fold_sharpes: list, n_boot=2000, ci=0.90):
    """
    Resamples WHOLE folds (not individual days) because adjacent folds share
    persistent macro regimes and aren't i.i.d. — treating 8-10 folds as fully
    independent overstates confidence (§13 fold-independence caveat).
    """
    fold_sharpes = np.asarray(fold_sharpes)
    if len(fold_sharpes) == 0:
        return {"low": 0.0, "high": 0.0, "mean": 0.0}
    rng = np.random.default_rng(0)
    means = []
    for _ in range(n_boot):
        sample = rng.choice(fold_sharpes, size=len(fold_sharpes), replace=True)
        means.append(np.mean(sample))
    lo = np.percentile(means, (1 - ci) / 2 * 100)
    hi = np.percentile(means, (1 + ci) / 2 * 100)
    return {"low": float(lo), "high": float(hi), "mean": float(np.mean(means)),
            "note": "block bootstrap over whole folds — wider/more honest than per-day i.i.d. CI (§13)"}


# ---------------------------------------------------------------------------
# §15 — benchmarks
# ---------------------------------------------------------------------------
def benchmark_always_flat(n_days: int) -> dict:
    """Equity stays at initial cash — the primary baseline (§15)."""
    returns = np.zeros(n_days)
    return {"sharpe": 0.0, "win_rate": 0.0, "cumulative_return": 0.0, "daily_returns": returns.tolist()}


def benchmark_buy_and_hold(edge_features: np.ndarray, pair_idx: int, day_starts: np.ndarray,
                            bars_per_episode=cfg.BARS_PER_EPISODE) -> dict:
    """
    Equal-weight or single-pair buy-and-hold. Holds overnight, which the
    strategy structurally cannot — informational only, not the primary bar (§15).
    """
    daily_returns = []
    for start in day_starts:
        end = min(start + bars_per_episode, edge_features.shape[0] - 1)
        ret = np.sum(edge_features[start:end, pair_idx, 0])  # sum of log returns
        daily_returns.append(ret)
    daily_returns = np.array(daily_returns)
    sharpe = float(np.mean(daily_returns) / (np.std(daily_returns) + 1e-9) * np.sqrt(252)) if len(daily_returns) else 0.0
    return {
        "sharpe": sharpe,
        "win_rate": float(np.mean(daily_returns > 0)) if len(daily_returns) else 0.0,
        "cumulative_return": float(np.sum(daily_returns)),
        "daily_returns": daily_returns.tolist(),
    }


# ---------------------------------------------------------------------------
# §15 — spread stress test
# ---------------------------------------------------------------------------
def stress_test(env, model, regime_extractor, spread_multiplier=2.5, n_episodes=20, cfg_module=cfg):
    """Re-run backtest with spread widened 2-3x to check cost sensitivity (§15)."""
    original_spread = env.edge_features[:, :, 3].copy()
    env.edge_features[:, :, 3] = original_spread * spread_multiplier
    try:
        metrics = evaluate_policy(env, model, regime_extractor, n_episodes, cfg_module)
    finally:
        env.edge_features[:, :, 3] = original_spread
    metrics["spread_multiplier"] = spread_multiplier
    return metrics
