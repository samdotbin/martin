"""
env.py — trading environment with time-awareness (§3). Calls safety.py for
action masks (§9). Gym-like API: reset() / step().

State layout (§4):
    edge_history        (LOOKBACK_BARS, n_pairs, EDGE_FEATURES)
    positions           (n_pairs,)
    currency_exposure   (n_currencies,)
    regime_embedding    (N_REGIME_COMPONENTS,)   -- injected externally, see set_regime()
    bars_remaining_norm scalar   (§3.1)
    risk_budget_used    scalar   (§3.2)
    time_of_day_sin/cos (2,)
    day_of_week         (1,) or one-hot(5)
    session_flag        one-hot(4)  Asian/London/NY/overlap
"""
import numpy as np

import config
import safety


def _session_flag(hour_utc: int) -> np.ndarray:
    # Rough UTC session bands; overlap = London+NY (12-16 UTC).
    onehot = np.zeros(4, dtype=np.float32)  # [Asian, London, NY, Overlap]
    if 12 <= hour_utc < 16:
        onehot[3] = 1.0
    elif 7 <= hour_utc < 16:
        onehot[1] = 1.0
    elif 12 <= hour_utc < 21:
        onehot[2] = 1.0
    else:
        onehot[0] = 1.0
    return onehot


class TradingEnv:
    def __init__(self, edge_features: np.ndarray, timestamps: np.ndarray,
                 valid_day_starts: np.ndarray, pair_currency_map, cfg=config):
        """
        edge_features: (n_bars, n_pairs, EDGE_FEATURES) full-history array
        timestamps:    (n_bars,) datetime64, aligned to edge_features
        valid_day_starts: indices into `timestamps` marking contiguous episode starts
                           (output of data_pipeline.contiguous_days, index-aligned)
        pair_currency_map: list of (base_currency_idx, quote_currency_idx) per pair
        """
        self.edge_features = edge_features
        self.timestamps = timestamps
        self.valid_day_starts = valid_day_starts
        self.pair_currency_map = pair_currency_map
        self.cfg = cfg
        self.n_pairs = edge_features.shape[1]
        self.n_currencies = len(cfg.CURRENCIES)
        self.regime_embedding = np.zeros(cfg.N_REGIME_COMPONENTS, dtype=np.float32)

        # Differential Sharpe running stats (EMA of first/second moment of bar return).
        self._A = 0.0
        self._B = 0.0

        # Seeded off the global RNG state (itself fixed by utils.set_seed()),
        # not np.random.default_rng() with no argument — that draws fresh
        # OS entropy every construction and silently breaks the §16
        # reproducibility promise ("fix and log a random seed ... data
        # shuffling") for episode day-sampling in reset().
        self._rng = np.random.default_rng(np.random.randint(0, 2**31 - 1))
        self.reset()

    def set_regime(self, embedding: np.ndarray):
        """Externally inject the current fold's regime embedding (§11)."""
        self.regime_embedding = np.asarray(embedding, dtype=np.float32)

    # ------------------------------------------------------------------
    def reset(self, day_start_idx: int = None):
        if day_start_idx is None:
            day_start_idx = int(self._rng.choice(self.valid_day_starts))
        self.day_start_idx = day_start_idx
        self.bar_idx = 0  # 0..BARS_PER_EPISODE-1 within the episode
        self.positions = np.zeros(self.n_pairs, dtype=np.float32)
        self.equity = 1.0
        self.equity_0 = 1.0
        self.equity_curve = [1.0]
        self.hold_bars = np.zeros(self.n_pairs, dtype=np.int32)
        self._A, self._B = 0.0, 0.0
        self._trailing_returns = []
        self._max_equity = 1.0
        self._max_drawdown = 0.0
        self.done = False
        return self._get_state()

    # ------------------------------------------------------------------
    def _current_bar_absolute_idx(self) -> int:
        return self.day_start_idx + self.bar_idx

    def _bars_remaining_norm(self) -> float:
        # (24 - current_bar) / 24, in [0, 1] — §3.1
        return (self.cfg.BARS_PER_EPISODE - self.bar_idx) / self.cfg.BARS_PER_EPISODE

    def _risk_budget_used(self) -> float:
        # fraction of the daily stop-loss threshold already spent — §3.2
        drawdown_from_start = abs(self.equity - self.equity_0) / self.equity_0
        return float(min(drawdown_from_start / self.cfg.DAILY_STOP_LOSS, 1.0))

    def _get_state(self) -> dict:
        idx = self._current_bar_absolute_idx()
        lookback_start = idx - self.cfg.LOOKBACK_BARS + 1
        if lookback_start < 0:
            pad = np.zeros((-lookback_start, self.n_pairs, self.cfg.EDGE_FEATURES), dtype=np.float32)
            hist = np.concatenate([pad, self.edge_features[0:idx + 1]], axis=0)
        else:
            hist = self.edge_features[lookback_start:idx + 1]

        currency_exposure = np.zeros(self.n_currencies, dtype=np.float32)
        for i, (b, q) in enumerate(self.pair_currency_map):
            currency_exposure[b] += self.positions[i]
            currency_exposure[q] -= self.positions[i]

        import pandas as pd
        ts_pd = pd.Timestamp(self.timestamps[idx])
        hour = ts_pd.hour
        dow = ts_pd.weekday()
        tod_frac = hour / 24.0

        return {
            "edge_history": hist.astype(np.float32),
            "positions": self.positions.copy(),
            "currency_exposure": currency_exposure,
            "regime_embedding": self.regime_embedding.copy(),
            "bars_remaining_norm": np.float32(self._bars_remaining_norm()),
            "risk_budget_used": np.float32(self._risk_budget_used()),
            "time_of_day_sin": np.float32(np.sin(2 * np.pi * tod_frac)),
            "time_of_day_cos": np.float32(np.cos(2 * np.pi * tod_frac)),
            "day_of_week": np.int64(dow),
            "session_flag": _session_flag(hour),
        }

    # ------------------------------------------------------------------
    def _vol_scale_all_pairs(self, idx: int) -> np.ndarray:
        """(n_pairs,) vol_scale = clip(median_vol_across_pairs / symbol_vol,
        0.25, 4.0) — §8. This value doesn't depend on which bucket is being
        priced, and the median/std only need computing once per (env, idx)
        rather than redundantly recomputed for every (pair, bucket)
        combination — 168 redundant recomputations per lane per step
        (140 for the mask grid + 28 in step()) collapsed to one vectorized
        call each."""
        window = self.edge_features[max(0, idx - 20):idx + 1, :, 0]  # log returns, 20-bar window
        if window.shape[0] < 2:
            return np.ones(self.n_pairs, dtype=np.float64)
        per_pair_vol = np.std(window, axis=0) + 1e-9
        median_vol = np.median(per_pair_vol)
        return np.clip(median_vol / per_pair_vol, *self.cfg.VOL_SCALE_CLIP)

    def _bucket_lots_grid(self, idx: int) -> np.ndarray:
        """(n_pairs, n_buckets) vol-scaled lot size each bucket would produce
        right now (§8) — the exact value env.step() would execute, so the
        safety mask (§9) can't call a bucket "legal" that would actually
        breach a cap once vol_scale is applied."""
        buckets = np.asarray(self.cfg.ACTION_BUCKETS, dtype=np.float64)
        vol_scale = self._vol_scale_all_pairs(idx)  # (n_pairs,)
        return buckets[None, :] * self.cfg.BASE_LOT * vol_scale[:, None]  # (n_pairs, n_buckets)

    def get_action_mask(self) -> np.ndarray:
        """Delegates to safety.py — the same module used at inference time (§9)."""
        vol_hist = np.array(self._trailing_returns[-self.cfg.VOL_BREAKER_WINDOW * 3:]) \
            if self._trailing_returns else np.array([0.0])
        longer_run_vol = float(np.std(vol_hist)) if len(vol_hist) > 1 else 1e-6
        idx = self._current_bar_absolute_idx()
        result = safety.apply_safety_layer(
            bar_idx=self.bar_idx,
            current_positions=self.positions,
            pair_currency_map=self.pair_currency_map,
            n_currencies=self.n_currencies,
            trailing_portfolio_returns=np.array(self._trailing_returns) if self._trailing_returns else np.array([0.0]),
            longer_run_avg_vol=longer_run_vol,
            bucket_lots=self._bucket_lots_grid(idx),
            hold_bars=self.hold_bars,
            cfg=self.cfg,
        )
        self._last_safety_result = result
        return result["action_mask"]

    # ------------------------------------------------------------------
    def step(self, bucket_indices: np.ndarray):
        """
        bucket_indices: (n_pairs,) int indices into cfg.ACTION_BUCKETS, already
        sampled from the masked categorical distribution (mask, don't clip — §8).
        """
        idx = self._current_bar_absolute_idx()
        buckets = np.asarray(self.cfg.ACTION_BUCKETS)
        vol_scale = self._vol_scale_all_pairs(idx)  # (n_pairs,)
        target_lots = (buckets[bucket_indices] * self.cfg.BASE_LOT * vol_scale).astype(np.float32)
        target_lots = np.clip(target_lots, -self.cfg.MAX_PAIR_EXPOSURE, self.cfg.MAX_PAIR_EXPOSURE)

        turnover = float(np.sum(np.abs(target_lots - self.positions)))  # §7

        # Transaction costs: spread on every trade, including forced close.
        spread_costs = self.edge_features[idx, :, 3]  # spread z-score channel as cost proxy
        traded = np.abs(target_lots - self.positions)
        cost = float(np.sum(traded * np.abs(spread_costs)) * 1e-4)

        next_idx = idx + 1
        next_idx = min(next_idx, self.edge_features.shape[0] - 1)
        pair_log_returns = self.edge_features[next_idx, :, 0]
        pnl = float(np.sum(target_lots * pair_log_returns))

        prev_equity = self.equity
        self.equity = prev_equity * (1.0 + pnl) - cost
        bar_return = (self.equity - prev_equity) / prev_equity
        self.equity_curve.append(self.equity)
        self._trailing_returns.append(bar_return)
        self._max_equity = max(self._max_equity, self.equity)
        self._max_drawdown = max(self._max_drawdown, (self._max_equity - self.equity) / self._max_equity)

        # --- Reward (§7): differential Sharpe ratio increment ---
        eta = self.cfg.DIFFERENTIAL_SHARPE_ETA
        eps = self.cfg.DIFF_SHARPE_EPS
        delta_A = bar_return - self._A
        delta_B = bar_return ** 2 - self._B
        denom = max(self._B - self._A ** 2, eps) ** 1.5
        D_t = (self._B * delta_A - 0.5 * self._A * delta_B) / denom
        self._A += eta * delta_A
        self._B += eta * delta_B

        close_penalty = 0.0
        if self._bars_remaining_norm() <= self.cfg.CLOSE_OUT_PENALTY_BARS_REMAINING:
            close_penalty = self.cfg.CLOSE_OUT_MU * float(np.sum(np.abs(target_lots)))

        r_t = D_t - cost - self.cfg.TURNOVER_KAPPA * turnover - close_penalty

        self.positions = target_lots
        self.bar_idx += 1
        self.hold_bars = np.where(np.abs(self.positions) > 1e-9, self.hold_bars + 1, 0)

        terminated = False
        # Cumulative drawdown from the START of the day (equity_0), not just
        # this bar's move — a string of small losing bars must trip the daily
        # stop as surely as one sharp bar (must match _risk_budget_used()'s
        # notion of drawdown-from-start, which this mirrors intentionally).
        stop_loss_hit = (self.equity_0 - self.equity) / self.equity_0 >= self.cfg.DAILY_STOP_LOSS \
            if self.equity < self.equity_0 else False
        max_hold_forced = self.cfg.MAX_HOLD_BARS is not None and np.any(
            self.hold_bars >= self.cfg.MAX_HOLD_BARS
        )

        terminal_reward = 0.0
        info = {"stop_loss_hit": False, "max_hold_forced": bool(max_hold_forced), "turnover": turnover}

        if stop_loss_hit:
            terminated = True
            terminal_reward = self.cfg.STOP_LOSS_TERMINAL_PENALTY
            info["stop_loss_hit"] = True
        elif self.bar_idx >= self.cfg.BARS_PER_EPISODE:
            terminated = True
            total_return = (self.equity - self.equity_0) / self.equity_0
            std_curve = float(np.std(self.equity_curve))
            terminal_reward = (
                total_return
                - self.cfg.LAMBDA_DD * self._max_drawdown
                - self.cfg.LAMBDA_VAR * std_curve
            )

        self.done = terminated
        reward = self.cfg.REWARD_ALPHA * r_t + (self.cfg.REWARD_BETA * terminal_reward if terminated else 0.0)

        state = self._get_state() if not terminated else None
        info["equity"] = self.equity
        info["max_drawdown"] = self._max_drawdown
        return state, float(reward), terminated, info
