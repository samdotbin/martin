"""
rollout.py — episode collection into a buffer, separate from the PPO update.

Separating this from ppo_trainer.py lets the PPO loss be unit-tested on a
fixed, synthetic buffer without needing a live environment (§21).

bars_remaining_norm and risk_budget_used are stored directly in each
transition AT COLLECTION TIME, rather than recomputed later from bar_idx —
`_current_bar` has already advanced by the time you'd recompute, which
quietly shifts every value off by one bar (§21 implementation hint).
"""
import numpy as np
import torch

import config as cfg
import safety


class RolloutBuffer:
    def __init__(self, n_pairs, n_buckets, capacity, device=cfg.DEVICE):
        self.n_pairs = n_pairs
        self.n_buckets = n_buckets
        self.capacity = capacity
        self.device = device
        self.reset()

    def reset(self):
        self.edge_history = []
        self.positions = []
        self.regime_embedding = []
        self.time_features = []
        self.bars_remaining_norm = []
        self.risk_budget_used = []
        self.action_mask = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []
        self.next_bar_returns = []  # aux-loss target, per pair
        self.size = 0

    def add(self, state, action_mask, action, log_prob, value, reward, done, next_bar_return):
        self.edge_history.append(state["edge_history"])
        self.positions.append(state["positions"])
        self.regime_embedding.append(state["regime_embedding_full"])
        self.time_features.append(state["time_features"])
        self.bars_remaining_norm.append(state["bars_remaining_norm"])
        self.risk_budget_used.append(state["risk_budget_used"])
        self.action_mask.append(action_mask)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)
        self.next_bar_returns.append(next_bar_return)
        self.size += 1

    def as_tensors(self):
        def t(x, dtype=torch.float32):
            return torch.as_tensor(np.array(x), dtype=dtype, device=self.device)

        return {
            "edge_history": t(self.edge_history),
            "positions": t(self.positions),
            "regime_embedding": t(self.regime_embedding),
            "time_features": t(self.time_features),
            "bars_remaining_norm": t(self.bars_remaining_norm),
            "risk_budget_used": t(self.risk_budget_used),
            "action_mask": t(self.action_mask, dtype=torch.bool),
            "actions": t(self.actions, dtype=torch.long),
            "log_probs": t(self.log_probs),
            "values": t(self.values),
            "rewards": t(self.rewards),
            "dones": t(self.dones, dtype=torch.float32),
            "next_bar_returns": t(self.next_bar_returns),
        }


def build_time_features(state):
    return np.concatenate([
        [state["time_of_day_sin"], state["time_of_day_cos"]],
        np.eye(5, dtype=np.float32)[min(int(state["day_of_week"]), 4)],
        state["session_flag"],
    ]).astype(np.float32)


@torch.no_grad()
def collect_rollout(envs, model, regime_extractor, n_episodes, cfg_module=cfg):
    """
    Runs n_episodes full episodes total, spread across len(envs) parallel
    "lanes" batched into ONE model forward call per synchronized step. This
    collapses the per-step Python-interpreter-bound cost (model.py's GNN
    loops over all 28 pairs, in Python, on every layer, every call) by
    roughly len(envs) — the dominant cost at batch-size 1, which a faster
    device alone can't fix since single-sample calls don't parallelize.

    envs: list of pre-constructed TradingEnv instances ("lanes"), built ONCE
    per fold and reused across every collect_rollout() call in that fold —
    each TradingEnv.__init__ draws from the global RNG stream, so rebuilding
    lanes every call would burn RNG draws and entangle rollout count with
    lane count.

    Buffer ordering (correctness-critical): each lane's transitions are
    accumulated locally and only flushed into the RolloutBuffer in
    lane-major order once every lane has finished. compute_gae's reversed
    recursion feeds values[t+1] into position t's bootstrap — the only real
    invariant it needs is that each lane's own transitions stay temporally
    contiguous in the buffer. Interleaving by wall-clock step across lanes
    without regrouping would feed one lane's bootstrap from a different
    lane's unrelated next state.
    """
    n_pairs = envs[0].n_pairs
    n_buckets = len(cfg_module.ACTION_BUCKETS)
    buffer = RolloutBuffer(n_pairs, n_buckets, capacity=n_episodes * cfg_module.BARS_PER_EPISODE)

    model.eval()
    device = next(model.parameters()).device

    # Clamp lane count to n_episodes so a small request (e.g. the smoke
    # test's 4 episodes) doesn't overshoot into extra, unwanted episodes.
    n_lanes = min(len(envs), n_episodes)
    active = list(range(n_lanes))  # stable ascending order: Categorical.sample()
                                    # draws are positional in the batch, so a
                                    # nondeterministic lane order would break
                                    # seed reproducibility.
    lane_states = [envs[i].reset() for i in range(n_lanes)]
    lane_transitions = [[] for _ in range(n_lanes)]
    episodes_completed = 0

    while active:
        batch_idx = list(active)  # fixed snapshot for this step
        states = [lane_states[i] for i in batch_idx]

        # Batched safety mask: replaces per-lane env.get_action_mask() calls
        # (each of which re-ran the same O(n_pairs*n_buckets) Python loop
        # internally) with one vectorized call across all active lanes —
        # see safety.apply_safety_layer_batched's docstring. Mathematically
        # identical to calling env.get_action_mask() once per lane
        # (tests/test_safety_batched.py fuzz-verifies this against the
        # original per-lane functions, which TradingEnv.get_action_mask()
        # still uses unchanged everywhere else).
        idxs = [envs[i]._current_bar_absolute_idx() for i in batch_idx]
        positions_batch = np.stack([envs[i].positions for i in batch_idx])
        hold_bars_batch = np.stack([envs[i].hold_bars for i in batch_idx])
        bucket_lots_batch = np.stack([envs[i]._bucket_lots_grid(idxs[row]) for row, i in enumerate(batch_idx)])
        bar_idx_list = [envs[i].bar_idx for i in batch_idx]
        trailing_returns_list = []
        longer_run_vol_list = []
        for i in batch_idx:
            env_i = envs[i]
            vol_hist = np.array(env_i._trailing_returns[-env_i.cfg.VOL_BREAKER_WINDOW * 3:]) \
                if env_i._trailing_returns else np.array([0.0])
            longer_run_vol_list.append(float(np.std(vol_hist)) if len(vol_hist) > 1 else 1e-6)
            trailing_returns_list.append(
                np.array(env_i._trailing_returns) if env_i._trailing_returns else np.array([0.0])
            )
        safety_result = safety.apply_safety_layer_batched(
            bar_idx_list, positions_batch, envs[0].pair_currency_map, envs[0].n_currencies,
            trailing_returns_list, longer_run_vol_list, bucket_lots_batch, hold_bars_batch,
            cfg=cfg_module,
        )
        action_masks_np = list(safety_result["action_mask"])

        time_feats_list = [build_time_features(s) for s in states]
        regime_full_list = [s["regime_embedding"] for s in states]

        edge_t = torch.as_tensor(np.stack([s["edge_history"] for s in states]), dtype=torch.float32, device=device)
        pos_t = torch.as_tensor(np.stack([s["positions"] for s in states]), dtype=torch.float32, device=device)
        regime_t = torch.as_tensor(np.stack(regime_full_list), dtype=torch.float32, device=device)
        time_t = torch.as_tensor(np.stack(time_feats_list), dtype=torch.float32, device=device)
        brn_t = torch.as_tensor([s["bars_remaining_norm"] for s in states], dtype=torch.float32, device=device)
        rbu_t = torch.as_tensor([s["risk_budget_used"] for s in states], dtype=torch.float32, device=device)
        mask_t = torch.as_tensor(np.stack(action_masks_np), dtype=torch.bool, device=device)

        out = model(edge_t, pos_t, regime_t, time_t, brn_t, rbu_t, action_mask=mask_t)
        dist = torch.distributions.Categorical(probs=out["probs"])
        action = dist.sample()  # [B, n_pairs]
        log_prob = dist.log_prob(action).sum(dim=-1)  # [B], joint over independent per-pair heads

        # One readout for the whole batch, not one .item() per lane — a
        # per-lane device sync would quietly undo a chunk of the speedup
        # this batching is meant to buy, especially on a GPU.
        actions_np = action.cpu().numpy()
        log_probs_np = log_prob.cpu().numpy()
        values_np = out["value"].cpu().numpy()

        retire = []
        for row, lane_i in enumerate(batch_idx):
            env = envs[lane_i]
            state = lane_states[lane_i]

            idx = env._current_bar_absolute_idx()
            next_idx = min(idx + 1, env.edge_features.shape[0] - 1)
            next_bar_return = env.edge_features[next_idx, :, 0].copy()  # aux target (§5)

            next_state, reward, done, info = env.step(actions_np[row])

            state_for_buffer = dict(state)
            state_for_buffer["regime_embedding_full"] = regime_full_list[row]
            state_for_buffer["time_features"] = time_feats_list[row]

            lane_transitions[lane_i].append((
                state_for_buffer, action_masks_np[row], actions_np[row],
                float(log_probs_np[row]), float(values_np[row]),
                reward, done, next_bar_return,
            ))

            if done:
                episodes_completed += 1
                if episodes_completed < n_episodes:
                    lane_states[lane_i] = env.reset()
                else:
                    retire.append(lane_i)
            else:
                lane_states[lane_i] = next_state

        if retire:
            active = [i for i in active if i not in retire]

    # Flush lane-major: lane 0's full trajectory, then lane 1's, etc.
    for lane_i in range(n_lanes):
        for transition in lane_transitions[lane_i]:
            buffer.add(*transition)

    model.train()
    return buffer
