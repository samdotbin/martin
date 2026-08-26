"""
test_rollout_vectorized.py — the lane-batched collect_rollout() must match
the old single-env, one-episode-at-a-time behavior exactly at n_lanes=1, and
must never let one lane's bootstrap leak into another's (§21).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

import config as cfg
import env as env_module
import fold_runner
import rollout as rollout_module
import utils
from model import PolicyValueNet


def _synthetic_universe(n_bars=180):
    pair_currency_map = fold_runner._pair_currency_map()
    n_pairs = len(cfg.PAIRS)
    rng = np.random.default_rng(0)
    edge_features = rng.normal(0, 0.001, size=(n_bars, n_pairs, cfg.EDGE_FEATURES)).astype(np.float32)
    timestamps = np.array(
        np.datetime64("2020-01-06") + np.arange(n_bars) * np.timedelta64(1, "h")
    )
    valid_day_starts = np.array([cfg.LOOKBACK_BARS, cfg.LOOKBACK_BARS + cfg.BARS_PER_EPISODE], dtype=int)
    return edge_features, timestamps, valid_day_starts, pair_currency_map


def _reference_sequential_rollout(env, model, n_episodes, cfg_module=cfg):
    """Hand-rolled reference mirroring the pre-vectorization collect_rollout:
    one env, one episode at a time, one batch-size-1 forward call per bar —
    independent of the production lane-batched implementation."""
    n_pairs = env.n_pairs
    n_buckets = len(cfg_module.ACTION_BUCKETS)
    buffer = rollout_module.RolloutBuffer(n_pairs, n_buckets, capacity=n_episodes * cfg_module.BARS_PER_EPISODE)
    device = next(model.parameters()).device

    with torch.no_grad():
        for _ in range(n_episodes):
            state = env.reset()
            done = False
            while not done:
                action_mask_np = env.get_action_mask()
                time_feats = rollout_module.build_time_features(state)
                regime_full = state["regime_embedding"]

                edge_t = torch.as_tensor(state["edge_history"], dtype=torch.float32, device=device).unsqueeze(0)
                pos_t = torch.as_tensor(state["positions"], dtype=torch.float32, device=device).unsqueeze(0)
                regime_t = torch.as_tensor(regime_full, dtype=torch.float32, device=device).unsqueeze(0)
                time_t = torch.as_tensor(time_feats, dtype=torch.float32, device=device).unsqueeze(0)
                brn_t = torch.as_tensor([state["bars_remaining_norm"]], dtype=torch.float32, device=device)
                rbu_t = torch.as_tensor([state["risk_budget_used"]], dtype=torch.float32, device=device)
                mask_t = torch.as_tensor(action_mask_np, dtype=torch.bool, device=device).unsqueeze(0)

                out = model(edge_t, pos_t, regime_t, time_t, brn_t, rbu_t, action_mask=mask_t)
                probs = out["probs"][0]
                dist = torch.distributions.Categorical(probs=probs)
                action = dist.sample()
                log_prob = dist.log_prob(action).sum()

                idx = env._current_bar_absolute_idx()
                next_idx = min(idx + 1, env.edge_features.shape[0] - 1)
                next_bar_return = env.edge_features[next_idx, :, 0].copy()

                next_state, reward, done, info = env.step(action.cpu().numpy())

                state_for_buffer = dict(state)
                state_for_buffer["regime_embedding_full"] = regime_full
                state_for_buffer["time_features"] = time_feats

                buffer.add(
                    state_for_buffer, action_mask_np, action.cpu().numpy(),
                    float(log_prob.item()), float(out["value"].item()),
                    reward, done, next_bar_return,
                )
                state = next_state
    return buffer


def _build_env(edge_features, timestamps, valid_day_starts, pair_currency_map):
    env = env_module.TradingEnv(edge_features, timestamps, valid_day_starts, pair_currency_map, cfg=cfg)
    env.set_regime(np.zeros(cfg.N_REGIME_COMPONENTS + cfg.N_REGIME_CLUSTERS, dtype=np.float32))
    return env


def test_single_lane_matches_sequential_reference():
    edge_features, timestamps, valid_day_starts, pair_currency_map = _synthetic_universe()
    n_episodes = 3

    utils.set_seed(123)
    env_ref = _build_env(edge_features, timestamps, valid_day_starts, pair_currency_map)
    model_ref = PolicyValueNet(pair_currency_map)
    ref_buffer = _reference_sequential_rollout(env_ref, model_ref, n_episodes)

    utils.set_seed(123)
    env_vec = _build_env(edge_features, timestamps, valid_day_starts, pair_currency_map)
    model_vec = PolicyValueNet(pair_currency_map)
    vec_buffer = rollout_module.collect_rollout([env_vec], model_vec, None, n_episodes, cfg)

    ref_t = ref_buffer.as_tensors()
    vec_t = vec_buffer.as_tensors()

    assert ref_t["actions"].shape == vec_t["actions"].shape
    assert torch.equal(ref_t["actions"], vec_t["actions"]), \
        "n_lanes=1 must sample the exact same actions as the old sequential loop under the same seed"
    assert torch.equal(ref_t["dones"], vec_t["dones"])
    assert torch.allclose(ref_t["rewards"], vec_t["rewards"], atol=1e-5)
    assert torch.allclose(ref_t["log_probs"], vec_t["log_probs"], atol=1e-5)
    assert torch.allclose(ref_t["values"], vec_t["values"], atol=1e-5)
    assert torch.allclose(ref_t["bars_remaining_norm"], vec_t["bars_remaining_norm"], atol=1e-6)


def test_multi_lane_never_leaks_across_lane_boundaries():
    """Every done=1 transition must be immediately followed (if any transition
    follows at all) by a fresh episode start (bars_remaining_norm == 1.0) —
    true whether that next transition continues the same lane (auto-reset)
    or belongs to the next lane's first episode (lane-major flush). A cross-
    lane leak in the buffer's ordering would show up as a done=1 NOT followed
    by a reset, since it would instead be followed by mid-episode data from
    a different lane's already-in-progress trajectory."""
    edge_features, timestamps, valid_day_starts, pair_currency_map = _synthetic_universe()
    utils.set_seed(7)
    lanes = [
        _build_env(edge_features, timestamps, valid_day_starts, pair_currency_map)
        for _ in range(3)
    ]
    model = PolicyValueNet(pair_currency_map)
    buffer = rollout_module.collect_rollout(lanes, model, None, n_episodes=9, cfg_module=cfg)

    tensors = buffer.as_tensors()
    dones = tensors["dones"]
    brn = tensors["bars_remaining_norm"]
    T = len(dones)

    assert T > 0
    done_positions = (dones == 1).nonzero(as_tuple=True)[0].tolist()
    assert len(done_positions) >= 9, "must have collected at least the requested number of episodes"
    for t in done_positions:
        if t + 1 < T:
            assert abs(float(brn[t + 1]) - 1.0) < 1e-6, (
                f"position {t} is a done transition but position {t + 1} doesn't start a "
                f"fresh episode (bars_remaining_norm={float(brn[t + 1])}) — buffer ordering "
                f"leaked across a lane/episode boundary"
            )


if __name__ == "__main__":
    test_single_lane_matches_sequential_reference()
    test_multi_lane_never_leaks_across_lane_boundaries()
    print("test_rollout_vectorized.py: all tests passed")
