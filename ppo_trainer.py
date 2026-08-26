"""
ppo_trainer.py — PPO update: GAE with gamma=1 (§5, §14), clipped loss, aux loss.

With gamma=1.0, the GAE recursion drops the discount factor entirely:
    A_t = sum_k lambda^k * delta_{t+k}          (no gamma term multiplying delta)
If you start from a copy-pasted GAE implementation, check it doesn't still
multiply by a hardcoded gamma from an infinite-horizon example (§21 hint).
"""
import torch
import torch.nn.functional as F

import config as cfg


def compute_gae(rewards, values, dones, lam=cfg.GAE_LAMBDA):
    """
    rewards, values, dones: 1D tensors, one flat sequence of transitions
    (episodes concatenated; `dones` marks episode boundaries so advantage
    does not leak across episodes). gamma is fixed at 1.0 (§3.3) — see the
    module docstring for why it's omitted from the recursion.
    """
    T = len(rewards)
    advantages = torch.zeros_like(rewards)
    last_gae = 0.0
    # bootstrap value after the final transition of the whole buffer is 0
    # (each episode terminates with a real terminal reward already folded in).
    next_value = 0.0
    for t in reversed(range(T)):
        next_non_terminal = 1.0 - dones[t]
        delta = rewards[t] + next_value * next_non_terminal - values[t]  # gamma=1, no discount
        last_gae = delta + lam * next_non_terminal * last_gae
        advantages[t] = last_gae
        next_value = values[t]
    returns = advantages + values
    return advantages, returns


def aux_weight_schedule(epoch: int, warmup_epochs=cfg.AUX_WARMUP_EPOCHS,
                         start=cfg.AUX_WEIGHT_START, end=cfg.AUX_WEIGHT_END) -> float:
    """Warm c3 from 0.01 -> 0.1 over the first ~50 epochs (§5)."""
    if epoch >= warmup_epochs:
        return end
    frac = epoch / max(warmup_epochs, 1)
    return start + frac * (end - start)


def ppo_update(model, optimizer, buffer_tensors, epoch: int, cfg_module=cfg):
    """
    Runs PPO_EPOCHS_PER_BATCH passes over the buffer in MINI_BATCH_SIZE chunks.
    Returns a dict of averaged loss components for logging.
    """
    rewards = buffer_tensors["rewards"]
    values = buffer_tensors["values"]
    dones = buffer_tensors["dones"]
    advantages, returns = compute_gae(rewards, values, dones, lam=cfg_module.GAE_LAMBDA)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    N = rewards.shape[0]
    c3 = aux_weight_schedule(epoch)
    device = rewards.device

    stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "aux_loss": 0.0, "n_updates": 0}

    for _ in range(cfg_module.PPO_EPOCHS_PER_BATCH):
        perm = torch.randperm(N, device=device)
        for start in range(0, N, cfg_module.MINI_BATCH_SIZE):
            idx = perm[start:start + cfg_module.MINI_BATCH_SIZE]
            if len(idx) == 0:
                continue

            out = model(
                buffer_tensors["edge_history"][idx],
                buffer_tensors["positions"][idx],
                buffer_tensors["regime_embedding"][idx],
                buffer_tensors["time_features"][idx],
                buffer_tensors["bars_remaining_norm"][idx],
                buffer_tensors["risk_budget_used"][idx],
                action_mask=buffer_tensors["action_mask"][idx],
            )
            dist = torch.distributions.Categorical(probs=out["probs"])
            actions = buffer_tensors["actions"][idx]
            new_log_probs = dist.log_prob(actions).sum(dim=-1)  # joint over 28 independent heads
            entropy = dist.entropy().sum(dim=-1).mean()

            old_log_probs = buffer_tensors["log_probs"][idx]
            ratio = torch.exp(new_log_probs - old_log_probs)

            batch_adv = advantages[idx]
            clip = cfg_module.CLIP_RANGE
            surr1 = ratio * batch_adv
            surr2 = torch.clamp(ratio, 1 - clip, 1 + clip) * batch_adv
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = F.mse_loss(out["value"], returns[idx])

            aux_target = buffer_tensors["next_bar_returns"][idx] * 100.0  # §5 scaling
            aux_loss = F.mse_loss(out["aux_pred"] * 100.0, aux_target)

            loss = (
                policy_loss
                + cfg_module.ENTROPY_COEF * (-entropy)  # c2 * L_entropy (maximize entropy)
                + 0.5 * value_loss  # c1
                + c3 * aux_loss
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()

            stats["policy_loss"] += policy_loss.item()
            stats["value_loss"] += value_loss.item()
            stats["entropy"] += entropy.item()
            stats["aux_loss"] += aux_loss.item()
            stats["n_updates"] += 1

    n = max(stats["n_updates"], 1)
    for k in ("policy_loss", "value_loss", "entropy", "aux_loss"):
        stats[k] /= n
    stats["aux_weight"] = c3
    return stats
