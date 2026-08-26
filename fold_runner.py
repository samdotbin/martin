"""
fold_runner.py — single-fold orchestration: fit regime -> train -> evaluate (§21).

`fold_runner.py` answers "how does one fold train end-to-end"; train.py
answers "how do many folds run together." This is where most early debugging
time goes — keep it callable and testable on its own before wrapping it in
the multi-fold harness (§21 hint).
"""
import numpy as np
import pandas as pd
import torch

import config as cfg
import data_pipeline
import env as env_module
import eval as eval_module
import regime as regime_module
import rollout as rollout_module
import ppo_trainer
import utils
from model import PolicyValueNet

logger = utils.get_logger(__name__)


def _pair_currency_map(pairs=None, currencies=None):
    pairs = pairs or cfg.PAIRS
    currencies = currencies or cfg.CURRENCIES
    idx = {c: i for i, c in enumerate(currencies)}
    mapping = []
    for pair in pairs:
        base, quote = pair[:3], pair[3:]
        mapping.append((idx[base], idx[quote]))
    return mapping


def run_fold(fold_id, train_start, train_end, test_start, test_end, seed,
             n_train_iterations=50, episodes_per_rollout=None, cfg_module=cfg,
             checkpoint=True, manifest: utils.RunManifest = None, resume=True):
    """
    One fold end-to-end:
      load data -> validate contiguity -> fit regime (train-only) -> PPO
      training loop with val early-stopping (§13) -> evaluate on held-out
      test -> checkpoint policy+regime together -> log to RUN_MANIFEST.
    """
    utils.set_seed(seed)
    episodes_per_rollout = episodes_per_rollout or cfg_module.EPISODES_PER_ROLLOUT
    pair_currency_map = _pair_currency_map()

    logger.info(f"[fold {fold_id}] loading edge features")
    edge_features, timestamps = data_pipeline.build_edge_feature_tensor()

    # Episode day-starts are computed ONCE on the FULL, fold-independent
    # array (not a fold-sliced sub-array). This means the 64-bar lookback
    # for a day near the start of a fold's val/test window can still reach
    # back into real prior-day bars from before that split boundary, rather
    # than zero-padding — matching §10 ("lookback spans the previous day on
    # purpose"). TradingEnv is then built on the full edge_features array
    # with only its valid_day_starts restricted to the split's own dates.
    all_day_starts = data_pipeline.contiguous_day_starts(timestamps, cfg_module.BARS_PER_EPISODE)
    day_start_ts = timestamps[all_day_starts]

    def starts_in_range(start_str, end_str):
        start_dt = data_pipeline._naive_ns(start_str)
        end_dt = data_pipeline._naive_ns(end_str)
        ts_naive = pd.DatetimeIndex(day_start_ts)
        if ts_naive.tz is not None:
            ts_naive = ts_naive.tz_convert("UTC").tz_localize(None)
        ts_naive = ts_naive.values.astype("datetime64[ns]")
        mask = (ts_naive >= start_dt) & (ts_naive < end_dt)
        return all_day_starts[mask]

    train_window_starts = starts_in_range(train_start, train_end)
    if len(train_window_starts) == 0:
        raise ValueError(
            f"[fold {fold_id}] no contiguous episode days found in the "
            f"training window [{train_start}, {train_end})"
        )

    # 80/10/10 chronological split of the TRAINING window's days (§13) — val
    # is used for early-stopping so the 1-year test set is never touched
    # until final eval. Test keeps its own separate date range.
    n_days = len(train_window_starts)
    tr_end = int(n_days * cfg_module.INTERNAL_SPLIT[0])
    val_end = tr_end + int(n_days * cfg_module.INTERNAL_SPLIT[1])
    tr_starts = train_window_starts[:tr_end]
    val_starts = train_window_starts[tr_end:val_end]
    test_starts = starts_in_range(test_start, test_end)

    if len(tr_starts) == 0 or len(val_starts) == 0 or len(test_starts) == 0:
        raise ValueError(
            f"[fold {fold_id}] no contiguous episode days found in one of "
            f"train/val/test — check raw data coverage for this date range"
        )

    # --- Regime pipeline: fit on TRAIN only (§11). This still uses the
    # fold's own sliced bar windows (load_fold) — regime fit/transform
    # leakage is a separate concern from episode-lookback continuity above,
    # and is unaffected by it.
    train_feats, train_ts = data_pipeline.load_fold(timestamps, edge_features, train_start, train_end)
    test_feats, test_ts = data_pipeline.load_fold(timestamps, edge_features, test_start, test_end)

    n_bars = len(train_feats)
    bar_tr_end = int(n_bars * cfg_module.INTERNAL_SPLIT[0])
    bar_val_end = bar_tr_end + int(n_bars * cfg_module.INTERNAL_SPLIT[1])
    n_pairs, n_feat = train_feats.shape[1], train_feats.shape[2]
    flat_tr = train_feats[:bar_tr_end].reshape(bar_tr_end, n_pairs * n_feat)
    flat_val = train_feats[bar_tr_end:bar_val_end].reshape(bar_val_end - bar_tr_end, n_pairs * n_feat)
    flat_test = test_feats.reshape(len(test_feats), n_pairs * n_feat)

    regime_extractor = regime_module.RegimeExtractor(cfg=cfg_module)
    tr_loadings, tr_clusters = regime_extractor.fit(flat_tr)
    val_loadings, val_clusters = regime_extractor.transform(flat_val)
    test_loadings, test_clusters = regime_extractor.transform(flat_test)

    tr_embed = regime_extractor.embedding(tr_loadings, tr_clusters)
    val_embed = regime_extractor.embedding(val_loadings, val_clusters)
    test_embed = regime_extractor.embedding(test_loadings, test_clusters)

    def mean_embed(embed_arr):
        return embed_arr.mean(axis=0) if len(embed_arr) else np.zeros(embed_arr.shape[-1])

    # Envs operate on the FULL edge_features/timestamps array; valid_day_starts
    # restricts which days get sampled as episodes per split, but lookback
    # can still see real bars from before that split's own boundary (§10).
    #
    # Training uses ROLLOUT_N_ENVS parallel lanes, built ONCE here and reused
    # across every collect_rollout() call in this fold — each TradingEnv
    # draws from the global RNG at construction, so rebuilding lanes per
    # call would burn RNG draws and entangle rollout count with lane count.
    n_lanes = cfg_module.ROLLOUT_N_ENVS
    train_lanes = [
        env_module.TradingEnv(edge_features, timestamps, tr_starts, pair_currency_map, cfg=cfg_module)
        for _ in range(n_lanes)
    ]
    for lane in train_lanes:
        lane.set_regime(mean_embed(tr_embed))
    val_env = env_module.TradingEnv(edge_features, timestamps, val_starts, pair_currency_map, cfg=cfg_module)
    val_env.set_regime(mean_embed(val_embed))
    test_env = env_module.TradingEnv(edge_features, timestamps, test_starts, pair_currency_map, cfg=cfg_module)
    test_env.set_regime(mean_embed(test_embed))

    regime_dim = tr_embed.shape[1]
    model = PolicyValueNet(pair_currency_map, regime_dim=regime_dim).to(cfg_module.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg_module.LEARNING_RATE)

    best_val_sharpe = -np.inf
    best_state = None
    start_it = 0

    if resume:
        latest = utils.load_latest_checkpoint_if_exists(fold_id, seed)
        if latest is not None:
            model.load_state_dict(latest["policy_state_dict"])
            optimizer.load_state_dict(latest["optimizer_state_dict"])
            best_val_sharpe = latest["best_val_sharpe"]
            best_state = latest["best_state"]
            start_it = latest["iteration"] + 1
            utils.restore_rng_state(latest["rng_state"], train_lanes)
            logger.info(f"[fold {fold_id}] resuming from iteration {start_it} "
                        f"(latest checkpoint found)")

    logger.info(f"[fold {fold_id}] starting PPO training ({n_train_iterations} iterations, "
                f"{n_lanes} rollout lanes)")
    for it in range(start_it, n_train_iterations):
        buffer = rollout_module.collect_rollout(train_lanes, model, regime_extractor, episodes_per_rollout, cfg_module)
        tensors = buffer.as_tensors()
        stats = ppo_trainer.ppo_update(model, optimizer, tensors, epoch=it, cfg_module=cfg_module)

        if it % max(n_train_iterations // 10, 1) == 0 or it == n_train_iterations - 1:
            val_metrics = eval_module.evaluate_policy(val_env, model, regime_extractor, n_episodes=20, cfg_module=cfg_module)
            logger.info(
                f"[fold {fold_id}] iter {it}: policy_loss={stats['policy_loss']:.4f} "
                f"val_sharpe={val_metrics['sharpe']:.3f} val_winrate={val_metrics['win_rate']:.2%}"
            )
            if val_metrics["sharpe"] > best_val_sharpe:
                best_val_sharpe = val_metrics["sharpe"]
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

            utils.save_latest_checkpoint(
                model, optimizer, it, best_val_sharpe, best_state, train_lanes,
                fold_id, seed,
                extra={"n_train_iterations": n_train_iterations, "episodes_per_rollout": episodes_per_rollout},
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics = eval_module.evaluate_policy(test_env, model, regime_extractor, n_episodes=len(test_starts), cfg_module=cfg_module)
    logger.info(f"[fold {fold_id}] TEST sharpe={test_metrics['sharpe']:.3f} win_rate={test_metrics['win_rate']:.2%}")

    if checkpoint:
        utils.save_checkpoint(
            model.state_dict(), regime_extractor, fold_id, seed,
            extra={"train_start": train_start, "train_end": train_end,
                   "test_start": test_start, "test_end": test_end,
                   "best_val_sharpe": best_val_sharpe},
        )
        # Fold finished cleanly — the mid-fold "latest" checkpoint is now
        # redundant (superseded by the best-checkpoint above).
        utils.delete_latest_checkpoint(fold_id, seed)

    if manifest is not None:
        manifest.log_run(
            fold_id=fold_id, seed=seed,
            cfg_snapshot={"n_train_iterations": n_train_iterations, "episodes_per_rollout": episodes_per_rollout},
            test_metrics=test_metrics,
        )

    return {"fold_id": fold_id, "seed": seed, "best_val_sharpe": best_val_sharpe, "test_metrics": test_metrics}
