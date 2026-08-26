"""
train.py — entrypoint: fold harness, calls fold_runner per fold (§13).

mp.Pool(processes=N) parallelizes CPU processes, not GPU compute — on a
single-GPU Colab instance this does NOT give N x training speed and can
cause CUDA context issues across forked processes. On one GPU, folds run
SEQUENTIALLY here (8 folds x 2-4 hrs ~= 1-2 days per the spec's own
estimate, §12/§21). Only reach for torch.multiprocessing with 'spawn' if
multiple GPUs are genuinely available to spread folds across.
"""
import argparse
import json
import os

import config as cfg
import data_pipeline
import fold_runner
import utils

logger = utils.get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Forex multi-pair RL — walk-forward trainer (spec v5)")
    parser.add_argument("--n-folds", type=int, default=cfg.N_FOLDS)
    parser.add_argument("--iterations", type=int, default=100,
                         help="PPO rollout+update iterations per fold (spec suggests 50-100 for convergence, §14)")
    parser.add_argument("--seeds", type=int, nargs="+", default=cfg.SEEDS)
    parser.add_argument("--folds", type=int, nargs="+", default=None,
                         help="run only these fold ids (for smoke-testing one fold at a time)")
    parser.add_argument("--smoke-test", action="store_true",
                         help="tiny run (1 fold, 1 seed, 2 iterations, few episodes) to verify the pipeline end-to-end")
    parser.add_argument("--resume", dest="resume", action="store_true", default=True,
                         help="skip fold/seed combos already completed in RUN_MANIFEST.json, and resume an "
                              "interrupted fold from its mid-fold checkpoint (default: on)")
    parser.add_argument("--no-resume", dest="resume", action="store_false",
                         help="ignore RUN_MANIFEST.json and any mid-fold checkpoints; start every fold from scratch")
    parser.add_argument("--shard-index", type=int, default=0,
                         help="which shard this process runs (0-indexed) — for splitting the fold x seed "
                              "sweep across multiple parallel machines/sessions (e.g. several Colab notebooks). "
                              "Each shard writes its own RUN_MANIFEST.json/checkpoints; merge them afterward with "
                              "scripts/merge_shard_results.py")
    parser.add_argument("--shard-count", type=int, default=1,
                         help="total number of shards. Fold x seed combos are assigned round-robin "
                              "(combos[shard_index::shard_count]) so no shard gets an unlucky run of only the "
                              "expensive folds")
    args = parser.parse_args()
    if not (0 <= args.shard_index < args.shard_count):
        parser.error(f"--shard-index must be in [0, --shard-count={args.shard_count})")

    os.makedirs(cfg.RUNS_DIR, exist_ok=True)
    manifest = utils.RunManifest()

    logger.info("running validate_data_coverage() (§2) — one-time data-engineering step")
    report = data_pipeline.validate_data_coverage()
    for w in report.warnings:
        logger.warning(w)
    if report.effective_start_year is None:
        logger.error("no usable data found — see README §Data before training")
        return
    logger.info(f"effective start year: {report.effective_start_year}")

    fold_configs = data_pipeline.make_fold_configs(report.effective_start_year)
    if args.n_folds:
        fold_configs = fold_configs[: args.n_folds]
    if args.folds:
        fold_configs = [f for f in fold_configs if f["fold_id"] in args.folds]

    seeds = args.seeds
    iterations = args.iterations
    episodes_per_rollout = cfg.EPISODES_PER_ROLLOUT

    if args.smoke_test:
        fold_configs = fold_configs[:1]
        seeds = seeds[:1]
        iterations = 2
        episodes_per_rollout = 4
        logger.info("SMOKE TEST mode: 1 fold, 1 seed, 2 iterations, 4 episodes/rollout")

    cfg_snapshot = {"n_train_iterations": iterations, "episodes_per_rollout": episodes_per_rollout}
    already_done = {}  # (fold_id, seed) -> manifest entry, so a multi-session
                        # resumed run still ends with a complete summary.json
    if args.resume:
        for entry in manifest.all_entries():
            if entry.get("config") == cfg_snapshot:
                already_done[(entry["fold_id"], entry["seed"])] = entry
        if already_done:
            logger.info(f"--resume: {len(already_done)} fold/seed combo(s) already complete "
                        f"in RUN_MANIFEST.json for this config, will be skipped")

    # Round-robin shard assignment (not a contiguous slice) so a shard
    # doesn't end up with, say, every fold's most expensive test window by
    # chance — each shard gets a spread across fold ids and seeds.
    all_combos = [(fold_cfg, seed) for fold_cfg in fold_configs for seed in seeds]
    if args.shard_count > 1:
        all_combos = all_combos[args.shard_index::args.shard_count]
        logger.info(f"shard {args.shard_index}/{args.shard_count}: running {len(all_combos)} of "
                    f"{len(fold_configs) * len(seeds)} fold/seed combos")

    all_results = []
    for fold_cfg, seed in all_combos:
        if (fold_cfg["fold_id"], seed) in already_done:
            logger.info(f"=== fold {fold_cfg['fold_id']} / seed {seed}: already complete, skipping ===")
            entry = already_done[(fold_cfg["fold_id"], seed)]
            all_results.append({
                "fold_id": fold_cfg["fold_id"], "seed": seed,
                "best_val_sharpe": None, "test_metrics": entry["test_metrics"],
            })
            continue
        logger.info(f"=== fold {fold_cfg['fold_id']} / seed {seed} ===")
        result = fold_runner.run_fold(
            fold_id=fold_cfg["fold_id"],
            train_start=fold_cfg["train_start"], train_end=fold_cfg["train_end"],
            test_start=fold_cfg["test_start"], test_end=fold_cfg["test_end"],
            seed=seed, n_train_iterations=iterations,
            episodes_per_rollout=episodes_per_rollout,
            manifest=manifest, resume=args.resume,
        )
        all_results.append(result)

    summary_path = os.path.join(cfg.RUNS_DIR, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"wrote {summary_path}")

    fold_sharpes = [r["test_metrics"]["sharpe"] for r in all_results]
    logger.info(f"fold test Sharpes: {fold_sharpes}")


if __name__ == "__main__":
    main()
