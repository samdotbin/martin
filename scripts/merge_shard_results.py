"""
scripts/merge_shard_results.py — merge RUN_MANIFEST.json, summary.json, and
checkpoints from N parallel shard runs (see train.py --shard-index/--shard-count)
back into THIS project's single runs/ and checkpoints/ directories.

Each shard should have been run via colab_train.ipynb (or plain
`python train.py --shard-index K --shard-count N`) from its own copy of this
project on its own Drive folder. This script does not touch anything remote —
download each shard's `runs/` + `checkpoints/` folders locally first (or
point this at the mounted Drive paths directly), then run it once.

Usage:
    python scripts/merge_shard_results.py /path/to/shard0 /path/to/shard1 ...

Each path should be a directory containing that shard's `runs/` and
`checkpoints/` subdirectories — i.e. a downloaded copy of that shard's
project folder, or just those two subfolders side by side.
"""
import argparse
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as cfg


def _load_manifest(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return json.load(f)


def merge(shard_dirs, dest_root=None):
    dest_root = dest_root or cfg.PROJECT_ROOT
    dest_runs = os.path.join(dest_root, "runs")
    dest_checkpoints = os.path.join(dest_root, "checkpoints")
    os.makedirs(dest_runs, exist_ok=True)
    os.makedirs(dest_checkpoints, exist_ok=True)

    dest_manifest_path = os.path.join(dest_runs, "RUN_MANIFEST.json")

    def _key(entry):
        return (entry["fold_id"], entry["seed"], json.dumps(entry.get("config"), sort_keys=True))

    merged = {_key(e): e for e in _load_manifest(dest_manifest_path)}

    checkpoints_copied = 0
    checkpoints_overwritten = 0
    for shard_dir in shard_dirs:
        shard_runs = os.path.join(shard_dir, "runs")
        shard_checkpoints = os.path.join(shard_dir, "checkpoints")

        for entry in _load_manifest(os.path.join(shard_runs, "RUN_MANIFEST.json")):
            # Same (fold_id, seed, config) should only ever complete in ONE
            # shard by construction (train.py's round-robin split), so a
            # collision here means two shards were run with overlapping
            # --shard-index/--shard-count settings — last one wins, silently.
            merged[_key(entry)] = entry

        if os.path.isdir(shard_checkpoints):
            for fname in os.listdir(shard_checkpoints):
                src = os.path.join(shard_checkpoints, fname)
                dst = os.path.join(dest_checkpoints, fname)
                if os.path.isfile(src):
                    if os.path.exists(dst):
                        checkpoints_overwritten += 1
                    shutil.copy2(src, dst)
                    checkpoints_copied += 1

    merged_entries = list(merged.values())
    with open(dest_manifest_path, "w") as f:
        json.dump(merged_entries, f, indent=2, default=str)

    summary_path = os.path.join(dest_runs, "summary.json")
    summary = [
        {"fold_id": e["fold_id"], "seed": e["seed"], "best_val_sharpe": None, "test_metrics": e["test_metrics"]}
        for e in merged_entries
    ]
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    return len(merged_entries), checkpoints_copied, checkpoints_overwritten


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("shard_dirs", nargs="+",
                         help="one directory per shard, each containing that shard's runs/ and checkpoints/")
    args = parser.parse_args()

    for d in args.shard_dirs:
        if not os.path.isdir(d):
            print(f"ERROR: not a directory: {d}")
            sys.exit(1)

    n_entries, n_copied, n_overwritten = merge(args.shard_dirs)
    print(f"merged {len(args.shard_dirs)} shard(s) -> {n_entries} manifest entries, "
          f"{n_copied} checkpoint file(s) copied into {cfg.CHECKPOINT_DIR}"
          + (f" ({n_overwritten} overwrote an existing file — check for overlapping shard ranges)"
             if n_overwritten else ""))
    print(f"wrote {cfg.RUN_MANIFEST_PATH}")
    print("Now run your usual aggregate/evaluate step (eval.py's deflated_sharpe_ratio / "
          "block_bootstrap_folds) against the merged runs/checkpoints.")


if __name__ == "__main__":
    main()
