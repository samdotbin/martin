"""
scripts/publish_results.py — merge shard results, then commit + push to
GitHub, so the hosted dashboard (Streamlit Community Cloud, watching this
repo) picks up the refresh automatically.

This is the OWNER-side counterpart to colab_train.ipynb's shared-folder
setup: contributors' Colab sessions write to a Drive folder you share with
them (see the notebook's SHARED_FOLDER config); this script pulls that
folder's shard{0..N-1} subfolders into this project's checkpoints/+runs/,
then publishes the result.

Prereqs:
  - This project is a git repo with a "origin" remote already pointing at
    your GitHub repo (git remote add origin <url>).
  - The Drive folder is visible on THIS machine as a normal path — easiest
    via Google Drive for Desktop, which mounts your Drive as a local folder
    that syncs automatically. Point --shared-folder at that.

Usage:
    python scripts/publish_results.py --shared-folder "G:\\My Drive\\forex_rl_v4_shared_results" --total-shards 12
    python scripts/publish_results.py shard_dir_0 shard_dir_1 ...   # or list shard dirs explicitly
    python scripts/publish_results.py --shared-folder ... --total-shards 12 --no-push  # commit only, don't push
"""
import argparse
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as cfg
from merge_shard_results import merge


def _run(cmd, cwd):
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    return result.returncode, result.stdout.strip(), result.stderr.strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("shard_dirs", nargs="*",
                         help="explicit shard directories to merge (alternative to --shared-folder)")
    parser.add_argument("--shared-folder", type=str, default=None,
                         help="path to the shared results folder (contains shard0/, shard1/, ...)")
    parser.add_argument("--total-shards", type=int, default=None,
                         help="required with --shared-folder: how many shard{i} subfolders to look for")
    parser.add_argument("--no-push", action="store_true", help="commit locally but don't push")
    args = parser.parse_args()

    if args.shared_folder:
        if args.total_shards is None:
            parser.error("--total-shards is required when using --shared-folder")
        shard_dirs = [os.path.join(args.shared_folder, f"shard{i}") for i in range(args.total_shards)]
    elif args.shard_dirs:
        shard_dirs = args.shard_dirs
    else:
        parser.error("pass either shard directories directly, or --shared-folder + --total-shards")

    existing = [d for d in shard_dirs if os.path.isdir(d)]
    missing = [d for d in shard_dirs if d not in existing]
    if missing:
        print(f"note: {len(missing)} shard folder(s) don't exist yet (no one's started them) — skipping: {missing}")
    if not existing:
        print("ERROR: none of the shard folders exist. Check the path and that Drive has synced.")
        sys.exit(1)

    n_entries, n_copied, n_overwritten = merge(existing)
    print(f"merged {len(existing)} shard(s) -> {n_entries} manifest entries, {n_copied} checkpoint file(s)"
          + (f" ({n_overwritten} overwrote an existing file)" if n_overwritten else ""))

    root = cfg.PROJECT_ROOT
    rc, out, err = _run(["git", "rev-parse", "--is-inside-work-tree"], cwd=root)
    if rc != 0:
        print(f"ERROR: {root} is not a git repo — run `git init` and add a remote first.")
        sys.exit(1)

    _run(["git", "add", "checkpoints", "runs"], cwd=root)
    # Check what's ACTUALLY staged after `add`, not git status beforehand —
    # a file can show as "modified" pre-add (e.g. a CRLF-normalization
    # false positive on JSON merge()'s always-\n output) yet stage to
    # something byte-identical to HEAD, leaving nothing for commit to do.
    rc, staged, err = _run(["git", "diff", "--cached", "--name-only", "--", "checkpoints", "runs"], cwd=root)
    if not staged.strip():
        print("Nothing changed in checkpoints/ or runs/ since the last publish — nothing to commit.")
        return

    n_files_changed = len(staged.strip().splitlines())
    commit_msg = f"Publish results: {n_entries} fold/seed combos, {n_files_changed} file(s) changed"
    # Scoped to checkpoints/runs explicitly — never sweep in unrelated
    # staged changes from something else going on in the working tree.
    rc, out, err = _run(["git", "commit", "-m", commit_msg, "--", "checkpoints", "runs"], cwd=root)
    print(out or err)
    if rc != 0:
        print("ERROR: git commit failed — see above.")
        sys.exit(1)

    if args.no_push:
        print("Committed locally (--no-push). Run `git push` when you're ready to publish.")
        return

    rc, out, err = _run(["git", "remote", "get-url", "origin"], cwd=root)
    if rc != 0:
        print("No 'origin' remote configured — committed locally, but can't push.")
        print("Run: git remote add origin <your-github-repo-url>, then `git push -u origin master`.")
        sys.exit(1)

    rc, out, err = _run(["git", "push"], cwd=root)
    print(out or err)
    if rc != 0:
        print("ERROR: git push failed — see above (often needs `git push -u origin master` the first time).")
        sys.exit(1)
    print("Published. Streamlit Community Cloud will pick this up and redeploy automatically.")


if __name__ == "__main__":
    main()
