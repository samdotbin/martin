import os
import subprocess
import sys

import pandas as pd
import streamlit as st

from lib import (
    config,
    get_training_config_summary,
    get_training_progress,
    list_checkpoint_files,
    merge_shards,
)


@st.fragment(run_every="30s")
def live_progress():
    """Auto-refreshing sweep status — every (fold, seed) the current
    config.py sweep covers, classified as Done (manifest entry) / In
    progress (mid-fold _latest.pt checkpoint, with its stored iteration
    count) / Not started. This is what's actually happening right now,
    as opposed to the Performance page, which only ever shows finished runs."""
    df = get_training_progress()
    total = len(df)
    done = int((df["Status"] == "Done").sum())
    in_progress = int((df["Status"] == "In progress").sum())
    not_started = total - done - in_progress

    with st.container(horizontal=True):
        st.metric("Sweep progress", f"{done}/{total}", border=True)
        st.metric("In progress", in_progress, border=True)
        st.metric("Not started", not_started, border=True)

    st.progress(done / total if total else 0.0)

    def _status_color(row):
        colors = {
            "Done": "background-color: rgba(38, 166, 154, 0.25)",
            "In progress": "background-color: rgba(255, 179, 0, 0.25)",
            "Not started": "",
        }
        return [colors.get(row["Status"], "")] * len(row)

    # st.dataframe(styler) only honors CSS from Styler.apply/.map — it does
    # NOT use Styler.format()'s na_rep or number formatting for the actual
    # rendered values (those go through Streamlit's own Arrow-based display,
    # which shows raw NaN as "None"). Pre-format into display strings
    # ourselves instead of relying on the Styler for that part.
    display_df = df.copy()
    for col in ("Best val Sharpe", "Test Sharpe"):
        display_df[col] = df[col].map(lambda v: "-" if pd.isna(v) else f"{v:.2f}")

    st.dataframe(
        display_df.style.apply(_status_color, axis=1),
        hide_index=True, width="stretch",
    )


live_progress()

st.divider()
st.subheader("Local training")
st.caption(
    "Launches train.py on THIS machine in the background — same --resume "
    "logic as Colab, so it skips combos already Done above and resumes an "
    "'In progress' one from its last mid-fold checkpoint. Safe to run "
    "alongside Colab or on its own to fill in whatever Colab doesn't get to. "
    "No GPU on this machine, so it's slower per iteration than Colab (roughly "
    "2 hours per fold/seed combo, last measured) — this is a backstop, not a "
    "replacement."
)

if "local_training_procs" not in st.session_state:
    st.session_state.local_training_procs = []
    st.session_state.local_training_n_parallel = 1

running = [e for e in st.session_state.local_training_procs if e["proc"].poll() is None]

col_a, col_b, col_c = st.columns(3)
with col_a:
    n_parallel = st.number_input(
        "Parallel processes", min_value=1, max_value=4, value=1,
        disabled=bool(running),
        help="This machine has 4 CPU cores and no GPU — unlike Colab, more "
             "processes compete for the SAME limited CPU that's already the "
             "bottleneck for one process. Start at 1; only raise it if the "
             "sweep-status table above still shows real progress with more.",
    )
with col_b:
    iterations = st.number_input(
        "Iterations per fold", min_value=2, max_value=200,
        value=100, disabled=bool(running),
    )
with col_c:
    st.write("")
    st.write("")
    start_clicked = st.button("Start local training", type="primary", disabled=bool(running))

if start_clicked:
    new_procs = []
    if n_parallel == 1:
        # No sharding needed for a single process — writes straight to the
        # main checkpoints/runs, so it shows up in the sweep table above
        # with no merge step required.
        os.makedirs(config.RUNS_DIR, exist_ok=True)
        log_path = os.path.join(config.RUNS_DIR, "local_train_log.txt")
        proc = subprocess.Popen(
            [sys.executable, "train.py", "--resume", "--iterations", str(iterations)],
            cwd=config.PROJECT_ROOT,
            stdout=open(log_path, "w"), stderr=subprocess.STDOUT,
        )
        new_procs.append({"idx": 0, "proc": proc, "log": log_path})
    else:
        # >1 process: same per-shard-directory pattern as the Colab notebook
        # (FOREX_RL_CHECKPOINT_DIR/RUNS_DIR overrides), to avoid concurrent
        # processes racing on the same RUN_MANIFEST.json. Use the "Merge
        # shard results" section below once these finish.
        results_root = os.path.join(config.PROJECT_ROOT, "local_training_results")
        for idx in range(n_parallel):
            shard_dir = os.path.join(results_root, f"shard{idx}")
            ckpt_dir = os.path.join(shard_dir, "checkpoints")
            runs_dir = os.path.join(shard_dir, "runs")
            os.makedirs(ckpt_dir, exist_ok=True)
            os.makedirs(runs_dir, exist_ok=True)
            env = os.environ.copy()
            env["FOREX_RL_CHECKPOINT_DIR"] = ckpt_dir
            env["FOREX_RL_RUNS_DIR"] = runs_dir
            log_path = os.path.join(shard_dir, "train_log.txt")
            proc = subprocess.Popen(
                [sys.executable, "train.py", "--resume", "--iterations", str(iterations),
                 "--shard-index", str(idx), "--shard-count", str(n_parallel)],
                cwd=config.PROJECT_ROOT, env=env,
                stdout=open(log_path, "w"), stderr=subprocess.STDOUT,
            )
            new_procs.append({"idx": idx, "proc": proc, "log": log_path, "shard_dir": shard_dir})
    st.session_state.local_training_procs = new_procs
    st.session_state.local_training_n_parallel = n_parallel
    st.rerun()


@st.fragment(run_every="10s")
def local_training_status():
    procs = st.session_state.local_training_procs
    if not procs:
        return

    any_running = False
    for entry in procs:
        rc = entry["proc"].poll()
        if rc is None:
            any_running = True
        status = "running" if rc is None else f"exited (code {rc})"
        st.caption(f"process {entry['idx']} (pid {entry['proc'].pid}): {status}")
        if os.path.exists(entry["log"]):
            with open(entry["log"]) as f:
                tail = f.readlines()[-6:]
            st.code("".join(tail) or "(no output yet)", language=None)

    if st.session_state.local_training_n_parallel > 1 and not any_running:
        st.info("All local shards finished — use 'Merge shard results' below "
                 "(shard dirs are under local_training_results/) to fold them in.")

    if st.button("Stop local training"):
        for entry in procs:
            if entry["proc"].poll() is None:
                entry["proc"].terminate()
        st.session_state.local_training_procs = []
        st.caption("Stopped — at most the last few iterations since the most "
                   "recent mid-fold checkpoint are lost, same as any other interruption.")
        st.rerun()


local_training_status()

st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader("Current config")
    st.caption("What THIS session's config.py currently says a full sweep looks like.")
    cfg_summary = get_training_config_summary()
    st.dataframe(
        pd.DataFrame({"Setting": cfg_summary.keys(), "Value": [str(v) for v in cfg_summary.values()]}),
        hide_index=True, width="stretch",
    )

with col2:
    st.subheader("Checkpoint files on disk")
    files_df = list_checkpoint_files()
    if files_df.empty:
        st.caption("No checkpoint files yet.")
    else:
        st.dataframe(files_df, hide_index=True, width="stretch", height=320)

st.divider()
st.subheader("Contributors auto-publish — one-time setup")
st.caption(
    "Contributors' Colab sessions push their own results straight to GitHub "
    "(contributions/{name}/ — see scripts/push_to_github.py), so this "
    "dashboard picks them up with no manual step on your part. Needs a "
    "fine-grained GitHub PAT (Contents: read and write, scoped to just this "
    "repo) saved as plain text at SHARED_FOLDER/github_token.txt — create "
    "it once and every contributor's session finds it automatically. Note: "
    "GitHub's PAT scopes are repo-wide, not path-restricted, so this token "
    "can write anywhere in the repo, not just contributions/ — only share "
    "the folder holding it with people you'd trust with that."
)

st.divider()
st.subheader("Merge shard results")
st.caption(
    "After downloading multiple Colab shards' runs/+checkpoints/ folders "
    "locally (see scripts/merge_shard_results.py), paste their paths below "
    "(one per line) to fold them into this project's runs/+checkpoints/ — "
    "same as running the script from a terminal."
)
shard_input = st.text_area(
    "Shard directories (one per line)", height=100,
    placeholder="C:\\Users\\you\\Downloads\\shard0\nC:\\Users\\you\\Downloads\\shard1",
)
if st.button("Merge"):
    shard_dirs = [line.strip() for line in shard_input.splitlines() if line.strip()]
    if not shard_dirs:
        st.warning("Paste at least one shard directory path first.")
    else:
        import os
        missing = [d for d in shard_dirs if not os.path.isdir(d)]
        if missing:
            st.error(f"Not a directory: {missing}")
        else:
            with st.spinner("Merging..."):
                n_entries, n_copied, n_overwritten = merge_shards(shard_dirs)
            st.success(
                f"Merged {len(shard_dirs)} shard(s) -> {n_entries} manifest entries, "
                f"{n_copied} checkpoint file(s) copied"
                + (f" ({n_overwritten} overwrote an existing file)" if n_overwritten else "")
            )
            get_training_progress.clear()
            st.rerun()
