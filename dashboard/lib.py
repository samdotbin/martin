"""
dashboard/lib.py — shared, cached data/model loading for the Streamlit app.
Page files (app_pages/*.py) import from here; keep this the only place that
touches the project's core modules so caching stays centralized.
"""
import os
import re
import sys

_DASHBOARD_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_DASHBOARD_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import json
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import streamlit as st
import torch

import config
import data_pipeline
import env as env_module
import eval as eval_module
import fold_runner
import rollout as rollout_module
import utils
from model import PolicyValueNet


@st.cache_data(ttl="20s", show_spinner=False)
def load_manifest_entries():
    """Short TTL (not unbounded) so the Performance page's auto-refresh
    fragment actually picks up new fold/seed results as training finishes
    them, instead of only ever showing what existed when the app started."""
    path = config.RUN_MANIFEST_PATH
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return json.load(f)


@st.cache_data(ttl="20s", show_spinner=False)
def load_gpu_status():
    """Contributor/session status snapshot — written by
    scripts/publish_results.py (via claim_shards.gpu_status()) each time
    the owner publishes. Not live for the hosted dashboard (only as fresh
    as the last publish), same staleness as everything else here."""
    path = os.path.join(config.RUNS_DIR, "gpu_status.json")
    if not os.path.exists(path):
        return {"generated_at": None, "contributors": []}
    with open(path) as f:
        return json.load(f)


@st.cache_data(ttl="20s", show_spinner=False)
def load_contributor_heartbeats(online_window_minutes: int = 4) -> list:
    """Reads contributions/{name}/heartbeat.json directly out of the cloned
    repo — each contributor's Colab session pushes its own (see
    scripts/push_to_github.py::push_heartbeat(), called from the notebook's
    fast status loop on a 60-second cadence, decoupled from the slower
    5-minute checkpoint push). Unlike load_gpu_status(), this needs NO
    owner-side step (no publish_results.py run from a machine with Drive
    access) — the hosted dashboard sees contributors the moment their
    session's first heartbeat lands.
    online_window_minutes should stay a bit above the push cadence so one
    missed cycle (a slow GitHub API call, a brief disconnect) doesn't flip
    someone to 'offline' prematurely."""
    contributions_dir = os.path.join(config.PROJECT_ROOT, "contributions")
    if not os.path.isdir(contributions_dir):
        return []
    now = datetime.now(timezone.utc)
    out = []
    for name in os.listdir(contributions_dir):
        hb_path = os.path.join(contributions_dir, name, "heartbeat.json")
        if not os.path.isfile(hb_path):
            continue
        try:
            with open(hb_path) as f:
                hb = json.load(f)
            last_seen = datetime.fromisoformat(hb["last_seen"])
            online = (now - last_seen) <= timedelta(minutes=online_window_minutes)
            out.append({
                "name": hb.get("name", name),
                "shards": hb.get("shards", []),
                "last_seen": hb["last_seen"],
                "online": online,
                "shard_status": hb.get("shard_status", []),
                "session_started_at": hb.get("session_started_at"),
            })
        except (json.JSONDecodeError, KeyError, ValueError):
            continue  # malformed/partial heartbeat file (e.g. a push caught mid-write) — skip, not fatal
    return out


@st.cache_data(ttl="20s", show_spinner=False)
def get_shard_activity_table() -> pd.DataFrame:
    """One row per shard, across every contributor's heartbeat — answers
    "who is training what, since when, is it alive" directly, instead of
    waiting on the slower checkpoint push (which only reflects a combo
    once its first mid-fold checkpoint has actually been written AND
    pushed). Reads shard_status straight from heartbeat.json, so this is
    only as fresh as the last heartbeat (every 60s) rather than the
    5-minute checkpoint cadence."""
    columns = ["Contributor", "Shard", "Fold", "Seed", "Status", "Last log line", "Since"]
    heartbeats = load_contributor_heartbeats()
    rows = []
    for hb in heartbeats:
        for s in hb.get("shard_status", []):
            fold_id = s.get("fold_id")
            seed = s.get("seed")
            rows.append({
                "Contributor": hb["name"],
                "Shard": s.get("shard"),
                # Stringified, not left as int|"-" — a column mixing raw ints
                # with a "-" placeholder fails pyarrow's Arrow conversion
                # (Streamlit's st.dataframe needs one consistent dtype).
                "Fold": str(fold_id) if fold_id is not None else "-",
                "Seed": str(seed) if seed is not None else "-",
                "Status": s.get("status", "-"),
                "Last log line": s.get("last_log") or "-",
                "Since": hb.get("session_started_at") or "-",
            })
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows).sort_values(["Contributor", "Shard"]).reset_index(drop=True)


@st.cache_data(ttl="30s", show_spinner=False)
def merge_contributions() -> int:
    """Contributors' Colab sessions push their own shard results directly
    to contributions/{name}/shard{i}/ via scripts/push_to_github.py — no
    owner publish step needed. Folds those into the local checkpoints/+runs/
    every other function here already reads from, reusing
    merge_shard_results.merge() unchanged (a contribution dir has the exact
    same checkpoints/+runs/ layout as any other shard dir).

    Runs once per app boot (called from streamlit_app.py) and re-runs on
    any cache miss after the TTL — covers both the common case (Streamlit
    Community Cloud restarts fresh on every push that lands here) and a
    long-lived session that doesn't happen to restart between pushes.
    Returns how many checkpoint files were copied (0 if nothing new).
    """
    contributions_dir = os.path.join(config.PROJECT_ROOT, "contributions")
    if not os.path.isdir(contributions_dir):
        return 0

    shard_dirs = []
    for name in os.listdir(contributions_dir):
        name_dir = os.path.join(contributions_dir, name)
        if not os.path.isdir(name_dir):
            continue
        for shard_name in os.listdir(name_dir):
            shard_dir = os.path.join(name_dir, shard_name)
            if os.path.isdir(shard_dir):
                shard_dirs.append(shard_dir)

    if not shard_dirs:
        return 0

    from scripts.merge_shard_results import merge
    _n_entries, n_copied, _n_overwritten = merge(shard_dirs)
    return n_copied


@st.cache_data(show_spinner="Loading price data...")
def load_price_csv(pair: str) -> pd.DataFrame:
    return data_pipeline.load_raw_csv(pair)


@st.cache_data(show_spinner="Loading and aligning all 28 pairs (cached after first run)...")
def load_edge_features_and_timestamps():
    return data_pipeline.build_edge_feature_tensor()


@st.cache_data(show_spinner=False)
def get_effective_start_year():
    report = data_pipeline.validate_data_coverage()
    return report.effective_start_year


@st.cache_data(show_spinner=False)
def get_fold_configs():
    year = get_effective_start_year()
    if year is None:
        return []
    return data_pipeline.make_fold_configs(year)


def list_available_checkpoints():
    """(fold_id, seed) pairs with a saved fold_{i}_seed_{s}_best.pt checkpoint."""
    ckpt_dir = config.CHECKPOINT_DIR
    if not os.path.isdir(ckpt_dir):
        return []
    out = []
    for fname in os.listdir(ckpt_dir):
        m = re.match(r"fold_(-?\d+)_seed_(-?\d+)_best\.pt$", fname)
        if m:
            out.append((int(m.group(1)), int(m.group(2))))
    return sorted(set(out))


@st.cache_resource(show_spinner="Loading checkpoint...")
def load_model_and_regime(fold_id: int, seed: int):
    pair_currency_map = fold_runner._pair_currency_map()
    policy_state_dict, regime_extractor, extra = utils.load_checkpoint(fold_id, seed)
    regime_dim = regime_extractor.n_components + regime_extractor.n_clusters
    model = PolicyValueNet(pair_currency_map, regime_dim=regime_dim).to(config.DEVICE)
    model.load_state_dict(policy_state_dict)
    model.eval()
    return model, regime_extractor, extra, pair_currency_map


@st.cache_data(show_spinner=False)
def get_test_day_starts(fold_id: int):
    """Absolute-index day starts inside fold_id's TEST window (§10 contiguity)."""
    edge_features, timestamps = load_edge_features_and_timestamps()
    fold_configs = get_fold_configs()
    fold_cfg = fold_configs[fold_id]

    all_day_starts = data_pipeline.contiguous_day_starts(timestamps, config.BARS_PER_EPISODE)
    day_start_ts = timestamps[all_day_starts]

    start_dt = data_pipeline._naive_ns(fold_cfg["test_start"])
    end_dt = data_pipeline._naive_ns(fold_cfg["test_end"])
    ts_idx = pd.DatetimeIndex(day_start_ts)
    if ts_idx.tz is not None:
        ts_idx = ts_idx.tz_convert("UTC").tz_localize(None)
    ts_naive = ts_idx.values.astype("datetime64[ns]")
    mask = (ts_naive >= start_dt) & (ts_naive < end_dt)
    return all_day_starts[mask]


@st.cache_data(show_spinner=False)
def _mean_test_regime_embedding(fold_id: int, seed: int):
    """Same train-only-fit regime embedding fold_runner.py computes for the
    test split: transform (never re-fit) the test window, mean-pool it.
    Shared by replay_day() and new_challenge_env() so a human's challenge
    run sees the IDENTICAL regime context the agent's replay did."""
    _model, regime_extractor, _extra, pair_currency_map = load_model_and_regime(fold_id, seed)
    edge_features, timestamps = load_edge_features_and_timestamps()
    fold_cfg = get_fold_configs()[fold_id]

    test_feats, _ = data_pipeline.load_fold(timestamps, edge_features, fold_cfg["test_start"], fold_cfg["test_end"])
    n_pairs, n_feat = test_feats.shape[1], test_feats.shape[2]
    flat_test = test_feats.reshape(len(test_feats), n_pairs * n_feat)
    test_loadings, test_clusters = regime_extractor.transform(flat_test)
    test_embed = regime_extractor.embedding(test_loadings, test_clusters)
    mean_test_embed = test_embed.mean(axis=0) if len(test_embed) else np.zeros(test_embed.shape[-1])
    return mean_test_embed, pair_currency_map


def _build_replay_env(fold_id: int, seed: int, day_start_idx: int):
    """A fresh TradingEnv configured for one (fold_id, seed)'s test-window
    regime and day — shared by replay_day() and new_challenge_env() so the
    exact same construction (regime embedding, day window) can't drift
    between the two. NOT cached: returns a live, mutable object."""
    edge_features, timestamps = load_edge_features_and_timestamps()
    mean_test_embed, pair_currency_map = _mean_test_regime_embedding(fold_id, seed)
    env = env_module.TradingEnv(
        edge_features, timestamps, np.array([day_start_idx]), pair_currency_map, cfg=config
    )
    env.set_regime(mean_test_embed)
    return env


def new_challenge_env(fold_id: int, seed: int, day_start_idx: int):
    """A fresh TradingEnv for the Arena 'Beat the AI' challenge, configured
    exactly like the trained policy's replay_day() (same regime embedding,
    same day window) — so a human's chosen bucket_indices go through
    env.step() exactly as the agent's would: same vol-scaled lot sizing,
    same safety layer (get_action_mask() reflects the SAME rules), same 28
    pairs. NOT cached: this returns a live, mutable object the caller steps
    bar by bar and holds in st.session_state for one challenge's duration."""
    env = _build_replay_env(fold_id, seed, day_start_idx)
    env.reset(day_start_idx)
    return env


@st.cache_data(show_spinner="Replaying policy on the selected day...")
def replay_day(fold_id: int, seed: int, day_start_idx: int) -> pd.DataFrame:
    """
    Steps the trained (fold_id, seed) policy deterministically through ONE
    trading day and returns a per-bar DataFrame: timestamp, equity, reward,
    and each pair's position size after that bar's action. Mirrors
    eval.py's evaluate_policy() loop for a single day, but keeps full
    per-bar detail instead of collapsing to aggregate metrics — eval.py
    itself is untouched (this lives only here, for visualization).
    """
    model, _regime_extractor, _extra, _pair_currency_map = load_model_and_regime(fold_id, seed)
    env = _build_replay_env(fold_id, seed, day_start_idx)

    device = next(model.parameters()).device
    state = env.reset(day_start_idx)
    pair_names = config.PAIRS
    currency_names = config.CURRENCIES
    records = []
    done = False
    with torch.no_grad():
        while not done:
            time_feats = rollout_module.build_time_features(state)
            action_mask_np = env.get_action_mask()
            # get_action_mask() just ran the full safety layer (§9) for this
            # bar and stashed its verdict — capture it so the replay can show
            # exactly when/why the safety layer overrode the policy, not just
            # what position resulted.
            safety_result = env._last_safety_result

            edge_t = torch.as_tensor(state["edge_history"], dtype=torch.float32, device=device).unsqueeze(0)
            pos_t = torch.as_tensor(state["positions"], dtype=torch.float32, device=device).unsqueeze(0)
            regime_t = torch.as_tensor(state["regime_embedding"], dtype=torch.float32, device=device).unsqueeze(0)
            time_t = torch.as_tensor(time_feats, dtype=torch.float32, device=device).unsqueeze(0)
            brn_t = torch.as_tensor([state["bars_remaining_norm"]], dtype=torch.float32, device=device)
            rbu_t = torch.as_tensor([state["risk_budget_used"]], dtype=torch.float32, device=device)
            mask_t = torch.as_tensor(action_mask_np, dtype=torch.bool, device=device).unsqueeze(0)

            out = model(edge_t, pos_t, regime_t, time_t, brn_t, rbu_t, action_mask=mask_t)
            action = out["probs"][0].argmax(dim=-1)  # deterministic replay

            idx = env._current_bar_absolute_idx()
            ts = pd.Timestamp(env.timestamps[idx])

            next_state, reward, done, info = env.step(action.cpu().numpy())

            record = {
                "timestamp": ts, "equity": env.equity, "reward": reward,
                "vol_halt": bool(safety_result["vol_halt"]),
                "var_halt": bool(safety_result["var_halt"]),
                "max_hold_halt": bool(safety_result["max_hold_halt"]),
                "forced_flatten": bool(safety_result["forced_flatten"]),
            }
            for i, pair in enumerate(pair_names):
                record[f"pos_{pair}"] = float(env.positions[i])

            # Net exposure PER CURRENCY (not per pair) — the same aggregation
            # the safety layer's exposure cap actually enforces (§9), mirrors
            # env._get_state()'s currency_exposure computation.
            currency_exposure = np.zeros(len(currency_names))
            for i, (base_idx, quote_idx) in enumerate(env.pair_currency_map):
                currency_exposure[base_idx] += env.positions[i]
                currency_exposure[quote_idx] -= env.positions[i]
            for i, currency in enumerate(currency_names):
                record[f"exp_{currency}"] = float(currency_exposure[i])

            records.append(record)
            state = next_state

    return pd.DataFrame(records)


@st.cache_data(show_spinner=False)
def get_benchmark_comparison(fold_id: int, benchmark_pair: str = "EURUSD"):
    """Always-flat and buy-and-hold baselines (eval.py) over fold_id's test
    window, for the SAME benchmark_pair — the 'did this beat doing nothing
    or naive buy-and-hold' comparison eval.py already computes but nothing
    previously surfaced."""
    edge_features, timestamps = load_edge_features_and_timestamps()
    day_starts = get_test_day_starts(fold_id)
    pair_idx = config.PAIRS.index(benchmark_pair)
    buy_and_hold = eval_module.benchmark_buy_and_hold(edge_features, pair_idx, day_starts, config.BARS_PER_EPISODE)
    always_flat = eval_module.benchmark_always_flat(len(day_starts))
    return {"buy_and_hold": buy_and_hold, "always_flat": always_flat, "pair": benchmark_pair}


def run_stress_test(fold_id: int, seed: int, spread_multiplier: float, n_episodes: int):
    """Explicit-action (button-triggered), not cached with a long TTL like
    the read-only views — re-runs the policy with spreads widened
    spread_multiplier x via eval.stress_test() (§15 cost sensitivity), next
    to a normal-spread run over the same days for direct comparison."""
    model, regime_extractor, extra, pair_currency_map = load_model_and_regime(fold_id, seed)
    edge_features, timestamps = load_edge_features_and_timestamps()
    fold_cfg = get_fold_configs()[fold_id]
    test_day_starts = get_test_day_starts(fold_id)

    test_feats, _ = data_pipeline.load_fold(timestamps, edge_features, fold_cfg["test_start"], fold_cfg["test_end"])
    n_pairs, n_feat = test_feats.shape[1], test_feats.shape[2]
    flat_test = test_feats.reshape(len(test_feats), n_pairs * n_feat)
    test_loadings, test_clusters = regime_extractor.transform(flat_test)
    test_embed = regime_extractor.embedding(test_loadings, test_clusters)
    mean_test_embed = test_embed.mean(axis=0) if len(test_embed) else np.zeros(test_embed.shape[-1])

    # stress_test() temporarily mutates env.edge_features then restores it —
    # copy so we're not racing/mutating the shared cached array other pages
    # read from load_edge_features_and_timestamps().
    env = env_module.TradingEnv(edge_features.copy(), timestamps, test_day_starts, pair_currency_map, cfg=config)
    env.set_regime(mean_test_embed)

    normal_metrics = eval_module.evaluate_policy(env, model, regime_extractor, n_episodes, cfg_module=config)
    stressed_metrics = eval_module.stress_test(
        env, model, regime_extractor, spread_multiplier=spread_multiplier, n_episodes=n_episodes, cfg_module=config
    )
    return normal_metrics, stressed_metrics


def is_degenerate_flat(daily_returns) -> bool:
    """True if a run's test-window daily returns are all exactly zero —
    i.e. the policy chose flat/no-trade on every single test day."""
    return len(daily_returns) > 0 and bool(np.allclose(daily_returns, 0.0))


def get_all_combos():
    """Every (fold_id, seed) the CURRENT config.py's full sweep covers —
    the denominator for 'how much of the sweep is actually done', not just
    what happens to have a manifest entry already."""
    fold_configs = get_fold_configs()
    return [(f["fold_id"], seed) for f in fold_configs for seed in config.SEEDS]


@st.cache_data(ttl="20s", show_spinner=False)
def get_training_progress() -> pd.DataFrame:
    """One row per (fold_id, seed) in the current sweep, classified as:
    - Done: has a RUN_MANIFEST.json entry (whatever config it was run with)
    - In progress: no manifest entry, but a mid-fold `_latest.pt` checkpoint
      exists (utils.save_latest_checkpoint, see fold_runner.py) — read its
      stored iteration/best_val_sharpe directly, no log-scraping needed.
    - Not started: neither exists yet.
    Same 20s TTL as load_manifest_entries() so this stays live during training.
    """
    entries = load_manifest_entries()
    done_lookup = {}
    for e in entries:
        key = (e["fold_id"], e["seed"])
        done_lookup.setdefault(key, e)  # first entry wins if duplicates exist

    rows = []
    for fold_id, seed in get_all_combos():
        if (fold_id, seed) in done_lookup:
            entry = done_lookup[(fold_id, seed)]
            n_iter = entry.get("config", {}).get("n_train_iterations")
            rows.append({
                "Fold": fold_id, "Seed": seed, "Status": "Done",
                "Iteration": f"{n_iter}/{n_iter}" if n_iter else "-",
                "Best val Sharpe": np.nan,
                "Test Sharpe": entry["test_metrics"]["sharpe"],
            })
            continue

        latest = utils.load_latest_checkpoint_if_exists(fold_id, seed)
        if latest is not None:
            n_iter = latest.get("extra", {}).get("n_train_iterations")
            it = latest["iteration"]
            rows.append({
                "Fold": fold_id, "Seed": seed, "Status": "In progress",
                "Iteration": f"{it}/{n_iter}" if n_iter else str(it),
                "Best val Sharpe": latest["best_val_sharpe"],
                "Test Sharpe": np.nan,
            })
        else:
            rows.append({
                "Fold": fold_id, "Seed": seed, "Status": "Not started",
                "Iteration": "-", "Best val Sharpe": np.nan, "Test Sharpe": np.nan,
            })

    # Same empty-list pitfall as list_checkpoint_files(): pd.DataFrame([])
    # has no columns at all, so the .astype() calls below (and the caller's
    # df["Status"] reads) would KeyError. Happens if get_all_combos() comes
    # back empty — e.g. a fresh session where data/raw isn't in place yet
    # and validate_data_coverage() found no usable data.
    columns = ["Fold", "Seed", "Status", "Iteration", "Best val Sharpe", "Test Sharpe"]
    if not rows:
        return pd.DataFrame(columns=columns)

    df = pd.DataFrame(rows)
    df["Best val Sharpe"] = df["Best val Sharpe"].astype(float)
    df["Test Sharpe"] = df["Test Sharpe"].astype(float)
    return df


def list_checkpoint_files():
    """Every file in checkpoints/ with size + last-modified, for a quick
    'what's actually on disk right now' sanity check."""
    columns = ["File", "Size (MB)", "Modified"]
    ckpt_dir = config.CHECKPOINT_DIR
    if not os.path.isdir(ckpt_dir):
        return pd.DataFrame(columns=columns)
    rows = []
    for fname in sorted(os.listdir(ckpt_dir)):
        path = os.path.join(ckpt_dir, fname)
        if os.path.isfile(path):
            stat = os.stat(path)
            rows.append({
                "File": fname,
                "Size (MB)": round(stat.st_size / 1e6, 2),
                "Modified": pd.Timestamp(stat.st_mtime, unit="s"),
            })
    # An empty `rows` list builds a DataFrame with NO columns at all (not
    # even ones with no data), so .sort_values("Modified") would KeyError —
    # hit this on a fresh Colab session where checkpoints/ exists (from the
    # zip) but nothing has landed in it yet.
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows).sort_values("Modified", ascending=False).reset_index(drop=True)


@st.cache_data(ttl="20s", show_spinner=False)
def get_contributor_leaderboard() -> pd.DataFrame:
    """Ranks contributors by checkpoint files pushed to their own
    contributions/{name}/ subtree — a simple, always-available proxy for how
    much training work each person's sessions have produced, since
    RUN_MANIFEST.json entries don't carry contributor attribution. Combined
    with load_contributor_heartbeats() for online/shards/last-seen — NOT
    load_gpu_status(), which needs an owner-run publish_results.py step
    that the hosted dashboard has no way to trigger (see fb79b64)."""
    contributions_dir = os.path.join(config.PROJECT_ROOT, "contributions")
    heartbeats = load_contributor_heartbeats()
    by_name = {c["name"]: c for c in heartbeats}

    counts = {}
    if os.path.isdir(contributions_dir):
        for name in os.listdir(contributions_dir):
            name_dir = os.path.join(contributions_dir, name)
            if not os.path.isdir(name_dir):
                continue
            n = 0
            for _root, _dirs, files in os.walk(name_dir):
                n += sum(1 for f in files if f.endswith(".pt"))
            counts[name] = n

    columns = ["Name", "Checkpoints pushed", "Online", "Shards", "Last seen"]
    all_names = set(counts) | set(by_name)
    if not all_names:
        return pd.DataFrame(columns=columns)

    now = datetime.now(timezone.utc)
    rows = []
    for name in all_names:
        c = by_name.get(name)
        last_seen = "-"
        if c:
            mins_ago = (now - datetime.fromisoformat(c["last_seen"])).total_seconds() / 60
            last_seen = "just now" if mins_ago < 1 else f"{mins_ago:.0f} min ago"
        rows.append({
            "Name": name,
            "Checkpoints pushed": counts.get(name, 0),
            "Online": bool(c and c["online"]),
            "Shards": ", ".join(str(s) for s in c["shards"]) if c else "-",
            "Last seen": last_seen,
        })
    return pd.DataFrame(rows).sort_values(
        ["Checkpoints pushed", "Online"], ascending=[False, False]
    ).reset_index(drop=True)


def get_best_run():
    """The single (fold_id, seed) manifest entry with the highest test
    Sharpe so far, or None if nothing has finished yet — used to headline
    the Arena page's Charts tab with the system's best result to date."""
    entries = load_manifest_entries()
    if not entries:
        return None
    return max(entries, key=lambda e: e["test_metrics"]["sharpe"])


def get_training_config_summary() -> dict:
    """Read-only snapshot of the config.py knobs that actually affect a
    training run — so you can sanity-check what's running without opening
    the file. Not cached: config.py doesn't change while the app is up."""
    return {
        "N_FOLDS": config.N_FOLDS, "SEEDS": config.SEEDS,
        "TRAIN_YEARS_PER_FOLD": config.TRAIN_YEARS_PER_FOLD,
        "TEST_YEARS_PER_FOLD": config.TEST_YEARS_PER_FOLD,
        "EPISODES_PER_ROLLOUT": config.EPISODES_PER_ROLLOUT,
        "ROLLOUT_N_ENVS": config.ROLLOUT_N_ENVS,
        "LEARNING_RATE": config.LEARNING_RATE,
        "PPO_EPOCHS_PER_BATCH": config.PPO_EPOCHS_PER_BATCH,
        "MINI_BATCH_SIZE": config.MINI_BATCH_SIZE,
        "AUX_WARMUP_EPOCHS": config.AUX_WARMUP_EPOCHS,
        "DEVICE": config.DEVICE,
        "CHECKPOINT_DIR": config.CHECKPOINT_DIR,
        "RUNS_DIR": config.RUNS_DIR,
    }


def merge_shards(shard_dirs: list):
    """Thin wrapper around scripts/merge_shard_results.merge() so the
    Training page can trigger a merge without leaving the browser."""
    import importlib
    import sys as _sys

    scripts_dir = os.path.join(_PROJECT_ROOT, "scripts")
    if scripts_dir not in _sys.path:
        _sys.path.insert(0, scripts_dir)
    merge_module = importlib.import_module("merge_shard_results")
    return merge_module.merge(shard_dirs)
