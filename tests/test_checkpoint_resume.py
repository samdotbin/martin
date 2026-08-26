"""
test_checkpoint_resume.py — save_latest_checkpoint()/load_latest_checkpoint_if_exists()
must round-trip RNG state (python/numpy/torch/per-lane) and optimizer
momentum exactly, or a resumed fold silently diverges from the run it's
supposed to be continuing (§21).
"""
import os
import random
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

import config as cfg
import env as env_module
import fold_runner
import utils
from model import PolicyValueNet


def _tiny_universe(n_days=10):
    pair_currency_map = fold_runner._pair_currency_map()
    n_pairs = len(cfg.PAIRS)
    n_bars = cfg.LOOKBACK_BARS + n_days * cfg.BARS_PER_EPISODE + 1
    rng = np.random.default_rng(1)
    edge_features = rng.normal(0, 0.001, size=(n_bars, n_pairs, cfg.EDGE_FEATURES)).astype(np.float32)
    timestamps = np.array(
        np.datetime64("2020-01-06") + np.arange(n_bars) * np.timedelta64(1, "h")
    )
    valid_day_starts = np.arange(cfg.LOOKBACK_BARS, cfg.LOOKBACK_BARS + n_days * cfg.BARS_PER_EPISODE,
                                  cfg.BARS_PER_EPISODE, dtype=int)
    return edge_features, timestamps, valid_day_starts, pair_currency_map


def _build_lanes(edge_features, timestamps, valid_day_starts, pair_currency_map, n_lanes=2):
    lanes = []
    for _ in range(n_lanes):
        lane = env_module.TradingEnv(edge_features, timestamps, valid_day_starts, pair_currency_map, cfg=cfg)
        lane.set_regime(np.zeros(cfg.N_REGIME_COMPONENTS + cfg.N_REGIME_CLUSTERS, dtype=np.float32))
        lanes.append(lane)
    return lanes


def test_checkpoint_round_trips_rng_and_optimizer_state():
    tmp_dir = tempfile.mkdtemp()
    try:
        utils.set_seed(55)
        edge_features, timestamps, valid_day_starts, pair_currency_map = _tiny_universe()
        lanes = _build_lanes(edge_features, timestamps, valid_day_starts, pair_currency_map)

        model = PolicyValueNet(pair_currency_map)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)

        # A couple of real optimizer steps so Adam's momentum buffers are non-trivial.
        for _ in range(2):
            loss = sum((p ** 2).sum() for p in model.parameters())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Advance each lane's own RNG a bit before the snapshot, same as mid-training.
        for lane in lanes:
            lane.reset()

        momentum_at_save = {
            k: v["exp_avg"].clone()
            for k, v in optimizer.state_dict()["state"].items()
        }

        utils.save_latest_checkpoint(
            model, optimizer, iteration=5, best_val_sharpe=1.23, best_state=None,
            lane_envs=lanes, fold_id=999, seed=999,
            extra={"n_train_iterations": 50, "episodes_per_rollout": 256},
            out_dir=tmp_dir,
        )

        # Reference draws taken right after the snapshot, from every RNG
        # source restore_rng_state() is supposed to cover.
        ref_python = random.random()
        ref_numpy = float(np.random.rand())
        ref_torch = torch.rand(3).clone()
        ref_lane_days = [lane.reset()["edge_history"].sum() for lane in lanes for _ in range(3)]

        # Simulate a fresh process: new model/optimizer/lanes, then load+restore.
        model2 = PolicyValueNet(pair_currency_map)
        optimizer2 = torch.optim.Adam(model2.parameters(), lr=cfg.LEARNING_RATE)
        lanes2 = _build_lanes(edge_features, timestamps, valid_day_starts, pair_currency_map)

        loaded = utils.load_latest_checkpoint_if_exists(999, 999, in_dir=tmp_dir)
        assert loaded is not None
        assert loaded["iteration"] == 5
        assert loaded["best_val_sharpe"] == 1.23

        model2.load_state_dict(loaded["policy_state_dict"])
        optimizer2.load_state_dict(loaded["optimizer_state_dict"])
        utils.restore_rng_state(loaded["rng_state"], lanes2)

        # Optimizer momentum must match exactly at load time (before any new step).
        momentum_after_load = {
            k: v["exp_avg"] for k, v in optimizer2.state_dict()["state"].items()
        }
        for k in momentum_at_save:
            assert torch.allclose(momentum_at_save[k], momentum_after_load[k]), \
                f"optimizer momentum for param {k} did not round-trip through the checkpoint"

        # Replayed draws must reproduce the reference exactly.
        got_python = random.random()
        got_numpy = float(np.random.rand())
        got_torch = torch.rand(3)
        got_lane_days = [lane.reset()["edge_history"].sum() for lane in lanes2 for _ in range(3)]

        assert got_python == ref_python, "python random state did not restore correctly"
        assert got_numpy == ref_numpy, "numpy legacy random state did not restore correctly"
        assert torch.equal(got_torch, ref_torch), "torch CPU RNG state did not restore correctly"
        assert got_lane_days == ref_lane_days, (
            "per-lane Generator state did not restore correctly — resumed day-sampling "
            "would silently diverge from the interrupted run"
        )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_lane_count_mismatch_warns_but_does_not_crash():
    tmp_dir = tempfile.mkdtemp()
    try:
        utils.set_seed(1)
        edge_features, timestamps, valid_day_starts, pair_currency_map = _tiny_universe()
        lanes = _build_lanes(edge_features, timestamps, valid_day_starts, pair_currency_map, n_lanes=2)
        model = PolicyValueNet(pair_currency_map)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)

        utils.save_latest_checkpoint(
            model, optimizer, iteration=1, best_val_sharpe=0.0, best_state=None,
            lane_envs=lanes, fold_id=998, seed=998, extra={}, out_dir=tmp_dir,
        )
        loaded = utils.load_latest_checkpoint_if_exists(998, 998, in_dir=tmp_dir)

        # Different lane count than what was saved (e.g. ROLLOUT_N_ENVS was tuned).
        mismatched_lanes = _build_lanes(edge_features, timestamps, valid_day_starts, pair_currency_map, n_lanes=4)
        utils.restore_rng_state(loaded["rng_state"], mismatched_lanes)  # must not raise
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_delete_latest_checkpoint_removes_file():
    tmp_dir = tempfile.mkdtemp()
    try:
        utils.set_seed(2)
        edge_features, timestamps, valid_day_starts, pair_currency_map = _tiny_universe()
        lanes = _build_lanes(edge_features, timestamps, valid_day_starts, pair_currency_map, n_lanes=1)
        model = PolicyValueNet(pair_currency_map)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)
        path = utils.save_latest_checkpoint(
            model, optimizer, iteration=0, best_val_sharpe=0.0, best_state=None,
            lane_envs=lanes, fold_id=997, seed=997, extra={}, out_dir=tmp_dir,
        )
        assert os.path.exists(path)
        utils.delete_latest_checkpoint(997, 997, out_dir=tmp_dir)
        assert not os.path.exists(path)
        assert utils.load_latest_checkpoint_if_exists(997, 997, in_dir=tmp_dir) is None
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    test_checkpoint_round_trips_rng_and_optimizer_state()
    test_lane_count_mismatch_warns_but_does_not_crash()
    test_delete_latest_checkpoint_removes_file()
    print("test_checkpoint_resume.py: all tests passed")
