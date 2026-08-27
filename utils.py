"""
utils.py — seeding, logging, checkpoint I/O, RUN_MANIFEST.json writer (§16, §21).
"""
import json
import logging
import os
import random
import tempfile
from datetime import datetime, timezone

import numpy as np
import torch

import config


def _atomic_write(path, write_fn):
    """Writes via a temp file in the same directory + os.replace(), so a
    concurrent reader (the notebook's background auto-push thread reads
    these same checkpoint files every few minutes, on a Drive-mounted path
    where writes are slower than local disk) never sees a partially-written
    file. os.replace() is atomic on both POSIX and Windows, same filesystem
    guaranteed by using the target's own directory for the temp file."""
    directory = os.path.dirname(path) or "."
    fd, tmp_path = tempfile.mkstemp(dir=directory, prefix=".tmp_")
    try:
        with os.fdopen(fd, "wb") as f:
            write_fn(f)
        os.replace(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def set_seed(seed: int) -> None:
    """Fix and log a random seed across every source of randomness (§16)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


logger = get_logger(__name__)


def save_checkpoint(policy_state_dict, regime_model, fold_id, seed, out_dir=None, extra=None):
    """
    Checkpoint naming: fold_{i}_seed_{s}_best.pt, saved alongside its matching
    RegimeExtractor pickle (§16, §21) — a checkpoint must never be separated
    from the regime model it was trained with.
    """
    import pickle

    out_dir = out_dir or config.CHECKPOINT_DIR
    os.makedirs(out_dir, exist_ok=True)
    base = f"fold_{fold_id}_seed_{seed}_best"
    policy_path = os.path.join(out_dir, f"{base}.pt")
    regime_path = os.path.join(out_dir, f"{base}_regime.pkl")

    payload = {"policy_state_dict": policy_state_dict, "extra": extra or {}}
    _atomic_write(policy_path, lambda f: torch.save(payload, f))
    _atomic_write(regime_path, lambda f: pickle.dump(regime_model, f))

    return {"policy_path": policy_path, "regime_path": regime_path}


def load_checkpoint(fold_id, seed, in_dir=None):
    import pickle

    in_dir = in_dir or config.CHECKPOINT_DIR
    base = f"fold_{fold_id}_seed_{seed}_best"
    policy_path = os.path.join(in_dir, f"{base}.pt")
    regime_path = os.path.join(in_dir, f"{base}_regime.pkl")

    # weights_only=False explicitly: these are checkpoints this project wrote
    # itself (not third-party downloads), and the payload legitimately needs
    # more than tensors — RNG state (numpy arrays), the regime extractor's
    # extra dict, etc. Newer torch releases default weights_only to True,
    # which raises WeightsUnpickler errors on exactly that content — this
    # bit a live Colab run (numpy._core.multiarray._reconstruct not an
    # allowed global) once the Colab image's torch moved past the version
    # this project was first written against.
    #
    # map_location="cpu", NOT config.DEVICE: torch.load's map_location moves
    # EVERY tensor in the payload, including rng_state["torch"] — a CPU-only
    # ByteTensor from torch.get_rng_state(). Remapping it to "cuda" (as
    # config.DEVICE does on any GPU box) makes torch.set_rng_state() fail
    # with "RNG state must be a torch.ByteTensor" — this bit a live Colab
    # run the moment training actually resumed on its GPU. model/optimizer
    # tensors still land on the right device via load_state_dict()'s own
    # per-tensor copy, which is device-transparent regardless of where the
    # loaded state_dict tensors started out.
    payload = torch.load(policy_path, map_location="cpu", weights_only=False)
    with open(regime_path, "rb") as f:
        regime_model = pickle.load(f)

    return payload["policy_state_dict"], regime_model, payload.get("extra", {})


def _capture_rng_state(lane_envs):
    """Every source of randomness set_seed() fixes, plus each rollout lane's
    own Generator (env.py's TradingEnv._rng, one per lane — NOT covered by
    set_seed since it's a np.random.default_rng() Generator, not the legacy
    global numpy state)."""
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "lanes": [lane._rng.bit_generator.state for lane in lane_envs],
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state, lane_envs):
    """Restore RNG state captured by _capture_rng_state(). Guards against a
    lane-count mismatch (e.g. config.ROLLOUT_N_ENVS was tuned between the
    interrupted run and this one) by skipping ONLY the per-lane restore and
    logging a warning — model/optimizer state (restored separately by the
    caller) still makes this a correct resume, just not bit-for-bit
    identical day-sampling."""
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "torch_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])

    saved_lanes = state.get("lanes", [])
    if len(saved_lanes) != len(lane_envs):
        logger.warning(
            f"resume: saved lane count ({len(saved_lanes)}) != current "
            f"ROLLOUT_N_ENVS ({len(lane_envs)}) — skipping per-lane RNG "
            f"restore. Model/optimizer state is still restored correctly; "
            f"only exact day-sampling reproducibility is forfeited."
        )
        return
    for lane, bit_gen_state in zip(lane_envs, saved_lanes):
        lane._rng.bit_generator.state = bit_gen_state


def save_latest_checkpoint(model, optimizer, iteration, best_val_sharpe, best_state,
                            lane_envs, fold_id, seed, extra=None, out_dir=None):
    """
    Mid-fold checkpoint, saved on the same cadence as fold_runner's periodic
    val-eval — survives an interrupted fold (e.g. a Colab session disconnect)
    without losing more than a few iterations. Distinct from
    save_checkpoint()'s end-of-fold "best" checkpoint: this one is deleted
    once the fold finishes cleanly (see delete_latest_checkpoint). Does NOT
    include the regime extractor — run_fold always re-fits it deterministically
    from train_feats before this point in the function, so it doesn't need
    to survive a restart.
    """
    out_dir = out_dir or config.CHECKPOINT_DIR
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"fold_{fold_id}_seed_{seed}_latest.pt")

    payload = {
        "policy_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "iteration": iteration,
        "best_val_sharpe": best_val_sharpe,
        "best_state": best_state,
        "rng_state": _capture_rng_state(lane_envs),
        "extra": extra or {},
    }
    _atomic_write(path, lambda f: torch.save(payload, f))
    return path


def load_latest_checkpoint_if_exists(fold_id, seed, in_dir=None):
    in_dir = in_dir or config.CHECKPOINT_DIR
    path = os.path.join(in_dir, f"fold_{fold_id}_seed_{seed}_latest.pt")
    if not os.path.exists(path):
        return None
    # weights_only=False, map_location="cpu" — see load_checkpoint()'s
    # comment above. This payload's rng_state (a CPU ByteTensor) is exactly
    # what breaks under weights_only=True and under map_location="cuda" —
    # this is the specific call that crashed on Colab both times.
    return torch.load(path, map_location="cpu", weights_only=False)


def delete_latest_checkpoint(fold_id, seed, out_dir=None):
    out_dir = out_dir or config.CHECKPOINT_DIR
    path = os.path.join(out_dir, f"fold_{fold_id}_seed_{seed}_latest.pt")
    if os.path.exists(path):
        os.remove(path)


class RunManifest:
    """
    Appends one entry per training run to RUN_MANIFEST.json (§16). eval.py reads
    this to compute the trial count that the deflated Sharpe calculation (§15)
    needs — without it, that count is a guess.
    """

    def __init__(self, path=None):
        self.path = path or config.RUN_MANIFEST_PATH
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        if not os.path.exists(self.path):
            with open(self.path, "w") as f:
                json.dump([], f)

    def log_run(self, fold_id, seed, cfg_snapshot, test_metrics):
        with open(self.path, "r") as f:
            entries = json.load(f)
        entries.append(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "fold_id": fold_id,
                "seed": seed,
                "config": cfg_snapshot,
                "test_metrics": test_metrics,
            }
        )
        with open(self.path, "w") as f:
            json.dump(entries, f, indent=2, default=str)

    def trial_count(self):
        """Number of implicit trials (configs x folds x seeds) run so far (§15)."""
        with open(self.path, "r") as f:
            entries = json.load(f)
        return len(entries)

    def all_entries(self):
        with open(self.path, "r") as f:
            return json.load(f)
