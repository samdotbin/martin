"""
scripts/claim_shards.py — self-service shard assignment for multi-contributor
training. Each contributor calls claim() to grab N unclaimed shard indices
out of TOTAL_SHARDS, recorded in a claims.json in the shared results folder
-- nobody needs to be manually handed a range by the project owner.

Each claim record stores WHO claimed a shard and WHEN, so the dashboard's
GPU-status view (see gpu_status()) can show how many contributors are
active and how long each has been running -- not just the bare assignment.

Not perfectly atomic (Drive has no real file-locking), but the
write-then-reread-and-check-for-collision loop below closes almost all of
that window, and the "contributors aren't training at the exact same
moment" assumption this whole setup already relies on closes the rest.
"""
import datetime
import json
import os
import time


def _now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _claim_name(entry) -> str:
    """Claim records are {"name": ..., "claimed_at": ...} — but tolerate a
    bare name string too, in case an older claims.json (before this field
    existed) is still in play."""
    return entry["name"] if isinstance(entry, dict) else entry


def _claim_time(entry):
    return entry.get("claimed_at") if isinstance(entry, dict) else None


def claim(shared_folder: str, total_shards: int, n_wanted: int, name: str, max_attempts: int = 5) -> list:
    """Returns a sorted list of shard indices assigned to `name`. Idempotent
    across re-runs: if `name` already has claims (e.g. re-running this cell,
    or resuming after a disconnect), those are kept (with their original
    claimed_at) and only the shortfall (if any) is claimed fresh."""
    if not name or not name.strip():
        raise ValueError("name must be a non-empty, identifiable string (e.g. your first name)")

    os.makedirs(shared_folder, exist_ok=True)
    claims_path = os.path.join(shared_folder, "claims.json")

    for attempt in range(max_attempts):
        claims = {}
        if os.path.exists(claims_path):
            with open(claims_path) as f:
                try:
                    claims = json.load(f)
                except json.JSONDecodeError:
                    claims = {}

        mine = sorted(int(i) for i, entry in claims.items() if _claim_name(entry) == name)
        shortfall = n_wanted - len(mine)
        if shortfall <= 0:
            return mine[:n_wanted]

        free = sorted(i for i in range(total_shards) if str(i) not in claims)
        new_claims = free[:shortfall]
        if not new_claims:
            raise RuntimeError(
                f"no free shards left (wanted {shortfall} more, 0 available out of "
                f"{total_shards} total) -- ask the project owner to raise TOTAL_SHARDS, "
                f"or check {claims_path} for stale claims to clear."
            )

        claimed_at = _now_iso()
        for i in new_claims:
            claims[str(i)] = {"name": name, "claimed_at": claimed_at}
        with open(claims_path, "w") as f:
            json.dump(claims, f, indent=2, sort_keys=True)

        # Reread after a short delay to catch a near-simultaneous claim by
        # someone else on the same index.
        time.sleep(0.5)
        with open(claims_path) as f:
            reread = json.load(f)
        collisions = [i for i in new_claims if _claim_name(reread.get(str(i))) != name]
        if not collisions:
            return sorted(mine + new_claims)
        print(f"shard claim collision on {collisions} (someone else grabbed it first) — retrying...")

    raise RuntimeError(f"could not claim {n_wanted} shard(s) for '{name}' after {max_attempts} attempts — "
                        f"check {claims_path} manually.")


def release(shared_folder: str, name: str) -> list:
    """Frees every shard currently claimed by `name` (e.g. if you're done
    contributing, or claimed by mistake). Returns the released indices."""
    claims_path = os.path.join(shared_folder, "claims.json")
    if not os.path.exists(claims_path):
        return []
    with open(claims_path) as f:
        claims = json.load(f)
    released = sorted(int(i) for i, entry in claims.items() if _claim_name(entry) == name)
    claims = {i: entry for i, entry in claims.items() if _claim_name(entry) != name}
    with open(claims_path, "w") as f:
        json.dump(claims, f, indent=2, sort_keys=True)
    return released


def gpu_status(shared_folder: str, total_shards: int, online_window_minutes: int = 20) -> list:
    """One row per contributor (grouped by claimed name): their shard
    indices, when they first claimed, when any of their shards last showed
    real activity (newest mtime among that shard's checkpoint/log files),
    and whether that's recent enough to call them 'online' right now.

    No separate heartbeat mechanism — training subprocesses already write
    checkpoints/logs periodically on their own, so their file mtimes ARE
    the activity signal. online_window_minutes should be a bit more than
    one iteration's wall-clock time so a shard between checkpoints doesn't
    read as offline.
    """
    claims_path = os.path.join(shared_folder, "claims.json")
    claims = {}
    if os.path.exists(claims_path):
        with open(claims_path) as f:
            try:
                claims = json.load(f)
            except json.JSONDecodeError:
                claims = {}

    by_name = {}
    for idx_str, entry in claims.items():
        name = _claim_name(entry)
        by_name.setdefault(name, {"name": name, "shards": [], "claimed_at": None})
        by_name[name]["shards"].append(int(idx_str))
        t = _claim_time(entry)
        if t and (by_name[name]["claimed_at"] is None or t < by_name[name]["claimed_at"]):
            by_name[name]["claimed_at"] = t

    now = datetime.datetime.now(datetime.timezone.utc)
    rows = []
    for name, info in by_name.items():
        last_activity = None
        for idx in info["shards"]:
            shard_dir = os.path.join(shared_folder, f"shard{idx}")
            candidates = []
            ckpt_dir = os.path.join(shard_dir, "checkpoints")
            if os.path.isdir(ckpt_dir):
                candidates += [os.path.join(ckpt_dir, f) for f in os.listdir(ckpt_dir)]
            log_path = os.path.join(shard_dir, "train_log.txt")
            if os.path.exists(log_path):
                candidates.append(log_path)
            for path in candidates:
                mtime = datetime.datetime.fromtimestamp(os.path.getmtime(path), tz=datetime.timezone.utc)
                if last_activity is None or mtime > last_activity:
                    last_activity = mtime

        online = bool(last_activity and (now - last_activity) < datetime.timedelta(minutes=online_window_minutes))
        claimed_at = info["claimed_at"]
        duration_minutes = None
        if claimed_at:
            started = datetime.datetime.fromisoformat(claimed_at)
            duration_minutes = round((now - started).total_seconds() / 60, 1)

        rows.append({
            "name": name,
            "shards": sorted(info["shards"]),
            "claimed_at": claimed_at,
            "last_activity": last_activity.isoformat() if last_activity else None,
            "online": online,
            "duration_minutes": duration_minutes,
        })

    rows.sort(key=lambda r: (-r["online"], r["name"]))
    return rows
