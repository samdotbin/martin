"""
scripts/claim_shards.py — self-service shard assignment for multi-contributor
training. Each contributor calls claim() to grab N unclaimed shard indices
out of TOTAL_SHARDS -- nobody needs to be manually handed a range by the
project owner.

Each shard's claim lives in its OWN file (claims/shard{i}.json), not as an
entry inside one shared claims.json. A single shared file needs a
read-modify-write cycle to update, which on Drive (no real file locking)
can lose updates if two sessions' reads/writes interleave badly -- this bit
for real in production: two contributors both ended up claiming the same 4
shards after enough Drive-folder churn (wrong shortcuts, remounts) put a
stale/incomplete copy of the shared file in front of a writer, silently
dropping an earlier legitimate claim. One file per shard sidesteps that
whole class of bug: claiming shard N is "create claims/shardN.json, fail if
it already exists" -- an atomic, unambiguous primitive, not a
read-then-write race on a file everyone shares.
"""
import datetime
import json
import os


def _now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _claims_dir(shared_folder: str) -> str:
    d = os.path.join(shared_folder, "claims")
    os.makedirs(d, exist_ok=True)
    return d


def _claim_path(shared_folder: str, idx: int) -> str:
    return os.path.join(_claims_dir(shared_folder), f"shard{idx}.json")


def _read_claim(path: str):
    """Returns {"name": ..., "claimed_at": ...} or None if missing/unreadable
    (a corrupt or partially-written file is treated as absent, not as a
    claim -- see the module docstring's note on the tradeoff this makes)."""
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            data = json.load(f)
        return data if isinstance(data, dict) and "name" in data else None
    except (json.JSONDecodeError, OSError):
        return None


def _list_claims(shared_folder: str, total_shards: int) -> dict:
    """{shard_index: {"name": ..., "claimed_at": ...}} for every currently
    claimed shard in [0, total_shards)."""
    out = {}
    for i in range(total_shards):
        entry = _read_claim(_claim_path(shared_folder, i))
        if entry is not None:
            out[i] = entry
    return out


def claim(shared_folder: str, total_shards: int, n_wanted: int, name: str) -> list:
    """Returns a sorted list of shard indices assigned to `name`. Idempotent
    across re-runs: if `name` already has claims (e.g. re-running this cell,
    or resuming after a disconnect), those are kept and only the shortfall
    (if any) is claimed fresh."""
    if not name or not name.strip():
        raise ValueError("name must be a non-empty, identifiable string (e.g. your first name)")

    claims = _list_claims(shared_folder, total_shards)
    mine = sorted(i for i, entry in claims.items() if entry["name"] == name)
    shortfall = n_wanted - len(mine)
    if shortfall <= 0:
        return mine[:n_wanted]

    newly_claimed = []
    for i in (idx for idx in range(total_shards) if idx not in claims):
        if shortfall <= 0:
            break
        path = _claim_path(shared_folder, i)
        try:
            # 'x' = exclusive create: succeeds only if the file does NOT
            # already exist, atomically. Whoever's create call actually
            # lands first wins; the loser gets FileExistsError immediately
            # and just tries the next free index -- no ambiguity, no lost
            # updates, unlike read-modify-write on one shared file.
            with open(path, "x") as f:
                json.dump({"name": name, "claimed_at": _now_iso()}, f)
        except FileExistsError:
            continue  # someone else's create won the race on this index
        newly_claimed.append(i)
        shortfall -= 1

    if shortfall > 0 and not newly_claimed:
        raise RuntimeError(
            f"no free shards left (wanted {n_wanted - len(mine)} more, 0 available out of "
            f"{total_shards} total) -- ask the project owner to raise TOTAL_SHARDS, "
            f"or check {_claims_dir(shared_folder)} for stale claims to clear."
        )

    return sorted(mine + newly_claimed)


def release(shared_folder: str, name: str) -> list:
    """Frees every shard currently claimed by `name` (e.g. if you're done
    contributing, or claimed by mistake). Returns the released indices."""
    claims_dir = _claims_dir(shared_folder)
    released = []
    for fname in os.listdir(claims_dir):
        if not (fname.startswith("shard") and fname.endswith(".json")):
            continue
        path = os.path.join(claims_dir, fname)
        entry = _read_claim(path)
        if entry and entry["name"] == name:
            idx = int(fname[len("shard"):-len(".json")])
            os.remove(path)
            released.append(idx)
    return sorted(released)


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
    claims = _list_claims(shared_folder, total_shards)

    by_name = {}
    for idx, entry in claims.items():
        name = entry["name"]
        by_name.setdefault(name, {"name": name, "shards": [], "claimed_at": None})
        by_name[name]["shards"].append(idx)
        t = entry.get("claimed_at")
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
