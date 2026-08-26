"""
scripts/claim_shards.py — self-service shard assignment for multi-contributor
training. Each contributor calls claim() to grab N unclaimed shard indices
out of TOTAL_SHARDS, recorded in a claims.json in the shared results folder
-- nobody needs to be manually handed a range by the project owner.

Not perfectly atomic (Drive has no real file-locking), but the
write-then-reread-and-check-for-collision loop below closes almost all of
that window, and the "contributors aren't training at the exact same
moment" assumption this whole setup already relies on closes the rest.
"""
import json
import os
import time


def claim(shared_folder: str, total_shards: int, n_wanted: int, name: str, max_attempts: int = 5) -> list:
    """Returns a sorted list of shard indices assigned to `name`. Idempotent
    across re-runs: if `name` already has claims (e.g. re-running this cell,
    or resuming after a disconnect), those are kept and only the shortfall
    (if any) is claimed fresh."""
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

        mine = sorted(int(i) for i, who in claims.items() if who == name)
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

        for i in new_claims:
            claims[str(i)] = name
        with open(claims_path, "w") as f:
            json.dump(claims, f, indent=2, sort_keys=True)

        # Reread after a short delay to catch a near-simultaneous claim by
        # someone else on the same index.
        time.sleep(0.5)
        with open(claims_path) as f:
            reread = json.load(f)
        collisions = [i for i in new_claims if reread.get(str(i)) != name]
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
    released = sorted(int(i) for i, who in claims.items() if who == name)
    claims = {i: who for i, who in claims.items() if who != name}
    with open(claims_path, "w") as f:
        json.dump(claims, f, indent=2, sort_keys=True)
    return released
