"""
scripts/push_to_github.py — direct contributor-side push: each Colab
session uploads its OWN shard results straight to GitHub via the REST API,
into a namespaced contributions/{name}/shard{i}/ path so concurrent pushes
from different contributors never conflict on the same file (each person
only ever touches their own subtree).

Needs a fine-grained PAT with Contents: read-and-write on just this repo.
The project owner creates ONE and drops it as plain text in the shared
Drive folder (see colab_train.ipynb) so every contributor's session can
read it with zero per-person setup — "press and run". Note this token's
write scope is the WHOLE repo (GitHub's fine-grained PATs aren't
path-restricted), not just contributions/ — anyone with access to read it
could in principle write anywhere in the repo. Only share the folder
holding it with people you'd trust with that.
"""
import base64
import json
import os
import tempfile
import urllib.error
import urllib.request
from datetime import datetime, timezone


def _api_request(method, url, token, payload=None):
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req) as resp:
            body = resp.read()
            return resp.status, json.loads(body) if body else None
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return 404, None
        raise RuntimeError(f"GitHub API {method} {url} -> {e.code}: {e.read().decode(errors='replace')}") from e


def push_file(repo: str, token: str, repo_path: str, local_path: str, branch: str = "master") -> bool:
    """Uploads/updates ONE file at repo_path with local_path's content.
    Returns True if it actually changed something (False if content was
    already identical — GitHub's API still 200s a no-op PUT, so this checks
    first to avoid a pointless commit)."""
    with open(local_path, "rb") as f:
        raw = f.read()
    content_b64 = base64.b64encode(raw).decode()

    get_url = f"https://api.github.com/repos/{repo}/contents/{repo_path}?ref={branch}"
    status, existing = _api_request("GET", get_url, token)
    sha = existing["sha"] if status == 200 else None

    payload = {"message": f"contribute: update {repo_path}", "content": content_b64, "branch": branch}
    if sha:
        payload["sha"] = sha

    put_url = f"https://api.github.com/repos/{repo}/contents/{repo_path}"
    status, _ = _api_request("PUT", put_url, token, payload)
    return status in (200, 201)


def push_shard_dir(repo: str, token: str, name: str, shard_dir: str, shard_idx: int, branch: str = "master") -> list:
    """Pushes ONE shard's checkpoints+runs into contributions/{name}/shard{idx}/.
    Returns the list of repo paths actually written."""
    pushed = []
    for sub in ("checkpoints", "runs"):
        local_sub = os.path.join(shard_dir, sub)
        if not os.path.isdir(local_sub):
            continue
        for fname in os.listdir(local_sub):
            local_path = os.path.join(local_sub, fname)
            if not os.path.isfile(local_path) or fname.startswith("_"):
                continue  # skip transient files (e.g. a stray _download.zip)
            repo_path = f"contributions/{name}/shard{shard_idx}/{sub}/{fname}"
            try:
                if push_file(repo, token, repo_path, local_path, branch):
                    pushed.append(repo_path)
            except RuntimeError as e:
                print(f"  push failed for {repo_path}: {e}")
    return pushed


def push_all(repo: str, token: str, name: str, shared_folder: str, my_shards: list, branch: str = "master") -> list:
    """Pushes every one of my_shards for `name`. Returns all repo paths written."""
    all_pushed = []
    for idx in my_shards:
        shard_dir = os.path.join(shared_folder, f"shard{idx}")
        if not os.path.isdir(shard_dir):
            continue
        all_pushed += push_shard_dir(repo, token, name, shard_dir, idx, branch)
    return all_pushed


def push_heartbeat(repo: str, token: str, name: str, my_shards: list, branch: str = "master") -> bool:
    """Pushes contributions/{name}/heartbeat.json — name, claimed shards,
    and a UTC timestamp. This is what lets the HOSTED dashboard show who's
    online right now with zero owner intervention: the owner's machine
    doesn't need Drive access or to run anything, since the dashboard reads
    this file straight out of the (already-cloned) repo like everything
    else in contributions/. Call this on the same cadence as push_all() —
    a stale last_seen just means the dashboard will show that contributor
    as offline after a few missed cycles, nothing more."""
    payload = {
        "name": name,
        "shards": list(my_shards),
        "last_seen": datetime.now(timezone.utc).isoformat(),
    }
    fd, tmp_path = tempfile.mkstemp(suffix=".json")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f)
        return push_file(repo, token, f"contributions/{name}/heartbeat.json", tmp_path, branch)
    finally:
        os.remove(tmp_path)


def read_shared_token(shared_folder: str, filename: str = "github_token.txt") -> str:
    """Reads the shared write token from the shared Drive folder — the
    project owner drops it there once; every contributor's session picks
    it up automatically, no per-person setup."""
    path = os.path.join(shared_folder, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found — the project owner needs to create a fine-grained GitHub PAT "
            f"(Contents: read and write, scoped to this repo) and save it as plain text at that path."
        )
    with open(path) as f:
        return f.read().strip()
