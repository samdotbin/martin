import json
import os
import urllib.request
import zipfile

import streamlit as st

st.set_page_config(
    page_title="forex_rl_v4 dashboard",
    page_icon=":material/candlestick_chart:",
    layout="wide",
)

# data/raw's 182MB of historical CSVs is intentionally NOT in git (GitHub's
# free LFS budget can't hold it — see .gitignore) — it's attached to a
# GitHub Release instead. A hosted deployment (Streamlit Community Cloud,
# cloning fresh each time) starts with no data/raw at all, so fetch it once
# per boot before any page tries to read it. Local dev already has the CSVs
# on disk, so this is a no-op there (cheap existence check every rerun).
GITHUB_REPO = "samdotbin/martin"
DATA_RELEASE_TAG = "data-v1"
DATA_ASSET_NAME = "forex_rl_v4_data_raw.zip"
DATA_RELEASE_URL = f"https://github.com/{GITHUB_REPO}/releases/download/{DATA_RELEASE_TAG}/{DATA_ASSET_NAME}"


def _download_public(tmp_zip):
    urllib.request.urlretrieve(DATA_RELEASE_URL, tmp_zip)


def _download_private_via_api(tmp_zip, token):
    """The plain release URL 404s for a private repo when unauthenticated —
    this repo is private. Downloads the asset via the GitHub API instead,
    using a token that lives ONLY in Streamlit Cloud's own Secrets (Settings
    -> Secrets, as GITHUB_TOKEN) — never in this repo, never passed through
    chat. A fine-grained PAT scoped to just this repo with Contents:
    read-only is enough."""
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    release_req = urllib.request.Request(
        f"https://api.github.com/repos/{GITHUB_REPO}/releases/tags/{DATA_RELEASE_TAG}",
        headers=headers,
    )
    with urllib.request.urlopen(release_req) as resp:
        release = json.load(resp)

    asset = next((a for a in release.get("assets", []) if a["name"] == DATA_ASSET_NAME), None)
    if asset is None:
        raise RuntimeError(
            f"Release '{DATA_RELEASE_TAG}' exists but has no asset named '{DATA_ASSET_NAME}' — "
            f"check the exact filename attached to the release."
        )

    asset_req = urllib.request.Request(asset["url"], headers={**headers, "Accept": "application/octet-stream"})
    with urllib.request.urlopen(asset_req) as resp, open(tmp_zip, "wb") as f:
        f.write(resp.read())


def _ensure_data():
    data_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "data", "raw"))
    if os.path.isdir(data_dir) and len([f for f in os.listdir(data_dir) if f.endswith(".csv")]) >= 20:
        return
    os.makedirs(data_dir, exist_ok=True)
    with st.spinner("First boot: downloading historical price data (~43MB, one time only)..."):
        tmp_zip = os.path.join(data_dir, "_download.zip")
        try:
            try:
                _download_public(tmp_zip)
            except urllib.error.HTTPError as e:
                if e.code != 404:
                    raise
                token = st.secrets.get("GITHUB_TOKEN")
                if not token:
                    raise RuntimeError(
                        f"{DATA_RELEASE_URL} returned 404 and no GITHUB_TOKEN is set in this app's "
                        f"Secrets — this repo is private, so the plain release URL only works if "
                        f"the repo is public. Either make the repo public, or add a fine-grained "
                        f"PAT (Contents: read-only, scoped to this repo) as GITHUB_TOKEN in "
                        f"Settings -> Secrets."
                    ) from e
                _download_private_via_api(tmp_zip, token)
            with zipfile.ZipFile(tmp_zip) as z:
                z.extractall(data_dir)
        finally:
            if os.path.exists(tmp_zip):
                os.remove(tmp_zip)


_ensure_data()

page = st.navigation(
    [
        st.Page("app_pages/contribute.py", title="Contribute", icon=":material/volunteer_activism:"),
        st.Page("app_pages/training.py", title="Training", icon=":material/model_training:"),
        st.Page("app_pages/performance.py", title="Performance", icon=":material/bar_chart:"),
        st.Page("app_pages/price_chart.py", title="Price chart", icon=":material/candlestick_chart:"),
        st.Page("app_pages/agent_replay.py", title="Agent replay", icon=":material/smart_toy:"),
        st.Page("app_pages/compare_seeds.py", title="Compare seeds", icon=":material/compare_arrows:"),
    ],
    position="top",
)

st.title(f"{page.icon} {page.title}")
page.run()
