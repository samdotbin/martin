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
DATA_RELEASE_URL = "https://github.com/samdotbin/martin/releases/download/data-v1/forex_rl_v4_data_raw.zip"


def _ensure_data():
    data_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "data", "raw"))
    if os.path.isdir(data_dir) and len([f for f in os.listdir(data_dir) if f.endswith(".csv")]) >= 20:
        return
    os.makedirs(data_dir, exist_ok=True)
    with st.spinner("First boot: downloading historical price data (~43MB, one time only)..."):
        tmp_zip = os.path.join(data_dir, "_download.zip")
        try:
            urllib.request.urlretrieve(DATA_RELEASE_URL, tmp_zip)
            with zipfile.ZipFile(tmp_zip) as z:
                z.extractall(data_dir)
        finally:
            if os.path.exists(tmp_zip):
                os.remove(tmp_zip)


_ensure_data()

page = st.navigation(
    [
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
