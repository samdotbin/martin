from datetime import datetime, timezone

import pandas as pd
import streamlit as st

from lib import get_training_progress, load_contributor_heartbeats

GITHUB_REPO = "samdotbin/martin"
COLAB_URL = f"https://colab.research.google.com/github/{GITHUB_REPO}/blob/master/colab_train.ipynb"
SHARED_DRIVE_URL = "https://drive.google.com/drive/folders/1KKSoBLl8CurDqFtUz3UM9h_9fmDFDiqe?usp=sharing"

st.markdown(
    f"This project trains a multi-pair FX trading policy via a walk-forward "
    f"sweep — 12 folds x 4 seeds, 48 fold/seed combos total, each needing "
    f"real GPU time. Training is split into shards contributors can run in "
    f"parallel on their own Colab (free or paid) — the more GPUs helping, "
    f"the faster the sweep finishes."
)

heartbeats = load_contributor_heartbeats()

with st.container(horizontal=True):
    df = get_training_progress()
    total = len(df)
    done = int((df["Status"] == "Done").sum()) if total else 0
    n_online = sum(1 for c in heartbeats if c["online"])
    st.metric("Sweep progress", f"{done}/{total}" if total else "-", border=True)
    st.metric("Contributors online now", n_online, border=True)
    st.metric("Total contributors", len(heartbeats), border=True)

st.divider()
st.subheader("How to contribute your GPU")
st.markdown(
    f"""
1. **Get link access to the shared results folder** (one time only):
   [Open the shared Drive folder]({SHARED_DRIVE_URL}) and click
   **Add shortcut to Drive**. This is what lets your session write results
   somewhere everyone else's session can also see.
2. **Open the notebook directly from GitHub** — no download, no upload, no zip:
   [Open in Colab]({COLAB_URL})
3. `Runtime -> Change runtime type -> GPU`
4. `Runtime -> Run all`. First run asks you to allow Drive access — accept
   it. Nothing else to fill in: your name is generated for you and
   remembered for next time, and the notebook clones the code, downloads
   the historical price data, claims 4 free training slots automatically,
   and starts training.
5. Leave the tab open. Your progress shows up on the **Training** tab here
   as it goes. Closing the tab stops your shards — that's fine, `--resume`
   picks them back up if you come back, or someone else's claim eventually
   reassigns them if you don't (see the notebook for how to release shards
   you're done with).

Free Colab GPUs disconnect after a while regardless of what you do — normal,
not something to fix. Just re-run the notebook (`Runtime -> Run all`) when
you're back — same generated name, same shards.

New here? You won't show up below until your notebook's background
auto-push cell has run at least once (every 5 minutes) — give it a few
minutes after `Run all`.
"""
)

st.divider()
st.subheader("Who's contributing")
if not heartbeats:
    st.caption("No contributors recorded yet — be the first.")
else:
    now = datetime.now(timezone.utc)
    rows = []
    for c in sorted(heartbeats, key=lambda c: c["last_seen"], reverse=True):
        mins_ago = (now - datetime.fromisoformat(c["last_seen"])).total_seconds() / 60
        rows.append({
            "Name": c["name"],
            "Status": "Online" if c["online"] else "Offline",
            "Shards": ", ".join(str(s) for s in c["shards"]),
            "Last seen": "just now" if mins_ago < 1 else f"{mins_ago:.0f} min ago",
        })
    display_df = pd.DataFrame(rows)

    def _status_color(row):
        color = "background-color: rgba(38, 166, 154, 0.25)" if row["Status"] == "Online" else ""
        return [color] * len(row)

    st.dataframe(display_df.style.apply(_status_color, axis=1), hide_index=True, width="stretch")

st.divider()
st.caption(f"Code: [github.com/{GITHUB_REPO}](https://github.com/{GITHUB_REPO})")
