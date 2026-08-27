import pandas as pd
import streamlit as st

from lib import get_training_progress, load_gpu_status

GITHUB_REPO = "samdotbin/martin"
COLAB_URL = f"https://colab.research.google.com/github/{GITHUB_REPO}/blob/master/colab_train.ipynb"
SHARED_DRIVE_URL = "https://drive.google.com/drive/folders/1KKSoBLl8CurDqFtUz3UM9h_9fmDFDiqe?usp=sharing"

st.markdown(
    f"This project trains a multi-pair FX trading policy via a walk-forward "
    f"sweep — 12 folds x 3 seeds, 36 fold/seed combos total, each needing "
    f"real GPU time. Training is split into shards contributors can run in "
    f"parallel on their own Colab (free or paid) — the more GPUs helping, "
    f"the faster the sweep finishes."
)

with st.container(horizontal=True):
    df = get_training_progress()
    total = len(df)
    done = int((df["Status"] == "Done").sum()) if total else 0
    status = load_gpu_status()
    n_online = sum(1 for c in status["contributors"] if c["online"])
    st.metric("Sweep progress", f"{done}/{total}" if total else "-", border=True)
    st.metric("Contributors online now", n_online, border=True)
    st.metric("Total contributors", len(status["contributors"]), border=True)

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
5. Leave the tab open. Your progress shows up on the **Arena** and
   **Training** tabs here as it goes. Closing the tab stops your shards —
   that's fine, `--resume` picks them back up if you come back, or someone
   else's claim eventually reassigns them if you don't (see the notebook
   for how to release shards you're done with).

Free Colab GPUs disconnect after a while regardless of what you do — normal,
not something to fix. Just re-run the notebook (`Runtime -> Run all`) when
you're back — same generated name, same shards.

While you wait: check the **Arena** tab — leaderboard, charts, and a
"Beat the AI" day-trading challenge to pass the time.
"""
)

st.divider()
st.subheader("Who's contributing")
contributors = status["contributors"]
if not contributors:
    st.caption("No contributors recorded yet — be the first.")
else:
    rows = []
    for c in contributors:
        rows.append({
            "Name": c["name"],
            "Status": "Online" if c["online"] else "Offline",
            "Shards": ", ".join(str(s) for s in c["shards"]),
            "Running for": f"{c['duration_minutes']:.0f} min" if c["duration_minutes"] is not None else "-",
        })
    display_df = pd.DataFrame(rows)

    def _status_color(row):
        color = "background-color: rgba(38, 166, 154, 0.25)" if row["Status"] == "Online" else ""
        return [color] * len(row)

    st.dataframe(display_df.style.apply(_status_color, axis=1), hide_index=True, width="stretch")
    if status["generated_at"]:
        st.caption(f"As of the last publish: {status['generated_at']}")

st.divider()
st.caption(f"Code: [github.com/{GITHUB_REPO}](https://github.com/{GITHUB_REPO})")
