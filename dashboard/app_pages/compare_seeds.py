import pandas as pd
import streamlit as st

import charts
from lib import (
    config,
    get_test_day_starts,
    list_available_checkpoints,
    load_edge_features_and_timestamps,
    load_price_csv,
    replay_day,
)

checkpoints = list_available_checkpoints()
if not checkpoints:
    st.info(
        "No checkpoints found yet in checkpoints/ — come back once at least "
        "two seeds of the same fold have finished training."
    )
    st.stop()

folds_with_multiple_seeds = sorted({
    f for f in {fid for fid, _ in checkpoints}
    if len([s for fid, s in checkpoints if fid == f]) > 1
})
if not folds_with_multiple_seeds:
    st.info("Need at least 2 seeds finished for the SAME fold to compare them side by side.")
    st.stop()

edge_features, timestamps = load_edge_features_and_timestamps()

with st.container(horizontal=True):
    fold_id = st.selectbox("Fold", folds_with_multiple_seeds)
    available_seeds = sorted(s for f, s in checkpoints if f == fold_id)
    chosen_seeds = st.multiselect("Seeds to compare", available_seeds, default=available_seeds[:3])

    day_starts = get_test_day_starts(fold_id)
    if len(day_starts) == 0:
        st.warning("No contiguous test days found for this fold.")
        st.stop()
    day_options = {pd.Timestamp(timestamps[int(i)]).date(): int(i) for i in day_starts}
    chosen_date = st.selectbox("Test day", sorted(day_options.keys()))
    pair = st.selectbox("Pair to chart", config.PAIRS, index=config.PAIRS.index("EURUSD"), key="compare_pair")

if len(chosen_seeds) < 2:
    st.info("Pick at least 2 seeds.")
    st.stop()

day_start_idx = day_options[chosen_date]
price_df = load_price_csv(pair)

st.subheader(f"{pair} — {chosen_date} (fold {fold_id})")

replays = {}
summary_rows = []
for seed in chosen_seeds:
    replay_df = replay_day(fold_id, seed, day_start_idx)
    merged = price_df.merge(
        replay_df[["timestamp", "equity", "reward", f"pos_{pair}"]],
        on="timestamp", how="inner",
    )
    if not merged.empty:
        merged = charts.with_bar_labels(merged, fmt="%H:%M")
    replays[seed] = merged
    if not merged.empty:
        summary_rows.append({
            "Seed": seed,
            "Day return": merged["equity"].iloc[-1] - 1.0,
            f"Final {pair} position": merged[f"pos_{pair}"].iloc[-1],
            "Cumulative reward": merged["reward"].sum(),
            "Bars with a position": int((merged[f"pos_{pair}"].abs() > 1e-9).sum()),
        })

with st.container(border=True):
    st.dataframe(
        pd.DataFrame(summary_rows).style.format({
            "Day return": "{:.2%}", f"Final {pair} position": "{:.2f}", "Cumulative reward": "{:.4f}",
        }),
        hide_index=True, width="stretch",
    )

# Shared price chart as a common backdrop — same bar_label ordering for every
# seed since they all replay the identical day, so this stays aligned with
# every position chart below it.
any_merged = next((m for m in replays.values() if not m.empty), None)
if any_merged is not None:
    with st.container(border=True):
        st.caption("Shared price backdrop (same for every seed)")
        candles = charts.candlestick_layers(any_merged)
        st.altair_chart(candles.properties(height=250).interactive(), width="stretch")

st.caption(f"Agent's {pair} position over the day, one row per seed")
for seed, merged in replays.items():
    if merged.empty:
        continue
    with st.container(border=True):
        st.caption(f"seed {seed}")
        st.altair_chart(
            charts.area_chart(merged, f"pos_{pair}", y_title="Position", height=110),
            width="stretch",
        )
