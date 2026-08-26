import altair as alt
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
        "one fold has finished training (locally or on Colab)."
    )
    st.stop()

folds = sorted({f for f, s in checkpoints})
edge_features, timestamps = load_edge_features_and_timestamps()

with st.container(horizontal=True):
    fold_id = st.selectbox("Fold", folds)
    seeds = sorted(s for f, s in checkpoints if f == fold_id)
    seed = st.selectbox("Seed", seeds)

    day_starts = get_test_day_starts(fold_id)
    if len(day_starts) == 0:
        st.warning("No contiguous test days found for this fold — the test window may be too short.")
        st.stop()
    day_options = {pd.Timestamp(timestamps[int(i)]).date(): int(i) for i in day_starts}
    chosen_date = st.selectbox("Test day", sorted(day_options.keys()))
    pair = st.selectbox("Pair to chart", config.PAIRS, index=config.PAIRS.index("EURUSD"), key="replay_pair")

day_start_idx = day_options[chosen_date]

replay_df = replay_day(fold_id, seed, day_start_idx)
price_df = load_price_csv(pair)

SAFETY_COLS = ["vol_halt", "var_halt", "max_hold_halt", "forced_flatten"]
merged = price_df.merge(
    replay_df[["timestamp", "equity", "reward", f"pos_{pair}", *SAFETY_COLS]],
    on="timestamp", how="inner",
)

if merged.empty:
    st.warning("No overlapping bars between the replay and this pair's raw price data.")
    st.stop()

# One trading day of H1 bars — hour-level labels are always unique here.
merged = charts.with_bar_labels(merged, fmt="%H:%M")
# vol/var/max-hold halts are notable interventions; forced_flatten fires
# unconditionally on the last bar of every day (§9), so it's excluded here
# to avoid marking something that always happens as if it were unusual.
merged["notable_halt"] = merged[["vol_halt", "var_halt", "max_hold_halt"]].any(axis=1)

with st.container(horizontal=True):
    day_return = merged["equity"].iloc[-1] - 1.0
    st.metric("Day return", f"{day_return:.2%}", border=True)
    st.metric(f"Final {pair} position", f"{merged[f'pos_{pair}'].iloc[-1]:.2f} lots", border=True)
    st.metric("Cumulative reward", f"{merged['reward'].sum():.4f}", border=True)

with st.container(border=True):
    st.subheader(f"{pair} — {chosen_date} (fold {fold_id}, seed {seed})")
    price_layers = charts.candlestick_layers(merged, extra_tooltip=[f"pos_{pair}:Q", "equity:Q", "reward:Q"])
    if merged["notable_halt"].any():
        halt_points = merged[merged["notable_halt"]]
        halt_marks = (
            alt.Chart(halt_points)
            .mark_point(shape="triangle-down", size=120, color="#ffb300", filled=True)
            .encode(
                x=alt.X("bar_label:O", sort=merged["bar_label"].tolist()),
                y=alt.Y("high:Q"),
                tooltip=["timestamp:T", "vol_halt:N", "var_halt:N", "max_hold_halt:N"],
            )
        )
        price_layers = price_layers + halt_marks
        st.caption("Triangle markers show bars where the vol circuit breaker, "
                   "VaR cap, or max-hold check overrode the policy — hover for which one.")
    st.altair_chart(price_layers.properties(height=350).interactive(), width="stretch")

    st.caption(f"Agent's {pair} position over the day (lots — positive long, negative short)")
    st.altair_chart(charts.area_chart(merged, f"pos_{pair}", y_title="Position"), width="stretch")

    st.caption("Equity over the day")
    st.altair_chart(charts.line_chart(merged, "equity"), width="stretch")

with st.container(border=True):
    st.subheader("Net exposure per currency")
    st.caption(
        "MAX_ABS_EXPOSURE (§9) is enforced per CURRENCY, not per pair — this is "
        "the aggregation the safety layer itself acts on, not just this one pair's position."
    )
    exp_cols = [c for c in replay_df.columns if c.startswith("exp_")]
    exp_replay = charts.with_bar_labels(replay_df, fmt="%H:%M")
    exp_long = exp_replay.melt(
        id_vars=["timestamp", "bar_label"], value_vars=exp_cols,
        var_name="Currency", value_name="Net exposure",
    )
    exp_long["Currency"] = exp_long["Currency"].str.removeprefix("exp_")
    st.altair_chart(
        charts.line_chart(exp_long, "Net exposure", color_field="Currency", height=280),
        width="stretch",
    )

with st.expander("All pairs' final positions this day"):
    final_row = replay_df.iloc[-1]
    pos_cols = [c for c in replay_df.columns if c.startswith("pos_")]
    final_positions = pd.DataFrame({
        "Pair": [c.removeprefix("pos_") for c in pos_cols],
        "Position (lots)": [final_row[c] for c in pos_cols],
    }).sort_values("Position (lots)", key=abs, ascending=False)
    st.dataframe(final_positions, hide_index=True, width="stretch")

with st.expander("Raw replay data"):
    st.dataframe(merged.drop(columns=["bar_label"]), hide_index=True, width="stretch")
