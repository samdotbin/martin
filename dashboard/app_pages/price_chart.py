import pandas as pd
import streamlit as st

import charts
from lib import config, load_price_csv

with st.container(horizontal=True):
    pair = st.selectbox("Pair", config.PAIRS, index=config.PAIRS.index("EURUSD"))
    df_full = load_price_csv(pair)
    min_date, max_date = df_full["timestamp"].min().date(), df_full["timestamp"].max().date()
    default_start = max(min_date, max_date - pd.Timedelta(days=30))
    date_range = st.date_input(
        "Date range", value=(default_start, max_date),
        min_value=min_date, max_value=max_date,
    )

if len(date_range) != 2:
    st.info("Pick a start and end date.")
    st.stop()
start, end = date_range

mask = (df_full["timestamp"].dt.date >= start) & (df_full["timestamp"].dt.date <= end)
view = df_full.loc[mask]

if view.empty:
    st.warning("No data in that range.")
    st.stop()

# Always label at full H1 precision so every bar gets its own x category —
# a coarser format (e.g. day-only) would collapse same-day bars onto one
# tick. Vega-Lite's labelOverlap thins which ticks actually render, so wide
# ranges still show readable axis labels without losing bar uniqueness.
view = charts.with_bar_labels(view, fmt="%b %d %Hh")

with st.container(border=True):
    st.subheader(f"{pair} — {start} to {end}")
    candles = charts.candlestick_layers(view, extra_tooltip=["volume:Q", "spread:Q"])
    st.altair_chart(candles.properties(height=450).interactive(), width="stretch")

    st.caption("Volume")
    st.altair_chart(charts.bar_chart(view, "volume", y_title=None), width="stretch")

with st.expander("Raw data"):
    st.dataframe(view.drop(columns=["bar_label"]), hide_index=True, width="stretch")
