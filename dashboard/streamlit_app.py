import streamlit as st

st.set_page_config(
    page_title="forex_rl_v4 dashboard",
    page_icon=":material/candlestick_chart:",
    layout="wide",
)

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
