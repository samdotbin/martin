import random

import altair as alt
import pandas as pd
import streamlit as st

import charts
from lib import (
    config,
    get_best_run,
    get_contributor_leaderboard,
    get_test_day_starts,
    list_available_checkpoints,
    load_manifest_entries,
    load_price_csv,
    replay_day,
)

MEDALS = ["🥇", "🥈", "🥉"]

tab_leaderboard, tab_charts, tab_challenge = st.tabs(
    ["🏆 Leaderboard", "📈 Charts", "⚔️ Beat the AI"]
)

# ---------------------------------------------------------------- Leaderboard
with tab_leaderboard:
    st.caption(
        "Every contributor's Colab session, ranked by checkpoint files "
        "pushed — a live proxy for training work done, not a claim about "
        "whose results are best (see the Charts tab for that)."
    )
    board = get_contributor_leaderboard()
    if board.empty:
        st.info("No contributors yet — be the first (see the Contribute tab).")
    else:
        with st.container(horizontal=True):
            st.metric("Contributors", len(board), border=True)
            st.metric("Online now", int(board["Online"].sum()), border=True)
            st.metric("Checkpoints pushed", int(board["Checkpoints pushed"].sum()), border=True)

        display = board.copy()
        display["Rank"] = [
            MEDALS[i] if i < len(MEDALS) else str(i + 1) for i in range(len(display))
        ]
        display["Status"] = display["Online"].map({True: "🟢 Online", False: "⚪ Offline"})
        display["Running for"] = display["Running for (min)"].map(
            lambda v: "-" if pd.isna(v) else f"{v:.0f} min"
        )
        display = display[["Rank", "Name", "Checkpoints pushed", "Status", "Shards", "Running for"]]

        def _row_color(row):
            color = "background-color: rgba(255, 215, 0, 0.15)" if row["Rank"] in MEDALS else ""
            return [color] * len(row)

        st.dataframe(
            display.style.apply(_row_color, axis=1),
            hide_index=True, width="stretch",
        )

# ---------------------------------------------------------------- Charts
with tab_charts:
    best = get_best_run()
    if best is None:
        st.info(
            "Nothing has finished a full fold/seed yet — the first result "
            "will headline this tab automatically. Nothing to look at? Go "
            "spar with the AI in the Beat the AI tab while you wait."
        )
    else:
        st.subheader(
            f"🏅 Best run so far — fold {best['fold_id']}, seed {best['seed']}, "
            f"Test Sharpe {best['test_metrics']['sharpe']:.2f}"
        )
        fold_id, seed = best["fold_id"], best["seed"]
        day_starts = get_test_day_starts(fold_id)
        if len(day_starts) == 0:
            st.warning("This fold's test window is too short to chart a sample day.")
        else:
            sample_idx = int(day_starts[len(day_starts) // 2])
            pair = "EURUSD"
            replay_df = replay_day(fold_id, seed, sample_idx)
            price_df = load_price_csv(pair)
            merged = price_df.merge(
                replay_df[["timestamp", "equity", f"pos_{pair}"]], on="timestamp", how="inner",
            )
            if not merged.empty:
                merged = charts.with_bar_labels(merged, fmt="%H:%M")
                st.caption(f"A sample day from this run's test window — {pair}, fold {fold_id} seed {seed}")
                st.altair_chart(
                    charts.candlestick_layers(merged, extra_tooltip=["equity:Q"])
                    .properties(height=320).interactive(),
                    width="stretch",
                )
                st.altair_chart(
                    charts.area_chart(merged, "equity", y_title="Equity", height=160),
                    width="stretch",
                )

        st.divider()
        st.caption("Test Sharpe across the whole sweep so far, by fold and seed")
        entries = load_manifest_entries()
        sharpe_df = pd.DataFrame([
            {"Fold": e["fold_id"], "Seed": e["seed"], "Test Sharpe": e["test_metrics"]["sharpe"]}
            for e in entries
        ])
        sharpe_chart = (
            alt.Chart(sharpe_df)
            .mark_bar()
            .encode(
                x=alt.X("Fold:O", title="Fold"),
                xOffset="Seed:N",
                y=alt.Y("Test Sharpe:Q"),
                color=alt.Color("Seed:N"),
                tooltip=["Fold", "Seed", "Test Sharpe"],
            )
            .properties(height=240)
        )
        st.altair_chart(sharpe_chart, width="stretch")

# ---------------------------------------------------------------- Beat the AI
with tab_challenge:
    checkpoints = list_available_checkpoints()
    if not checkpoints:
        st.info(
            "The AI needs at least one finished fold before it can play — "
            "check back once the first result lands. In the meantime, "
            "invite a friend to spin up a GPU on the Contribute tab."
        )
    else:
        st.caption(
            "Trade ONE pair, one lot, hour by hour through a real historical "
            "trading day — no peeking at bars that haven't happened yet, "
            "same as the AI. At the end, your return is compared against "
            "what the real trained agent actually did on that exact day "
            "(the agent trades all 28 pairs under the full safety layer — "
            "this is a simplified single-pair version, just for fun)."
        )

        game = st.session_state.get("arena_game")

        col_new, col_pair = st.columns([1, 2])
        with col_new:
            new_clicked = st.button("🎲 New challenge", type="primary")
        with col_pair:
            pair = st.selectbox(
                "Pair", config.PAIRS, index=config.PAIRS.index("EURUSD"), key="arena_pair",
                disabled=game is not None and not game.get("finished", False),
            )

        if new_clicked or game is None:
            fold_id, seed = random.choice(checkpoints)
            day_starts = get_test_day_starts(fold_id)
            day_start_idx = int(random.choice(day_starts)) if len(day_starts) else None
            game = {
                "fold_id": fold_id, "seed": seed, "day_start_idx": day_start_idx,
                "pair": pair, "bar_idx": 0, "equity": 1.0, "equity_curve": [1.0],
                "finished": False,
            }
            st.session_state["arena_game"] = game
            st.rerun()

        if game["day_start_idx"] is None:
            st.warning("Couldn't find a tradeable day for this run — hit New challenge to try another.")
        else:
            replay_df = replay_day(game["fold_id"], game["seed"], game["day_start_idx"])
            price_df = load_price_csv(game["pair"])
            merged = price_df.merge(
                replay_df[["timestamp", "equity"]], on="timestamp", how="inner",
            ).sort_values("timestamp").reset_index(drop=True)
            merged["bar_return"] = merged["close"] / merged["open"] - 1.0
            n_bars = len(merged)
            agent_return = merged["equity"].iloc[-1] - 1.0 if n_bars else 0.0

            revealed = merged.iloc[: game["bar_idx"]]
            with st.container(horizontal=True):
                st.metric("Bar", f"{game['bar_idx']}/{n_bars}", border=True)
                st.metric("Your equity", f"{game['equity']:.3f}", border=True)
                st.metric("Your return", f"{game['equity'] - 1.0:.2%}", border=True)

            if not revealed.empty:
                labeled = charts.with_bar_labels(revealed, fmt="%H:%M")
                st.altair_chart(
                    charts.candlestick_layers(labeled).properties(height=280),
                    width="stretch",
                )
            else:
                st.caption("Make your first call before seeing any bars — just like the open.")

            if not game["finished"] and game["bar_idx"] < n_bars:
                st.write(f"**Next bar ({merged['timestamp'].iloc[game['bar_idx']].strftime('%H:%M')}):** long, short, or flat?")
                c1, c2, c3 = st.columns(3)
                stance = None
                if c1.button("📈 Long", width="stretch"):
                    stance = 1.0
                if c2.button("📉 Short", width="stretch"):
                    stance = -1.0
                if c3.button("⏸ Flat", width="stretch"):
                    stance = 0.0

                if stance is not None:
                    r = float(merged["bar_return"].iloc[game["bar_idx"]])
                    game["equity"] *= (1.0 + stance * r)
                    game["bar_idx"] += 1
                    game["equity_curve"].append(game["equity"])
                    if game["bar_idx"] >= n_bars:
                        game["finished"] = True
                    st.session_state["arena_game"] = game
                    st.rerun()
            else:
                user_return = game["equity"] - 1.0
                st.divider()
                col_a, col_b = st.columns(2)
                col_a.metric("Your return", f"{user_return:.2%}")
                col_b.metric(f"AI's return (fold {game['fold_id']}, seed {game['seed']})", f"{agent_return:.2%}")
                if abs(user_return - agent_return) < 0.0005:
                    st.info("🤝 Dead heat!")
                elif user_return > agent_return:
                    st.success("🏆 You beat the AI on this one!")
                    st.balloons()
                else:
                    st.warning("🤖 The AI wins this round — hit New challenge for a rematch.")
