import random
import uuid

import altair as alt
import numpy as np
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
    new_challenge_env,
    replay_day,
)

MEDALS = ["🥇", "🥈", "🥉"]
BUCKET_LABELS = [str(b) if b < 0 else f"+{b}" if b > 0 else "0" for b in config.ACTION_BUCKETS]
FLAT_IDX = config.ACTION_BUCKETS.index(0)

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
        display = display[["Rank", "Name", "Checkpoints pushed", "Status", "Shards", "Last seen"]]

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
            # ONE trading day only (BARS_PER_EPISODE bars) — replay_day() and
            # the merge below never span more than that single day's window.
            sample_idx = int(day_starts[len(day_starts) // 2])
            pair = "EURUSD"
            replay_df = replay_day(fold_id, seed, sample_idx)
            price_df = load_price_csv(pair)
            merged = price_df.merge(
                replay_df[["timestamp", "equity", f"pos_{pair}"]], on="timestamp", how="inner",
            )
            if not merged.empty:
                merged = charts.with_bar_labels(merged, fmt="%H:%M")
                st.caption(f"One trading day from this run's test window — {pair}, fold {fold_id} seed {seed}")
                st.altair_chart(
                    charts.candlestick_layers(merged, extra_tooltip=["equity:Q"])
                    .properties(height=340).interactive(),
                    width="stretch",
                )
                st.altair_chart(
                    charts.line_chart(merged, "equity", y_title="Equity", height=160),
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
            .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
            .encode(
                x=alt.X("Fold:O", title="Fold", axis=alt.Axis(domainColor=charts.GRID_COLOR, tickColor=charts.GRID_COLOR)),
                xOffset="Seed:N",
                y=alt.Y("Test Sharpe:Q", axis=alt.Axis(gridColor=charts.GRID_COLOR, domainColor=charts.GRID_COLOR)),
                color=alt.Color("Seed:N", scale=alt.Scale(scheme="viridis")),
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
            "You run the SAME book as the agent: all 28 pairs, one historical "
            f"trading day, {config.BARS_PER_EPISODE} hourly bars, no peeking "
            "ahead. For every pair, every bar, pick from the exact same five "
            f"position buckets the policy picks from ({', '.join(BUCKET_LABELS)} "
            "— vol-scaled lots), and the same safety layer (exposure caps, "
            "vol/VaR circuit breakers, daily stop-loss) applies to you too — "
            "an illegal pick gets snapped to the nearest legal one, exactly "
            "like it would never have been sampled for the agent. At the end, "
            "your final equity is compared against what the real trained "
            "agent actually did on that exact day."
        )

        game = st.session_state.get("arena_game")
        new_clicked = st.button("🎲 New challenge", type="primary")

        if new_clicked or game is None:
            fold_id, seed = random.choice(checkpoints)
            day_starts = get_test_day_starts(fold_id)
            if len(day_starts) == 0:
                game = {"day_start_idx": None}
            else:
                day_start_idx = int(random.choice(day_starts))
                env = new_challenge_env(fold_id, seed, day_start_idx)
                game = {
                    "id": uuid.uuid4().hex,
                    "fold_id": fold_id, "seed": seed, "day_start_idx": day_start_idx,
                    "env": env,
                    "positions": {p: FLAT_IDX for p in config.PAIRS},
                    "equity_curve": [1.0],
                    "timestamps": [pd.Timestamp(env.timestamps[day_start_idx])],
                    "overrides": [], "finished": False, "stop_loss_hit": False,
                    "agent_return": None,
                }
            st.session_state["arena_game"] = game
            st.rerun()

        if game.get("day_start_idx") is None:
            st.warning("Couldn't find a tradeable day for this run — hit New challenge to try another.")
        else:
            env = game["env"]
            n_bars = config.BARS_PER_EPISODE

            with st.container(horizontal=True):
                st.metric("Bar", f"{env.bar_idx}/{n_bars}", border=True)
                st.metric("Your equity", f"{env.equity:.3f}", border=True)
                st.metric("Your return", f"{env.equity - 1.0:.2%}", border=True)

            pos_values = [config.ACTION_BUCKETS[game["positions"][p]] for p in config.PAIRS]
            st.caption("Current book — one cell per pair, red = short, green = long")
            st.altair_chart(charts.position_heatmap(config.PAIRS, pos_values), width="stretch")

            equity_df = pd.DataFrame({"timestamp": game["timestamps"], "equity": game["equity_curve"]})
            equity_df = charts.with_bar_labels(equity_df, fmt="%H:%M")
            st.altair_chart(
                charts.line_chart(equity_df, "equity", y_title="Equity", height=160),
                width="stretch",
            )

            if game["overrides"]:
                st.caption(
                    f"⚠️ Safety layer snapped {len(game['overrides'])} pick(s) to the "
                    f"nearest legal bucket last bar: {', '.join(game['overrides'])}"
                )

            if not game["finished"]:
                if game["stop_loss_hit"]:
                    st.error("🛑 Daily stop-loss breached — your day ends here, same as it would for the agent.")

                next_ts = pd.Timestamp(env.timestamps[env._current_bar_absolute_idx()])
                st.write(f"**Bar {env.bar_idx + 1}/{n_bars} ({next_ts.strftime('%H:%M')}) — set your book:**")

                mask_preview = env.get_action_mask()
                restricted = [
                    f"{pair}: only {{{', '.join(BUCKET_LABELS[j] for j in range(len(BUCKET_LABELS)) if mask_preview[i, j]) or 'flat'}}} allowed"
                    for i, pair in enumerate(config.PAIRS) if not mask_preview[i].all()
                ]
                if restricted:
                    with st.expander(f"⚠️ {len(restricted)} pair(s) under safety restrictions this bar"):
                        for line in restricted:
                            st.caption(line)

                editor_df = pd.DataFrame({
                    "Pair": config.PAIRS,
                    "Position": [BUCKET_LABELS[game["positions"][p]] for p in config.PAIRS],
                })
                edited = st.data_editor(
                    editor_df,
                    column_config={
                        "Pair": st.column_config.TextColumn("Pair", disabled=True),
                        "Position": st.column_config.SelectboxColumn("Position", options=BUCKET_LABELS, required=True),
                    },
                    hide_index=True, width="stretch", height=380, num_rows="fixed",
                    key=f"arena_editor_{game['id']}_{env.bar_idx}",
                )

                if st.button("✅ Confirm bar", type="primary"):
                    mask = env.get_action_mask()  # (n_pairs, n_buckets), reflects state BEFORE this step
                    chosen_idx = np.array([BUCKET_LABELS.index(v) for v in edited["Position"]])
                    final_idx = chosen_idx.copy()
                    overridden = []
                    for i, pair in enumerate(config.PAIRS):
                        if not mask[i, chosen_idx[i]]:
                            legal = np.where(mask[i])[0]
                            if len(legal):
                                final_idx[i] = legal[np.argmin(np.abs(legal - chosen_idx[i]))]
                                overridden.append(pair)

                    _state, _reward, done, info = env.step(final_idx)
                    game["positions"] = {p: int(final_idx[i]) for i, p in enumerate(config.PAIRS)}
                    game["equity_curve"].append(info["equity"])
                    game["timestamps"].append(next_ts)  # the bar that was just traded, not the upcoming one
                    game["overrides"] = overridden
                    game["stop_loss_hit"] = bool(info.get("stop_loss_hit", False))
                    if done:
                        game["finished"] = True
                    st.session_state["arena_game"] = game
                    st.rerun()
            else:
                if game["agent_return"] is None:
                    replay_df = replay_day(game["fold_id"], game["seed"], game["day_start_idx"])
                    game["agent_return"] = float(replay_df["equity"].iloc[-1] - 1.0)
                    st.session_state["arena_game"] = game

                user_return = env.equity - 1.0
                agent_return = game["agent_return"]
                st.divider()
                col_a, col_b = st.columns(2)
                col_a.metric("Your return", f"{user_return:.2%}")
                col_b.metric(f"AI's return (fold {game['fold_id']}, seed {game['seed']})", f"{agent_return:.2%}")
                if game["stop_loss_hit"]:
                    st.error("🛑 Your daily stop-loss hit before the day finished.")
                if abs(user_return - agent_return) < 0.0005:
                    st.info("🤝 Dead heat!")
                elif user_return > agent_return:
                    st.success("🏆 You beat the AI on this one!")
                    st.balloons()
                else:
                    st.warning("🤖 The AI wins this round — hit New challenge for a rematch.")
