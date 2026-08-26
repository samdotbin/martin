import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from lib import (
    config,
    get_benchmark_comparison,
    is_degenerate_flat,
    list_available_checkpoints,
    load_manifest_entries,
    run_stress_test,
)


@st.fragment(run_every="30s")
def live_results():
    """Auto-refreshing section — picks up new fold/seed results as training
    finishes them (load_manifest_entries has a 20s cache TTL) without
    needing to restart the app or touch the rest of the page."""
    entries = load_manifest_entries()
    if not entries:
        st.info(
            "No completed fold/seed runs yet in RUN_MANIFEST.json — this section "
            "checks every 30s and will fill in once training (locally or on "
            "Colab) produces a result."
        )
        return

    rows = []
    for e in entries:
        tm = e["test_metrics"]
        daily_returns = tm.get("daily_returns", [])
        rows.append({
            "Fold": e["fold_id"], "Seed": e["seed"],
            "Sharpe": tm["sharpe"], "Win rate": tm["win_rate"],
            "Max drawdown": tm["max_drawdown"], "Cumulative return": tm["cumulative_return"],
            "Residual exposure": tm.get("residual_exposure", 0.0),
            "All flat": is_degenerate_flat(daily_returns),
        })
    df = pd.DataFrame(rows).sort_values(["Fold", "Seed"]).reset_index(drop=True)

    n_flat = int(df["All flat"].sum())
    with st.container(horizontal=True):
        st.metric("Completed runs", len(entries), border=True)
        st.metric("Mean Sharpe", f"{df['Sharpe'].mean():.2f}", border=True)
        st.metric("Mean win rate", f"{df['Win rate'].mean():.1%}", border=True)
        st.metric(
            "All-flat runs", n_flat, border=True,
            help="Fold/seed combos that took zero trades across their entire test window — "
                 "a degenerate policy, not just a losing one.",
        )

    with st.container(border=True):
        st.subheader("Per-fold / seed results")
        if n_flat:
            st.caption(f"{n_flat} run(s) highlighted below never traded during testing.")

        def _highlight_flat(row):
            color = "background-color: rgba(239, 83, 80, 0.25)" if row["All flat"] else ""
            return [color] * len(row)

        st.dataframe(
            df.style.apply(_highlight_flat, axis=1).format({
                "Sharpe": "{:.2f}", "Win rate": "{:.1%}",
                "Max drawdown": "{:.2%}", "Cumulative return": "{:.2%}",
                "Residual exposure": "{:.3f}",
            }),
            hide_index=True, width="stretch",
        )

    with st.container(border=True):
        st.subheader("Equity curves")
        st.caption("Reconstructed from each run's per-day test returns (compounded).")
        curves = []
        for e in entries:
            daily_returns = e["test_metrics"].get("daily_returns", [])
            if not daily_returns:
                continue
            equity = np.cumprod(1 + np.array(daily_returns))
            label = f"fold {e['fold_id']} / seed {e['seed']}"
            for day_idx, eq in enumerate(equity):
                curves.append({"Day": day_idx, "Equity": eq, "Run": label})
        if curves:
            curve_df = pd.DataFrame(curves)
            chart = (
                alt.Chart(curve_df)
                .mark_line()
                .encode(
                    alt.X("Day:Q"),
                    alt.Y("Equity:Q", scale=alt.Scale(zero=False)),
                    alt.Color("Run:N"),
                    tooltip=["Run", "Day", "Equity"],
                )
                .properties(height=400)
                .interactive()
            )
            st.altair_chart(chart, width="stretch")
        else:
            st.caption("No daily-return series recorded yet.")

    with st.expander("Benchmark comparison (vs. always-flat / buy-and-hold)"):
        st.caption(
            "Buy-and-hold holds overnight, which this strategy structurally "
            "cannot — informational only, not the primary bar (§15)."
        )
        bench_pair = st.selectbox("Benchmark pair", config.PAIRS, index=config.PAIRS.index("EURUSD"), key="bench_pair")
        bench_rows = []
        for fold_id in sorted(df["Fold"].unique()):
            bench = get_benchmark_comparison(int(fold_id), bench_pair)
            fold_sharpe = df.loc[df["Fold"] == fold_id, "Sharpe"].mean()
            bench_rows.append({
                "Fold": fold_id, "Policy Sharpe (mean over seeds)": fold_sharpe,
                "Buy & hold Sharpe": bench["buy_and_hold"]["sharpe"],
                "Always-flat Sharpe": bench["always_flat"]["sharpe"],
            })
        bench_df = pd.DataFrame(bench_rows)
        st.dataframe(
            bench_df.style.format({
                "Policy Sharpe (mean over seeds)": "{:.2f}",
                "Buy & hold Sharpe": "{:.2f}", "Always-flat Sharpe": "{:.2f}",
            }),
            hide_index=True, width="stretch",
        )

    with st.expander("Aggregate significance (deflated Sharpe / block bootstrap)"):
        st.caption(
            "Deflated Sharpe corrects for the number of implicit trials "
            "(configs x folds x seeds) — a naive Sharpe on the best-looking run "
            "overstates significance. Block bootstrap resamples WHOLE folds, "
            "not individual days, since adjacent folds share macro regimes (§13)."
        )
        import eval as eval_module  # noqa: E402  (deferred: only needed inside this expander)

        n_trials = len(entries)
        all_daily_returns = [r for e in entries for r in e["test_metrics"].get("daily_returns", [])]
        mean_sharpe = df["Sharpe"].mean()
        if all_daily_returns:
            psr = eval_module.deflated_sharpe_ratio(mean_sharpe, n_obs=len(all_daily_returns), n_trials=n_trials)
            st.metric("Deflated/probabilistic Sharpe", f"{psr:.2f}", border=True)

        fold_sharpes = df.groupby("Fold")["Sharpe"].mean().tolist()
        if len(fold_sharpes) > 1:
            ci = eval_module.block_bootstrap_folds(fold_sharpes)
            st.write(f"Block-bootstrap 90% CI on mean fold Sharpe: "
                     f"**[{ci['low']:.2f}, {ci['high']:.2f}]** (mean {ci['mean']:.2f})")
        else:
            st.caption("Need results from more than one fold for a block-bootstrap CI.")


live_results()

st.divider()
st.subheader("Cost-sensitivity stress test")
st.caption(
    "Re-runs a chosen fold/seed's test window with spreads widened, to check "
    "how much of its edge (if any) survives higher transaction costs (§15). "
    "Runs on demand — not part of the auto-refreshing section above."
)

checkpoints = list_available_checkpoints()
if not checkpoints:
    st.info("No checkpoints available yet for stress testing.")
else:
    folds = sorted({f for f, s in checkpoints})
    col1, col2, col3 = st.columns(3)
    with col1:
        st_fold = st.selectbox("Fold", folds, key="stress_fold")
    with col2:
        st_seeds = sorted(s for f, s in checkpoints if f == st_fold)
        st_seed = st.selectbox("Seed", st_seeds, key="stress_seed")
    with col3:
        spread_mult = st.slider("Spread multiplier", 1.0, 5.0, 2.5, step=0.5, key="stress_mult")

    if st.button("Run stress test", key="run_stress"):
        with st.spinner("Running normal + stressed evaluation..."):
            normal_metrics, stressed_metrics = run_stress_test(st_fold, st_seed, spread_mult, n_episodes=20)
        st.session_state["stress_result"] = (st_fold, st_seed, spread_mult, normal_metrics, stressed_metrics)

    if "stress_result" in st.session_state:
        r_fold, r_seed, r_mult, normal_metrics, stressed_metrics = st.session_state["stress_result"]
        st.caption(f"fold {r_fold} / seed {r_seed}, {r_mult}x spread")
        with st.container(horizontal=True):
            st.metric("Normal Sharpe", f"{normal_metrics['sharpe']:.2f}", border=True)
            st.metric(
                "Stressed Sharpe", f"{stressed_metrics['sharpe']:.2f}",
                delta=f"{stressed_metrics['sharpe'] - normal_metrics['sharpe']:.2f}",
                border=True,
            )
            st.metric("Normal win rate", f"{normal_metrics['win_rate']:.1%}", border=True)
            st.metric("Stressed win rate", f"{stressed_metrics['win_rate']:.1%}", border=True)
