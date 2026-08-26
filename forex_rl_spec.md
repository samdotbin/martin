# Forex Multi-Pair RL Trading System — Full Spec (v5)

Builds on v4 (finite-horizon time-awareness, file structure with implementation
gotchas, §21) by closing the gaps that would otherwise block implementation:
concrete defaults for every previously-vague parameter (VaR cap, bucket-to-lots
mapping, close-out penalty, etc.), a data-quality validation step, fold
independence caveats, and a minimal path to live deployment.

---

## 1. Episode Length Is a Design Choice, Not a Constraint

The data (26 years of H1 bars across 28 pairs) can be sliced any way you want:

| Option | Bars/episode | Episodes available |
|---|---|---|
| A — 1 day | 24 | ~6,552 |
| B — 2 days | 48 | ~3,276 |
| C — 5 days | 120 | ~1,310 |
| D — 10 days | 240 | ~655 |
| E — variable length | — | as many as needed |

**Default: 24 bars (Option A).** Reasons to keep it:

1. Forces intraday focus — no overnight carry risk to model.
2. Natural reset — each day starts flat, no position-carryover bugs.
3. Clean evaluation — daily P&L is a natural, auditable unit.
4. More episodes per unit of data → more gradient updates, faster training.

Each day is an independent "game": start flat → 24 decisions → forced flat close.
The agent is learning *how to trade a given day under the current regime*, not
multi-day trend-following. If you later want multi-day positions, move to Option B
or C — nothing else in this spec changes structurally, only episode length and the
max-hold constraint in §10.

---

## 2. Data Quality & Coverage Validation

Everything downstream (episodes, folds, regimes) silently assumes the data is
complete and aligned across all 28 pairs. It usually isn't, and nothing else in
this spec catches that on its own — so this step runs once, before folds are cut.

**Known issues to check for:**

| Issue | Risk if unchecked |
|---|---|
| Symbol history start dates differ (exotic crosses often start later than majors) | Early folds silently trade on incomplete/synthetic pairs |
| Holiday/weekend gaps | Irregular day-of-week distribution skews the day-of-week feature |
| Delisted/renamed symbols, broker-specific point values | Silent unit mismatches (pips vs. points) across the 28-pair tensor |

**`validate_data_coverage()` — run once, before fold construction:**

- Report each symbol's effective start date and % bar coverage.
- Compute the **effective start year**: the earliest year in which all 28 pairs
  have ≥90% coverage. Fold construction (§13) starts from this year, not from
  the raw `DEFAULT_HISTORY_START`.
- Flag (don't silently drop) any day where a pair is missing >1 bar — days with
  partial data should be excluded from the episode pool, not filled with
  fabricated values.
- Warn if the resulting day-of-week distribution deviates from uniform by more
  than a small tolerance (catches systematic gaps, e.g. one broker's feed
  missing all Mondays in a given year).
- Assert all CSVs share the same pip/point convention before merging; a silent
  10x unit mismatch on one pair would corrupt that pair's z-scoring and the
  regime PCA (§11) without raising an error anywhere else.

This is a one-time data-engineering step, not a per-fold cost — run it once per
raw data refresh and cache the result.

---

## 3. Time-Awareness: Making the Agent Know It's a Finite-Horizon Game

A day-trading agent isn't just reacting bar-by-bar — it must know *how much runway
is left* and *how much risk budget it has already spent*. Without this, the network
can only infer urgency indirectly from history, and it tends to learn a single
"average" behavior that's mediocre at bar 3 and mediocre at bar 23, instead of a
policy that visibly winds down risk as the forced close approaches. This is the
correct, standard way to fix that (the same idea used for time-to-maturity features
in options-hedging RL).

**2.1 Add explicit time-to-close to the state**

```
bars_remaining_norm = (24 - current_bar) / 24     # ∈ [0, 1], added to state
```

This is the single most important addition. It lets the policy condition directly
on urgency instead of inferring it, and lets it learn distinct behavior near the
open vs. near the close.

**2.2 Add risk-budget-used to the state**

```
risk_budget_used = |equity_t - equity_0| / stop_loss_threshold    # e.g. fraction of -2% used
```

A trader behaves differently at −0.2% into a −2% daily stop than at −1.8% into it.
Exposing this lets the policy *anticipate* the safety layer (§9) instead of only
being clamped by it after the fact.

**2.3 Use an undiscounted return, γ = 1, not γ = 0.99**

The episode is genuinely finite and always ends at bar 24 by construction (forced
close). Discounting with γ<1 implicitly tells the agent that a unit of terminal
P&L matters less than the same unit earned at bar 1 — which isn't true here; both
contribute equally to the day's realized return. γ<1 is a workaround for
infinite/indefinite-horizon problems; with `bars_remaining_norm` already in state
and a hard 24-bar cutoff, undiscounted return (γ = 1) is the technically correct
setting and avoids mildly distorting terminal-bar incentives.

**2.4 Give the value function the same time inputs as the policy**

The value head must also see `bars_remaining_norm` and `risk_budget_used`. A given
position (e.g., three pairs long) has very different expected future value at bar
2 (time to develop or be reversed) than at bar 23 (about to be forced flat). If the
value net can't tell these apart, GAE advantage estimates get noisy exactly when
precision matters most — right before the close.

**2.5 Small explicit close-out shaping penalty**

On top of the safety layer and forced close mechanic, add a small penalty on held
size in the last 1–2 bars before bar 24 (§7, reward function). This makes
"wind down before mandatory close" a learned habit reflected in the reward signal
itself, not purely an artifact of the environment forcing a flat close — which
generalizes better if you later change max-hold or episode length (§1).

**Net effect:** these five changes turn "24 bars, forced close" from an
environment-level rule the agent merely obeys into something the agent internally
represents and plans around — which is what "aware it's day trading" concretely
means in this architecture.

---

## 4. Core Game Design

| Component | Spec |
|---|---|
| Episode | 24 hourly bars. Start flat, force-close all positions at bar 24. |
| State | Last 64 bars of edge features for all 28 pairs (log returns, HL range, volume/spread z-scored) + current per-pair position size + per-currency net exposure + **regime embedding** (5–10 dims, see §11) + **bars_remaining_norm** + **risk_budget_used** (§3) + time-of-day (sin/cos or one-hot) + day-of-week + session flag (Asian/London/NY/overlap). |
| Action | Per-pair discrete sized action: 5 buckets `{-2, -1, 0, +1, +2}` units per pair (28 independent categorical heads, 5 logits each). |
| Action Masking | Mask any bucket that would breach the per-pair cap or per-currency exposure cap. |
| Reward | Per-bar: differential Sharpe ratio increment minus transaction cost minus turnover penalty minus close-out shaping penalty near bar 24 (§3.5, §7). Terminal: risk-adjusted return with variance and drawdown penalties. Early termination with large penalty on daily stop-loss. |
| Termination | End of bar 24, daily stop-loss hit, or safety-layer circuit-breaker halt (§9). |

---

## 5. Algorithm: PPO

Multi-discrete action space (5^28 joint combinations) rules out joint-action value
methods. PPO with a factorized policy over independent per-pair heads, plus a
centralized value function for cross-pair coordination, is the right fit.

```
L_total = L_PPO_clip + c1 * L_value + c2 * L_entropy + c3 * L_aux

# target: log return (consistent with state feature 0, §4), scaled ×100 so the
# regression target is O(1e-2) rather than O(1e-4) — raw FX log returns are too
# small for MSE to produce a meaningful gradient relative to the PPO losses.
L_aux = MSE(predicted_next_bar_log_return * 100, actual_next_bar_log_return * 100)
```

Warm `c3` up from `0.01` to `0.1` over the first ~50 epochs rather than fixing it
at `0.1` from the start — early in training the auxiliary regression task is easy
relative to the policy/value losses and can otherwise dominate the gradient,
pulling representations toward next-bar prediction at the expense of the actual
control objective before the policy has anything reasonable to work with.

---

## 6. Network Architecture

```
Input: [Batch, 64 bars, 28 edges, features]
       + [positions, exposures, regime, bars_remaining_norm, risk_budget_used, time features]
       ↓
Temporal Conv1D / dilated causal conv over 64 bars (64 → 1)
       ↓
GNN over complete graph of 8 currencies
  - Node features: currency embedding + aggregated edge info
  - Edge features: pair-specific engineered features (sign-flipped for reverse direction)
  - 2-3 message-passing layers
       ↓
Outputs:
  1. Global embedding (pooled over nodes/edges)
  2. Per-edge embeddings (28 pairs, each d-dim)

Policy Head (per pair):
  MLP(edge_embedding_i + global_embedding + regime_embedding
      + bars_remaining_norm + risk_budget_used) → 5 logits (size buckets)
  Apply action masks → Softmax → categorical distribution

Value Head:
  MLP(global_embedding + mean(edge_embeddings) + regime_embedding
      + bars_remaining_norm + risk_budget_used) → scalar state value

Auxiliary Head (training only):
  MLP(edge_embedding_i) → predicted next-bar return
```

Both the policy and value heads take `bars_remaining_norm` and `risk_budget_used`
directly (§3.1, §3.2, §3.4), not just implicitly through the temporal history.
Model size: **2–5M parameters** — comfortably fits a 16GB GPU (see §12).

---

## 7. Reward Function

```
R_t = (eq_t - eq_{t-1}) / eq_{t-1}          # bar return
ΔA_t = R_t - A_{t-1}
ΔB_t = R_t^2 - B_{t-1}
D_t = (B_{t-1} * ΔA_t - 0.5 * A_{t-1} * ΔB_t) / (max(B_{t-1} - A_{t-1}^2, ε))^1.5   # ε = 1e-6

close_penalty_t = μ * total_held_size_t   if bars_remaining_norm <= 2/24 else 0   # §3.5

turnover_t = Σ_pairs |target_size_t - position_size_{t-1}|     # cumulative absolute change

r_t = D_t - cost_t - κ * turnover_t - close_penalty_t
```

The `ε` in the denominator matters in practice: in the first few bars of an episode
`A` and `B` are both near zero, so `(B - A²)` is near zero and the raw formula is
unstable without it. The `A`, `B` running stats use an EMA update
(`A ← A + η·ΔA`, same for `B`) with `η = 0.1` as a starting point — keep this as a
named, tunable constant rather than a hardcoded literal (§21).

Terminal:

```
R_terminal = (eq_24 - eq_0) / eq_0
             - λ_dd  * max_drawdown_intraday
             - λ_var * std(intraday_equity_curve)
```

`λ_dd = 0.5–1.0`, `λ_var = 0.25–0.5`. Start `μ = 0.01` for the close-out penalty
weight, then tune via grid search. Stop-loss breach → `R_terminal = -10.0`.

```
R = α * Σ r_t + β * R_terminal        # α = 0.05, β = 1.0 — starting point, see note
```

**Note on α/β balance:** with `α = 0.05` and `β = 1.0`, the terminal reward
dominates almost completely — the per-bar signal is small enough that early
training gradients come mostly from an outcome the agent has only partial control
over 24 steps out. If early training looks unstable or slow to improve, try
raising `α` toward `0.2–0.5` so the differential-Sharpe per-bar signal — which is
directly attributable to the action just taken — carries more weight relative to
the terminal outcome.

---

## 8. Action Space & Bucket Semantics

The 5 buckets `{-2, -1, 0, +1, +2}` (§4) need a concrete, defined mapping to
actual lot sizes, or the network's action can't be executed consistently.

```
lots = bucket * BASE_LOT * vol_scale
vol_scale = clip(median_vol_across_pairs / symbol_vol, 0.25, 4.0)
```

`vol_scale` is what makes a "+1" carry comparable risk on a low-vol pair
(e.g., EURCHF) and a high-vol pair (e.g., GBPJPY) — without the clip, a very
quiet pair could get an extreme size multiplier from a near-zero denominator.

- **Per-pair cap:** `MAX_PAIR_EXPOSURE = 2.0` lots — deliberately equal to
  `max(bucket) * BASE_LOT`, so the bucket range and the cap agree by
  construction rather than needing separate tuning.
- **Mask, don't clip, invalid actions.** If a bucket would breach the per-pair
  or per-currency cap, remove it from the categorical distribution (§4) before
  sampling. Clipping a sampled action to the nearest legal value instead would
  let the policy get credit/blame for an action it didn't actually take,
  which corrupts the PPO log-probability the update is based on.

---

## 9. Safety Layer (outside the learned policy)

- **Vol circuit breaker:** halt new entries if realized vol over the trailing
  `VOL_BREAKER_WINDOW = 20` bars exceeds `VOL_BREAKER_MULTIPLIER = 3.0` times
  its own longer-run trailing average.
- **Hard VaR cap:** use **historical VaR** — the specified percentile of the
  last 100 bars' portfolio returns — rather than a Gaussian or Cornish-Fisher
  assumption; it's the simplest method and makes no distributional assumption
  that FX returns (fat-tailed) would violate. Default `VAR_CAP = 0.02` (2% of
  portfolio).
- **Exposure caps:** ±1.5 units net per currency (`MAX_ABS_EXPOSURE = 0.5` per
  the config in §21, adjust to match), enforced via masking.
- **Hard forced flatten:** independent of anything the policy learned, force all
  positions flat at bar 24 — this backstops §3's shaping penalty in case the learned
  behavior doesn't fully wind down in time.

**Priority order when checks conflict:** exposure caps → vol circuit breaker →
VaR cap → hard forced flatten. Exposure caps are the cheapest, most local check
(per-pair/per-currency) and run first; the VaR cap is portfolio-wide and looks
at everything the earlier checks already allowed, so it's the final gate before
execution.

Runs identically at train and inference time.

---

## 10. Environment Mechanics

- **Lot sizing:** policy output (size buckets), scaled by ATR/vol-inverse so a
  "+1" carries comparable risk across pairs.
- **Transaction costs:** spread + slippage on every trade, including forced
  close at bar 24.
- **Position management:** buy/sell/hold semantics parameterized by target size.
- **Max hold:** optional forced close after N bars (e.g., 8) to encourage active
  management; scale N if you move to a longer episode (§1, Option B/C).
- **Episode start alignment:** each episode begins at the first bar of a
  calendar day (00:00 UTC), not an arbitrary offset — this keeps "bar 24" and
  "forced close" meaning the same thing across every episode.
- **Contiguity validation:** before adding a day to the episode pool, confirm
  its 24 bars have consecutive timestamps with no gap greater than 2 hours.
  A day straddling a holiday or a data outage should be dropped by
  `validate_data_coverage()` (§2), not silently trained on with a corrupted
  lookback window.
- **Lookback spans the previous day on purpose:** the 64-bar lookback (§4) at
  the start of a new episode intentionally includes bars from the prior day —
  the agent should have access to yesterday's close and late-session
  behavior as context, even though today's position book resets to flat.

---

## 11. Regime Extraction (PCA / SVD / ICA)

FX is non-stationary — a policy that averages over all regimes underperforms in
extremes. Condition the policy on regime instead of hoping retraining alone adapts.

| Method | What it does | Best for |
|---|---|---|
| PCA | Principal components of market covariance | Overall market state (risk-on/off) |
| SVD | Numerically stable variant of PCA | Dimensionality reduction |
| ICA | Independent (not just uncorrelated) components | Distinct "market modes" |

**Pipeline:**

```python
# 1. Extract features across all pairs for each bar
# Shape: (n_bars, n_pairs * n_features) = (157k, 28*4=112)

# 2. Minimum-data guard — PCA on too few days is unstable
assert len(train_features) >= MIN_REGIME_TRAINING_DAYS * 24, \
    "fold training window too short to fit a stable regime model"

# 3. Fit PCA on TRAINING data only (fit once per fold, never on test data)
from sklearn.decomposition import PCA
pca = PCA(n_components=10)
regime_loadings = pca.fit_transform(train_features)
test_loadings = pca.transform(test_features)   # transform only, no re-fit

# 3a. Fix SVD/PCA sign ambiguity so the same regime doesn't flip sign
#     across folds fit on different windows.
for i in range(regime_loadings.shape[1]):
    if regime_loadings[np.argmax(np.abs(regime_loadings[:, i])), i] < 0:
        regime_loadings[:, i] *= -1
        test_loadings[:, i] *= -1

# 4. Cluster into discrete regimes
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=5)
regime_labels = kmeans.fit_predict(regime_loadings)

# 5. Feed regime as continuous embedding (the 10 PCA loadings) AND/OR
#    discrete cluster id (one-hot) into the state — don't discard the continuous
#    signal in favor of only the cluster label.
```

`MIN_REGIME_TRAINING_DAYS = 100` as a starting floor — below this, PCA
components are liable to be dominated by noise rather than real cross-pair
structure. This should never actually trip given the 4–5 year fold windows in
§13, but it belongs in the code as an explicit assertion rather than an assumed
invariant, so a misconfigured fold fails loudly instead of silently training on
a garbage regime signal.

**Important fold-safety rule:** re-fit PCA/KMeans separately per walk-forward fold,
on that fold's training window only. Fitting once on the full 26-year history and
reusing it across folds leaks future information into earlier folds.

Regimes are unlabeled — the algorithm finds them; you interpret them after the fact.
Illustrative examples of what clusters often correspond to: high-vol trending
(crisis), low-vol ranging (quiet), risk-on, risk-off, and transition periods. Don't
hand-code these labels into the pipeline; they're a post-hoc read of the clusters.

Regime clusters found on 2010 data aren't guaranteed to mean the same thing by
2020 — market structure drifts. This is expected, not a bug: it's exactly why
regimes are re-fit per fold (above) rather than fit once globally. Treat
regime cluster identity as fold-local; don't assume "regime 3" means the same
thing in fold 1 and fold 6.

---

## 12. Training Infrastructure — 16GB Colab GPU Feasibility

| Component | Memory estimate | Feasible? |
|---|---|---|
| Raw data (157k bars × 28 pairs × 4 features × 4 bytes) | ~70 MB | ✅ |
| Regime features (157k × 10 × 4 bytes) | ~6 MB | ✅ |
| Network (2–5M params × 4 bytes) | ~20 MB | ✅ |
| Batch activations (128 episodes × 24 steps) | ~500 MB | ✅ |
| **Total** | **< 1 GB** | ✅ easy headroom |

A single fold (4–5 years of training data) should train in roughly **2–4 hours**
on a Colab GPU, plus a small, near-negligible **2–5 minutes** per fold to fit the
regime pipeline (§11) — PCA/KMeans on ~1,000 days × 112 features is fast; it
doesn't materially change the per-fold time budget above.

---

## 13. Fold-Parallel Walk-Forward Training

Train several walk-forward folds independently rather than one model on all history:

```
Data: 2010-2026 (16 years)

Fold 1: 2010-2014 Train (80/10/10 internal split) → 2015 Test
Fold 2: 2011-2015 Train (80/10/10 internal split) → 2016 Test
Fold 3: 2012-2016 Train (80/10/10 internal split) → 2017 Test
...(8-10 folds)

Each fold:
  - Train on 4-5 years (~1,000 days), split 80% train / 10% internal val / 10% internal test
  - Val used for early-stopping / hyperparameter choices, so the held-out
    1-year Test set is never touched until final evaluation
  - Test on 1 year (~250 days), strictly chronological, no leakage
  - Regime PCA/KMeans re-fit per fold (§11)
```

The internal 80/10/10 split matters: without it, any early-stopping or
hyperparameter decision made by watching performance on the 1-year test set is
itself a form of leakage — you'd be indirectly fitting to the same data you
report results on.

```python
def train_fold(fold_id, train_start, train_end, test_start, test_end):
    train_data, val_data = load_slice(train_start, train_end, split=(0.8, 0.1, 0.1))
    test_data = load_slice(test_start, test_end)

    regime_model = fit_regime_pipeline(train_data)          # §11, fold-local fit
    policy = train_ppo(train_data, val_data, regime_model)  # §5-§10, early-stop on val
    results = evaluate(policy, test_data, regime_model)

    return results

# Run sequentially on a single-GPU Colab instance (see §21 — mp.Pool over
# CPU processes does not parallelize GPU compute and can cause CUDA context
# conflicts). Only use torch.multiprocessing with 'spawn' if multiple GPUs
# are genuinely available to spread folds across.
for fold_config in fold_configs:
    results = train_fold(**fold_config)
```

**Why this works:** smaller per-fold training set → faster iteration; strictly
out-of-sample testing per fold; no cross-fold leakage since each fold's regime
fit and PPO training only see that fold's training window.

**Fold independence caveat:** consecutive folds' *test* years (2015, 2016, 2017...)
don't overlap, but they aren't independent samples either — adjacent years share
persistent macro regimes, and consecutive folds' *training* windows overlap by
3–4 years. Treat the "8–10 folds" as giving you meaningfully less than 8–10
independent data points when computing bootstrap confidence intervals (§15);
a block-bootstrap that resamples whole folds, or simply reporting wider,
more conservative intervals, is more honest than treating each fold as i.i.d.

**Champion/challenger deployment:** the most recently trained fold's policy is
the "challenger." Evaluate weekly against the currently deployed "champion" on
the trailing 4 weeks of new out-of-sample data; only promote the challenger if
it beats the champion's deflated Sharpe over that window.

---

## 14. Training Flow — One Fold

```
For each fold (4-5 years of data):

  Rollout:
    For each episode (day):
      Pick random day from train_data
      env.reset(day_start, day_start + 24)
      For t in 0..23:
        state  = env.get_state()          # includes regime embedding (§11),
                                           # bars_remaining_norm, risk_budget_used (§3)
        action = policy.sample(state)
        next_state, reward = env.step(action)
        store transition
      terminal_reward = compute_terminal()
      attach to last transition
    Buffer: 256 episodes × 24 steps = 6,144 transitions

  Compute Advantages (GAE, γ = 1 — see §3.3):
    A_t = Σ(λ)^k * (r_t + V(s_{t+1}) - V(s_t))

  PPO Update (4 epochs):
    For each mini-batch (64 transitions):
      Loss = L_clip + c1*L_value + c2*L_entropy + c3*L_aux
      Backprop → update policy

  Evaluate on validation data (never trained on):
    Greedy rollout (deterministic policy)
    Track daily returns, drawdowns

  Repeat until convergence (50-100 epochs)

  Final evaluation on held-out test data:
    Deflated Sharpe, max drawdown, win rate, spread-stress test (§15)
```

---

## 15. Evaluation Protocol

| Metric | Meaning |
|---|---|
| Daily Sharpe | mean(daily returns) / std(daily returns) × √252 |
| Deflated Sharpe Ratio | Sharpe corrected for the number of implicit trials. **Trials = (hyperparameter configs tried) × (folds) × (random seeds)** — log this count explicitly in a run manifest (§16) rather than guessing it after the fact in `eval.py`. |
| Win rate | % of days with positive return |
| Max daily drawdown | worst intraday drawdown across all days |
| Cumulative return | compounded daily returns |
| End-of-day residual exposure | `mean(abs(held_size in last 2 bars)) / (n_pairs * max_bucket)` — normalized to [0, 1] so it's comparable across runs and configs, not just a raw lot figure. Should shrink toward 0 as training progresses; a direct check that time-awareness (§3) is actually working. |
| Benchmark (primary) | **Always-flat** (equity stays at initial cash) — the most meaningful baseline for a strategy that must close flat every day; anything not beating "don't trade" isn't earning its risk. |
| Benchmark (secondary) | Buy-and-hold equal-weight basket of all pairs, or EURUSD — useful context, but note it holds overnight, which this strategy structurally cannot, so treat it as informational rather than the primary bar to clear. |

**Aggregation across folds:** bootstrap confidence intervals on Sharpe and drawdown
across the 8–10 fold test sets, but see the fold-independence caveat in §13 —
these intervals are likely narrower than the true uncertainty given that adjacent
folds are correlated, not i.i.d. samples.
**Stress test:** re-run backtest with spread widened 2–3× to check cost sensitivity.
Chronological splits only, per fold — never shuffle days across train/val/test.

---

## 16. Reproducibility & Run Tracking

Needed specifically because the deflated Sharpe calculation (§15) depends on
knowing exactly how many trials were run — this isn't generic engineering hygiene,
it's a direct input to a metric already in this spec.

- **Seed management:** fix and log a random seed per run (data shuffling, network
  init, action sampling). Run each config across a small number of seeds (e.g., 3)
  so "did this config work" isn't confounded with "did this seed get lucky."
- **Run manifest (`RUN_MANIFEST.json`):** for every training run, log the config
  used, the fold, the seed, and the resulting test-set metrics. This is what
  `eval.py` reads to compute the trial count for deflated Sharpe (§15) — without
  it, that number is a guess.
- **Checkpoint naming:** `fold_{i}_seed_{s}_best.pt`, saved alongside its matching
  `RegimeExtractor` pickle (§21) so a checkpoint is never separated from the
  regime model it was trained with.

**Not covered here, and not blocking:** learning-rate scheduling, early stopping
on validation plateau, gradient accumulation, mixed-precision (FP16) training,
and TensorBoard logging are all reasonable engineering additions, but they're
generic PPO/deep-learning practices rather than something specific to this
trading system's correctness — add them as needed once the core loop (§14) is
working, rather than as a prerequisite to a first working version.

---

## 17. Path to Live Deployment

Not a full production spec — a short list of what changes between backtesting
and running this against a live feed, worth knowing before you get there:

- **Inference latency:** target a forward pass under ~50ms per bar on CPU. At an
  H1 bar frequency this has enormous headroom, but confirm it early — if the GNN
  + 28 heads is slower than that, something is likely wrong (e.g., an
  unintended CPU/GPU sync point), not that you need a faster machine.
- **Online regime inference:** in production there's no "fold" to fit on —
  maintain a rolling window of the last 100 bars, apply the *already-fitted*
  PCA/KMeans from the most recently promoted champion fold (§13) via
  `.transform()`, never `.fit()`, on live data.
- **Model versioning:** track which fold's policy + regime model pair is
  currently live, separately from which one is the current "champion" in
  the offline evaluation loop (§13) — promotion should be a deliberate,
  logged step, not implicit in which checkpoint happens to be loaded.
- **Monitoring:** track live Sharpe, win rate, and the residual-exposure metric
  (§15) on a rolling weekly basis; alert if they drop meaningfully below the
  backtested distribution for the currently deployed model. A live Sharpe
  persistently below the backtest's lower confidence bound (§13) is the signal
  to fall back to the previous champion, not to keep waiting it out.

---

## 18. Config Defaults — All Subsystems

Every previously-vague parameter across the spec, consolidated in one place —
this is what `config.py` (§21) should contain as its starting values.

| Subsystem | Parameter | Value |
|---|---|---|
| PPO | Learning rate | 3e-4 |
| PPO | GAE λ | 0.95 |
| PPO | Discount γ | **1.0 (undiscounted — see §3.3)**; do not use 0.99, episode is genuinely finite |
| PPO | Entropy coefficient | 0.01 |
| PPO | Clip range | 0.2 |
| PPO | Epochs per batch | 4 |
| PPO | Mini-batch size | 512 (or 128 days) |
| PPO | Optimizer | Adam |
| Aux loss | Weight `c3` | warm up 0.01 → 0.1 over 50 epochs (§5) |
| Aux loss | Target | log return × 100 (§5) |
| Reward | `α` (per-bar weight) | 0.05 starting point; consider 0.2–0.5 (§7) |
| Reward | `β` (terminal weight) | 1.0 |
| Reward | Turnover penalty `κ` | ~0.001, tune via grid search (§7) |
| Reward | Close-out penalty `μ` | 0.01 starting point (§7) |
| Reward | Differential Sharpe `ε` | 1e-6 (§7) |
| Reward | Differential Sharpe EMA `η` | 0.1 (§7) |
| Reward | `λ_dd` (drawdown penalty) | 0.5–1.0 |
| Reward | `λ_var` (variance penalty) | 0.25–0.5 |
| Action space | Bucket set | `{-2, -1, 0, +1, +2}` (§8) |
| Action space | `MAX_PAIR_EXPOSURE` | 2.0 lots (§8) |
| Action space | `vol_scale` clip range | [0.25, 4.0] (§8) |
| Safety | `VOL_BREAKER_WINDOW` | 20 bars (§9) |
| Safety | `VOL_BREAKER_MULTIPLIER` | 3.0 (§9) |
| Safety | VaR method | historical, 100-bar trailing (§9) |
| Safety | `VAR_CAP` | 0.02 (2% of portfolio) (§9) |
| Safety | `MAX_ABS_EXPOSURE` | 0.5 per currency (§9) |
| Regime | `N_REGIME_COMPONENTS` | 10 (§11) |
| Regime | `N_REGIME_CLUSTERS` | 5 (§11) |
| Regime | `MIN_REGIME_TRAINING_DAYS` | 100 (§11) |
| Folds | `TRAIN_YEARS_PER_FOLD` | 5 |
| Folds | `TEST_YEARS_PER_FOLD` | 1 |
| Folds | `STEP_YEARS` | 1 |
| Folds | `N_FOLDS` | 8 |
| Folds | Internal split | 80% train / 10% val / 10% test (§13) |
| Data | `DAILY_STOP_LOSS` | 0.02 (−2%) |

---

## 19. Summary Table

| Question | Answer |
|---|---|
| Is 26 years enough data? | Yes; consider using 2010–2026 for cleaner quality/liquidity. |
| Why a daily (24-bar) game? | Design choice for intraday focus and clean per-day evaluation — not a data constraint. |
| How does the agent know it's day trading? | `bars_remaining_norm` + `risk_budget_used` in state, seen by both policy and value heads, undiscounted (γ=1) return, and a close-out shaping penalty (§3). |
| How are regimes found? | PCA/SVD/ICA on cross-pair features → optional KMeans clustering → embedding added to state, re-fit per fold. |
| How is it trained? | PPO, 24-bar episodes sampled randomly within each fold's training window. |
| What hardware is needed? | A 16GB Colab GPU is more than sufficient (<1GB used). |
| How is it made robust? | Fold-parallel walk-forward training + champion/challenger promotion. |
| How is it evaluated? | Deflated Sharpe + bootstrap across folds (with a correlation caveat, §13) + spread-stress test + normalized end-of-day residual exposure check. |
| Is the data trustworthy before training starts? | Only after `validate_data_coverage()` (§2) confirms per-symbol coverage, alignment, and unit consistency. |
| Are all parameters concretely defined? | Yes — every previously-vague value (VaR cap, bucket-to-lots mapping, close-out penalty, etc.) has a starting default in §18. |
| What changes for live trading? | Online (transform-only) regime inference, model versioning, and rolling monitoring — see §17. |

---

## 20. Changelog

**v2 → v3**

| v2 | v3 | Reason |
|---|---|---|
| Episode length asserted as 24 bars | Explicitly framed as a design choice, with alternatives tabulated | Makes the tradeoff visible; easy to revisit if you want multi-day positions |
| Regime features mentioned but unspecified | Concrete PCA/SVD/ICA + KMeans pipeline, fit per fold | Turns a placeholder into an implementable step |
| Walk-forward retraining described generally | Concrete fold-parallel scheme with code and a feasibility table | Makes the retraining cadence and infra cost explicit instead of abstract |
| No hardware sizing | Colab 16GB GPU memory budget and training-time estimate | Confirms the whole system fits comfortably before you commit to building it |
| No explicit anti-leakage rule for regime fitting | Regime model re-fit per fold, train-only | Prevents future-information leakage across folds |

**v3 → v4**

| v3 | v4 | Reason |
|---|---|---|
| Time-to-close only implicit (episode mechanic forces flat at bar 24) | Explicit `bars_remaining_norm` in state, seen by policy and value heads | Lets the agent condition directly on urgency instead of inferring it from history |
| No explicit risk-budget signal | `risk_budget_used` (fraction of daily stop spent) added to state | Lets the policy anticipate the safety layer instead of only being clamped by it |
| γ = 0.99 discounting | γ = 1.0 (undiscounted), since the horizon is genuinely finite | Discounting mildly distorts terminal-bar incentives in a fixed-length episode |
| Value head time-blind | Value head takes the same `bars_remaining_norm` / `risk_budget_used` inputs as the policy | Keeps GAE advantage estimates precise near the close, where it matters most |
| Wind-down relied only on the forced-flatten mechanic | Added a small close-out shaping penalty in the last 1–2 bars | Makes winding down a learned habit that generalizes if episode length/max-hold change |
| No direct metric for "did time-awareness work" | Added end-of-day residual exposure metric (§15) | Gives a concrete, checkable signal instead of just trusting the design |

**v4 → v5**

| v4 | v5 | Reason |
|---|---|---|
| Data assumed clean and aligned | `validate_data_coverage()` step, effective-start-year logic (§2) | 28 pairs rarely have equal, gap-free history; this was silently assumed, not checked |
| Safety layer thresholds unspecified ("3× trailing average", "fixed ceiling") | Concrete `VOL_BREAKER_WINDOW=20`, `VAR_CAP=0.02`, historical-VaR method, explicit priority order (§9) | Spec wasn't implementable as written; these are blocking, not cosmetic, gaps |
| Action bucket → lot size mapping undefined | Explicit `vol_scale`-based formula, `MAX_PAIR_EXPOSURE`, mask-not-clip rule (§8) | Same reason — the action space couldn't actually be executed without this |
| Reward had no numerical-stability guard, no turnover formula, no μ default | `ε` in the Sharpe denominator, explicit turnover formula, `μ=0.01` starting value, α/β balance note (§7) | Prevents NaNs early in training and removes another unspecified constant |
| Folds had only train/test, no internal validation | 80/10/10 internal split per fold (§13) | Without it, early-stopping decisions leak information from the reported test set |
| Folds treated as 8–10 independent samples | Explicit fold-independence caveat and block-bootstrap recommendation (§13, §15) | Adjacent folds share regimes; treating them as i.i.d. overstates confidence |
| No live-deployment guidance | Concise deployment section: latency target, online regime inference, versioning, monitoring (§17) | Backtesting and live trading differ in specific, foreseeable ways worth planning for early |
| No reproducibility/trial-tracking mechanism | Run manifest + seed policy (§16), which the deflated Sharpe calculation (§15) now explicitly depends on | Deflated Sharpe needs a real trial count, not a guess |

---

## 21. Implementation: File Structure & Module Notes

A minimal, self-contained layout — one file per major spec component, so each
section above maps to something you can open, test, and debug independently.

```
forex_rl_v4/
├── config.py             # single source of truth: hyperparams, paths, constants
├── data_pipeline.py      # download → daily → RL dataset; validate_data_coverage() (§2); load_fold() for walk-forward
├── regime.py             # PCA/SVD/ICA + KMeans extractor (§11) — fit/save/load per fold
├── safety.py             # NEW — vol circuit breaker, VaR cap, exposure caps, hard flatten (§9)
├── env.py                # trading env with time-awareness (§3) — calls safety.py for masks
├── model.py              # GNN encoder + policy/value/aux heads (§6)
├── rollout.py            # NEW — episode collection into a buffer, separate from the PPO update
├── ppo_trainer.py        # PPO update: GAE with γ=1 (§5, §14), clipped loss, aux loss
├── fold_runner.py        # NEW — single-fold orchestration: fit regime → train → evaluate
├── train.py              # entrypoint: fold-parallel harness, calls fold_runner per fold (§13)
├── eval.py               # deflated Sharpe, bootstrap CIs, stress test, residual-exposure check (§15)
├── utils.py              # seeding, logging, checkpoint I/O, RUN_MANIFEST.json writer (§16)
├── tests/
│   ├── test_reward.py           # differential Sharpe + close-out penalty vs. known sequences
│   ├── test_regime_leakage.py   # confirm fit/transform split is leak-free per fold
│   └── test_env_masking.py      # action masks + safety layer under edge cases
├── checkpoints/           # policy weights + regime pickle, saved together per fold
└── requirements.txt
```

**Why split further than the original proposal:**

- **`safety.py` as its own module**, imported by both `env.py` and (later) any live
  trading harness. §9 requires the safety layer to "run identically at train and
  inference time" — that's only guaranteed if it's one shared module, not logic
  duplicated inside the environment and reimplemented again for live trading.
- **`rollout.py` separated from `ppo_trainer.py`.** Rollout collection (interacting
  with the env, building the buffer) and the PPO gradient update are different
  concerns; separating them lets you unit-test the PPO loss on a fixed, synthetic
  buffer without needing a live environment.
- **`fold_runner.py` separated from `train.py`.** `fold_runner.py` answers "how does
  one fold train end-to-end"; `train.py` answers "how do many folds run together."
  You'll spend most of early debugging time in the former — keep it callable and
  testable on its own before wrapping it in `mp.Pool`.

**Implementation hints (small things that commonly break this specific design):**

- **`config.py`** — `GAMMA = 1.0` is load-bearing (§3.3); add a comment or an
  assertion near it so a later "helpful" default-hyperparameter cleanup doesn't
  silently reintroduce 0.99. Also make sure `torch` is actually imported before
  `DEVICE = "cuda" if torch.cuda.is_available() else "cpu"` — easy to drop.
- **`data_pipeline.py`** — `load_fold()`'s date filtering compares fold-config
  date *strings* against a `dates` array; cast both sides to `np.datetime64`
  explicitly before comparing, or the mask can silently return an empty slice
  instead of raising an error.
- **`regime.py`** — when `method='svd'`, singular-vector sign is arbitrary and can
  flip between folds fit on different windows. Fix a sign convention (e.g., flip
  each component so its largest-magnitude loading is positive) before feeding the
  continuous embedding into the network, or the same regime can look different
  to the model across folds for no real reason.
- **`safety.py`** — write each check (vol breaker, VaR cap, exposure cap) as a
  pure function of `(state, config) → mask`, with no internal state of its own.
  That makes each one trivially unit-testable in isolation and safe to reuse
  unchanged in a future live-trading harness.
- **`env.py`** — the differential-Sharpe running stats (`A`, `B`) use an EMA
  update rate that shouldn't be a magic `0.1` buried in `_compute_reward`; pull
  it into `config.py` as `DIFFERENTIAL_SHARPE_ETA` so it's tunable and visible.
- **`model.py`** — `bars_remaining_norm` and `risk_budget_used` are per-episode
  scalars; make sure they're broadcast across all 28 per-pair heads (§6) rather
  than accidentally only reaching the value head — this is the easiest way to
  silently lose the §3 time-awareness benefit while everything still "runs."
- **`rollout.py`** — store `bars_remaining_norm` and `risk_budget_used` directly
  in each transition at collection time, rather than recomputing them from
  `bar_idx` later. `_current_bar` has already advanced by the time you'd
  recompute, which quietly shifts every value off by one bar.
- **`ppo_trainer.py`** — with `γ=1.0`, the GAE recursion drops the discount
  factor entirely: `A_t = Σ λ^k δ_{t+k}` (no `γ` term multiplying `δ`). If you
  start from a copy-pasted GAE implementation, check it doesn't still multiply
  by a hardcoded `gamma` from an infinite-horizon example.
- **`fold_runner.py`** — always checkpoint the fitted `RegimeExtractor` (§11)
  alongside the policy weights for that fold. You need both together to run
  inference later; a policy without its matching regime model is unusable.
- **`eval.py`** — the deflated Sharpe ratio needs the number of implicit trials
  (configs/hyperparameters/folds tried) as an input. Log this count in a run
  manifest as you go rather than hardcoding a guess in `eval.py` after the fact.
- **`train.py`** — `mp.Pool(processes=4)` parallelizes CPU processes, not GPU
  compute; on a single-GPU Colab instance this does **not** give you 4x training
  speed and can cause CUDA context issues across forked processes. On one GPU,
  run folds sequentially (8 folds × 2–4 hrs ≈ 1–2 days) or use
  `torch.multiprocessing` with the `spawn` start method if you genuinely have
  multiple GPUs to spread folds across.

---

## 22. Suggested Build Order

1. `validate_data_coverage()` (§2) — run once before anything else touches the data.
2. Environment + safety layer + action masking (§8, §9, §10) — correctness before learning.
3. Reward function (§7) in isolation — unit test differential Sharpe (with `ε`), turnover, and the close-out penalty against known sequences.
4. Regime pipeline (§11) on a single fold — confirm the min-data check, SVD sign-fix, and fit/transform split are all correct.
5. Add time-awareness features (§3) to the state and confirm both policy and value heads receive them.
6. Network (§6) without the aux head — baseline PPO loop with γ=1, single fold (with its 80/10/10 internal split, §13), end-to-end.
7. Add the auxiliary loss (§5) once the base loop is stable, with its warm-up schedule.
8. Wrap into the fold-parallel harness (§13, §14) with run-manifest logging (§16) and run all folds sequentially.
9. Aggregate and evaluate (§15), including the end-of-day residual exposure check and the fold-correlation caveat, before considering live deployment (§17).
