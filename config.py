"""
config.py — single source of truth for hyperparameters, paths, and constants.

Every default here traces back to spec §18 (Config Defaults — All Subsystems).
Do not scatter magic numbers in other modules; import them from here.
"""
import os
import torch  # NOTE: must be imported before DEVICE is computed — easy to drop (§21 hint)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
# Overridable via env var so multiple train.py processes can share ONE code+
# data checkout (data/raw is read-only, safe to share) while each writes
# checkpoints/manifest to its own directory — avoids a RunManifest read-
# modify-write race if two shards' processes run concurrently on one
# machine (see train.py --shard-index/--shard-count).
CHECKPOINT_DIR = os.environ.get("FOREX_RL_CHECKPOINT_DIR") or os.path.join(PROJECT_ROOT, "checkpoints")
RUNS_DIR = os.environ.get("FOREX_RL_RUNS_DIR") or os.path.join(PROJECT_ROOT, "runs")
RUN_MANIFEST_PATH = os.path.join(RUNS_DIR, "RUN_MANIFEST.json")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# Universe
# ---------------------------------------------------------------------------
CURRENCIES = ["USD", "EUR", "GBP", "JPY", "CHF", "AUD", "CAD", "NZD"]  # 8 currencies
PAIRS = [
    "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD",
    "EURGBP", "EURJPY", "EURCHF", "EURAUD", "EURCAD", "EURNZD",
    "GBPJPY", "GBPCHF", "GBPAUD", "GBPCAD", "GBPNZD",
    "AUDJPY", "AUDCHF", "AUDCAD", "AUDNZD",
    "CADJPY", "CADCHF", "NZDJPY", "NZDCHF", "NZDCAD", "CHFJPY",
]  # 28 pairs spanning the complete graph over 8 currencies
assert len(PAIRS) == 28, "spec assumes exactly 28 pairs / 8 currencies (§4, §6)"

# ---------------------------------------------------------------------------
# Episode design (§1)
# ---------------------------------------------------------------------------
BARS_PER_EPISODE = 24          # Option A — 1 trading day of H1 bars
LOOKBACK_BARS = 64             # state includes last 64 bars of edge features (§4)
MAX_HOLD_BARS = 8              # optional forced close after N bars (§10); None to disable

# ---------------------------------------------------------------------------
# Data quality (§2)
# ---------------------------------------------------------------------------
MIN_COVERAGE_PCT = 0.90         # a symbol must have >=90% bar coverage to count
MAX_MISSING_BARS_PER_DAY = 1    # days with more than this many missing bars are dropped
DOW_UNIFORMITY_TOLERANCE = 0.08 # fractional deviation from uniform day-of-week allowed
DEFAULT_HISTORY_START = "2000-01-01"
CONTIGUITY_MAX_GAP_HOURS = 2    # §10 — max allowed gap between consecutive bars in a day

# ---------------------------------------------------------------------------
# Time-awareness (§3)
# ---------------------------------------------------------------------------
CLOSE_OUT_PENALTY_BARS_REMAINING = 2 / BARS_PER_EPISODE  # trigger window (last ~2 bars)

# ---------------------------------------------------------------------------
# PPO (§5, §18)
# ---------------------------------------------------------------------------
LEARNING_RATE = 3e-4
GAE_LAMBDA = 0.95
GAMMA = 1.0
# GAMMA = 1.0 is load-bearing (§3.3): the episode is genuinely finite (24 bars,
# forced close), so discounting would distort terminal-bar incentives. A later
# "helpful" hyperparameter cleanup must NOT silently reintroduce 0.99.
assert GAMMA == 1.0, "spec §3.3: episode is finite — do not discount with gamma<1"
ENTROPY_COEF = 0.01
CLIP_RANGE = 0.2
PPO_EPOCHS_PER_BATCH = 4
MINI_BATCH_SIZE = 512           # or 128 days worth of transitions
EPISODES_PER_ROLLOUT = 256      # buffer: 256 episodes x 24 steps = 6144 transitions
ROLLOUT_N_ENVS = 32             # parallel env "lanes" batched into one model
                                 # forward call during rollout collection —
                                 # collapses the Python-loop-bound per-step
                                 # cost by ~N. Tunable higher (64-128) on a
                                 # GPU (Colab): the model is small (2-5M
                                 # params) and 16GB is ample headroom.

# Auxiliary loss (§5)
AUX_WEIGHT_START = 0.01
AUX_WEIGHT_END = 0.10
AUX_WARMUP_EPOCHS = 50

# ---------------------------------------------------------------------------
# Reward function (§7, §18)
# ---------------------------------------------------------------------------
REWARD_ALPHA = 0.05             # per-bar (differential Sharpe) weight — try 0.2-0.5 if unstable
REWARD_BETA = 1.0               # terminal weight
TURNOVER_KAPPA = 0.001          # turnover penalty, tune via grid search
CLOSE_OUT_MU = 0.01             # close-out shaping penalty weight, starting point
DIFF_SHARPE_EPS = 1e-6          # epsilon in differential Sharpe denominator
DIFFERENTIAL_SHARPE_ETA = 0.1   # EMA update rate for running A, B stats — tunable, not magic
LAMBDA_DD = 0.75                # drawdown penalty, 0.5-1.0 range, midpoint default
LAMBDA_VAR = 0.375              # variance penalty, 0.25-0.5 range, midpoint default
STOP_LOSS_TERMINAL_PENALTY = -10.0

# ---------------------------------------------------------------------------
# Action space & bucket semantics (§8, §18)
# ---------------------------------------------------------------------------
ACTION_BUCKETS = [-2, -1, 0, 1, 2]
BASE_LOT = 1.0
MAX_PAIR_EXPOSURE = 2.0         # lots; equals max(bucket) * BASE_LOT by construction
VOL_SCALE_CLIP = (0.25, 4.0)

# ---------------------------------------------------------------------------
# Safety layer (§9, §18) — runs identically at train and inference time
# ---------------------------------------------------------------------------
VOL_BREAKER_WINDOW = 20
VOL_BREAKER_MULTIPLIER = 3.0
VAR_METHOD = "historical"       # percentile of trailing window, no distributional assumption
VAR_WINDOW = 100
VAR_CAP = 0.02                  # 2% of portfolio
MAX_ABS_EXPOSURE = 0.5          # per currency, net units
DAILY_STOP_LOSS = 0.02          # -2%
SAFETY_CHECK_PRIORITY = ["exposure_caps", "max_hold", "vol_circuit_breaker", "var_cap", "forced_flatten"]

# ---------------------------------------------------------------------------
# Regime extraction (§11, §18)
# ---------------------------------------------------------------------------
REGIME_METHOD = "pca"           # 'pca' | 'svd' | 'ica'
N_REGIME_COMPONENTS = 10
N_REGIME_CLUSTERS = 5
MIN_REGIME_TRAINING_DAYS = 100

# ---------------------------------------------------------------------------
# Network (§6)
# ---------------------------------------------------------------------------
EDGE_FEATURES = 4               # log return, HL range, volume z, spread z (per pair, per bar)
TEMPORAL_CHANNELS = 32
GNN_HIDDEN_DIM = 64
GNN_MESSAGE_PASSING_LAYERS = 3
POLICY_HEAD_HIDDEN = 64

# ---------------------------------------------------------------------------
# Walk-forward folds (§13, §18)
# ---------------------------------------------------------------------------
TRAIN_YEARS_PER_FOLD = 5
TEST_YEARS_PER_FOLD = 1
STEP_YEARS = 1
# 12, not the spec's original 8: with effective_start_year=2010 and real
# data now running through 2026-08, N_FOLDS=8 only covered test windows
# through 2022-2023 -- 2023-2026 was silently never trained OR tested on.
# 12 folds reaches a full 2025 test year (fold 10) plus one partial fold
# using through-Aug-2026 data (fold 11) -- everything currently on disk.
N_FOLDS = 12
INTERNAL_SPLIT = (0.8, 0.1, 0.1)  # train / val / test within a fold's training window

# ---------------------------------------------------------------------------
# Reproducibility (§16)
# ---------------------------------------------------------------------------
SEEDS = [13, 42, 7, 99]           # small number of seeds per config, per §16 -- freely editable, any distinct ints work
