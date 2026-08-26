"""
data_pipeline.py — loads already-downloaded per-pair CSVs into the aligned
edge-feature tensor the rest of the pipeline trains on. Does NOT fetch data
itself — see scripts/download_mt5_data.py (real, via a running MT5
terminal) or scripts/generate_synthetic_data.py (synthetic, for a smoke
test) to populate config.DATA_DIR first.

Implements:
  - validate_data_coverage()  (§2)  — run once before fold construction
  - contiguity validation      (§10) — per-day gap check before adding to episode pool
  - load_fold()                (§13/§21) — walk-forward date-range slicing with explicit
                                np.datetime64 casting (a common silent-empty-slice bug)

Expected raw layout: config.DATA_DIR/{PAIR}.csv with columns
  timestamp, open, high, low, close, volume, spread
one row per H1 bar, one file per pair in config.PAIRS.

Run this file directly for a coverage sanity check across all pairs:
    python data_pipeline.py
"""
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import config
from utils import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def _raw_path(pair: str, data_dir: str = None) -> str:
    data_dir = data_dir or config.DATA_DIR
    return os.path.join(data_dir, f"{pair}.csv")


def load_raw_csv(pair: str, data_dir: str = None) -> pd.DataFrame:
    path = _raw_path(pair, data_dir)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"missing raw data for {pair} at {path} — run "
            f"scripts/download_mt5_data.py for real data (needs the MT5 "
            f"terminal running and logged in), or "
            f"scripts/generate_synthetic_data.py for a smoke-test dataset"
        )
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# §2 — Data quality & coverage validation
# ---------------------------------------------------------------------------
@dataclass
class CoverageReport:
    per_symbol_start: dict = field(default_factory=dict)
    per_symbol_coverage_pct: dict = field(default_factory=dict)
    effective_start_year: int = None
    dow_distribution: dict = field(default_factory=dict)
    dow_warning: bool = False
    unit_mismatch_flags: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)


def _expected_h1_bars(start, end) -> int:
    # Rough FX-market bar count: the market is open Sun 22:00 UTC -> Fri
    # 22:00 UTC, i.e. 120 of every 168 hours. A flat "5.5 trading days"
    # assumption (132/168) overstates this by ~10% and made every pair,
    # including ones with genuinely clean data, read ~90% instead of ~99%
    # coverage. Used only as a coverage denominator, not for exact accounting.
    total_hours = (end - start).total_seconds() / 3600
    open_hours_per_week = 120
    return int(total_hours * (open_hours_per_week / (24 * 7)))


def validate_data_coverage(pairs=None, data_dir=None, config_module=config) -> CoverageReport:
    """
    Run once, before fold construction. Reports each symbol's effective start
    date and % bar coverage, computes the effective start year (earliest year
    in which ALL pairs have >=MIN_COVERAGE_PCT coverage), flags day-of-week
    skew, and asserts a consistent pip/point convention across pairs.
    """
    pairs = pairs or config_module.PAIRS
    report = CoverageReport()

    frames = {}
    for pair in pairs:
        try:
            df = load_raw_csv(pair, data_dir)
        except FileNotFoundError as e:
            report.warnings.append(str(e))
            continue
        frames[pair] = df
        start = df["timestamp"].iloc[0]
        end = df["timestamp"].iloc[-1]
        expected = max(_expected_h1_bars(start, end), 1)
        coverage = min(len(df) / expected, 1.0)
        report.per_symbol_start[pair] = start
        report.per_symbol_coverage_pct[pair] = coverage
        if coverage < config_module.MIN_COVERAGE_PCT:
            report.warnings.append(
                f"{pair}: coverage {coverage:.1%} below "
                f"{config_module.MIN_COVERAGE_PCT:.0%} threshold"
            )

    if not frames:
        report.warnings.append("no raw data found for any pair — cannot validate coverage")
        return report

    # Effective start year: earliest year where ALL pairs clear the coverage bar.
    candidate_years = sorted({d.year for d in report.per_symbol_start.values()})
    effective_year = None
    for year in candidate_years:
        ok = True
        for pair, df in frames.items():
            sub = df[df["timestamp"].dt.year >= year]
            if len(sub) == 0:
                ok = False
                break
            start_y = pd.Timestamp(f"{year}-01-01", tz="UTC")
            end_y = df["timestamp"].iloc[-1]
            expected = max(_expected_h1_bars(start_y, end_y), 1)
            cov = min(len(sub) / expected, 1.0)
            if cov < config_module.MIN_COVERAGE_PCT:
                ok = False
                break
        if ok:
            effective_year = year
            break
    report.effective_start_year = effective_year or max(candidate_years)

    # Day-of-week distribution check (catches systematic feed gaps).
    # Only Mon-Thu are genuinely expected to be uniform full trading days —
    # Friday legitimately gets fewer bars (market closes ~22:00 UTC) and
    # Sunday only a couple hours (market opens ~22:00 UTC), so including
    # either in a "should be uniform" comparison flags the normal shape of
    # the FX week as a fake anomaly every single time.
    all_days = pd.concat(
        [df["timestamp"].dt.dayofweek for df in frames.values()], ignore_index=True
    )
    dow_counts = all_days.value_counts(normalize=True).sort_index()
    report.dow_distribution = dow_counts.to_dict()
    weekday_counts = dow_counts.reindex(range(4), fill_value=0)  # Mon-Thu only
    if len(weekday_counts) == 4 and weekday_counts.max() > 0:
        deviation = (weekday_counts.max() - weekday_counts.min()) / weekday_counts.mean()
        if deviation > config_module.DOW_UNIFORMITY_TOLERANCE:
            report.dow_warning = True
            report.warnings.append(
                f"Mon-Thu day-of-week distribution deviates from uniform by "
                f"{deviation:.1%} (tolerance {config_module.DOW_UNIFORMITY_TOLERANCE:.0%}) — "
                f"check for a broker feed dropping a specific weekday "
                f"(Fri/Sun are excluded from this check — they're legitimately "
                f"shorter trading days, not a feed gap)"
            )

    # Pip/point convention consistency: compare median absolute bar-to-bar
    # move across pairs; a silent 10x unit mismatch shows up as an outlier.
    # JPY-quoted pairs (XXXJPY) are legitimately quoted 2 decimal places
    # (pip=0.01) vs 4 decimals (pip=0.0001) for everything else — a ~100x
    # raw-price-move gap that is a real FX quoting convention, not a data
    # bug. Compare JPY-quoted and non-JPY-quoted pairs within their own
    # group so that convention doesn't itself trip the mismatch check.
    median_moves = {}
    for pair, df in frames.items():
        moves = df["close"].diff().abs()
        median_moves[pair] = float(moves.median())
    jpy_quoted = {p: v for p, v in median_moves.items() if p.endswith("JPY")}
    non_jpy = {p: v for p, v in median_moves.items() if not p.endswith("JPY")}
    for group in (jpy_quoted, non_jpy):
        vals = np.array(list(group.values()))
        vals = vals[vals > 0]
        if len(vals) > 1:
            med = np.median(vals)
            for pair, v in group.items():
                if v > 0 and (v / med > 7 or v / med < 1 / 7):
                    report.unit_mismatch_flags[pair] = v / med
                    report.warnings.append(
                        f"{pair}: median bar move is {v / med:.1f}x the "
                        f"same-quote-convention median — possible pip/point "
                        f"unit mismatch, check before merging"
                    )

    for w in report.warnings:
        logger.warning(w)

    return report


# ---------------------------------------------------------------------------
# §10 — contiguity validation
# ---------------------------------------------------------------------------
def contiguous_days(df: pd.DataFrame, bars_per_episode: int = None) -> pd.DatetimeIndex:
    """
    Returns the set of calendar-day start timestamps (00:00 UTC) whose 24 bars
    have consecutive timestamps with no gap greater than CONTIGUITY_MAX_GAP_HOURS.
    A day straddling a holiday or outage is dropped here, not silently trained
    on with a corrupted lookback window (§2, §10).
    """
    bars_per_episode = bars_per_episode or config.BARS_PER_EPISODE
    max_gap = pd.Timedelta(hours=config.CONTIGUITY_MAX_GAP_HOURS)

    df = df.copy()
    df["day"] = df["timestamp"].dt.floor("D")
    good_days = []
    for day, group in df.groupby("day"):
        group = group.sort_values("timestamp")
        if len(group) < bars_per_episode:
            continue
        window = group.iloc[:bars_per_episode]
        gaps = window["timestamp"].diff().dropna()
        if (gaps > max_gap).any():
            continue
        good_days.append(day)
    return pd.DatetimeIndex(sorted(good_days))


def contiguous_day_starts(timestamps: np.ndarray, bars_per_episode: int = None) -> np.ndarray:
    """
    Global-index counterpart to contiguous_days(): given the FULL,
    fold-independent timestamps array (not a fold-sliced sub-array), returns
    the absolute integer index into that array of the first bar of every
    calendar day whose `bars_per_episode` bars have consecutive timestamps
    with no gap > CONTIGUITY_MAX_GAP_HOURS.

    Use this (via fold_runner) instead of contiguous_days() + a sliced
    per-split array when building TradingEnv instances: operating on the
    full array means the 64-bar lookback for a day near the start of a
    fold's val/test window can still reach back into real prior-day bars
    from before that boundary, instead of zero-padding — matching §10's
    "lookback spans the previous day on purpose."
    """
    bars_per_episode = bars_per_episode or config.BARS_PER_EPISODE
    max_gap = pd.Timedelta(hours=config.CONTIGUITY_MAX_GAP_HOURS)

    ts = pd.to_datetime(timestamps)
    df = pd.DataFrame({"timestamp": ts})
    df["day"] = df["timestamp"].dt.floor("D")

    starts = []
    for day, group in df.groupby("day"):
        group = group.sort_values("timestamp")
        if len(group) < bars_per_episode:
            continue
        window = group.iloc[:bars_per_episode]
        gaps = window["timestamp"].diff().dropna()
        if (gaps > max_gap).any():
            continue
        starts.append(int(window.index[0]))  # absolute position in the full array
    return np.array(sorted(starts), dtype=int)


# ---------------------------------------------------------------------------
# §13/§21 — fold slicing
# ---------------------------------------------------------------------------
def _naive_ns(x) -> np.datetime64:
    """Coerce a tz-aware or tz-naive/string timestamp to a tz-naive ns
    datetime64, without numpy's silent-tz-drop warning (compare like with
    like by explicitly stripping tz on the pandas side first)."""
    ts = pd.Timestamp(x)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return np.datetime64(ts, "ns")


def load_fold(dates: np.ndarray, values: np.ndarray, start: str, end: str):
    """
    Slice `values` (aligned to `dates`) to [start, end). Both sides are cast
    explicitly to tz-naive np.datetime64[ns] before comparison — comparing
    fold-config date *strings* against a datetime64 array can silently
    return an empty slice instead of raising an error (§21 implementation
    hint), and comparing tz-aware timestamps directly against raw
    datetime64[ns] throws a tz-drop warning, so both sides are normalized
    the same way first.
    """
    dates_idx = pd.DatetimeIndex(dates)
    if dates_idx.tz is not None:
        dates_idx = dates_idx.tz_convert("UTC").tz_localize(None)
    dates_naive = dates_idx.values.astype("datetime64[ns]")

    start_dt = _naive_ns(start)
    end_dt = _naive_ns(end)
    mask = (dates_naive >= start_dt) & (dates_naive < end_dt)
    if not mask.any():
        raise ValueError(
            f"load_fold produced an empty slice for [{start}, {end}) — "
            f"check the fold config dates against the available data range "
            f"({dates_naive.min()} .. {dates_naive.max()})"
        )
    return values[mask], dates[mask]


def make_fold_configs(effective_start_year: int, config_module=config):
    """Builds the N_FOLDS walk-forward fold definitions from §13."""
    folds = []
    train_years = config_module.TRAIN_YEARS_PER_FOLD
    test_years = config_module.TEST_YEARS_PER_FOLD
    step = config_module.STEP_YEARS
    for i in range(config_module.N_FOLDS):
        train_start_year = effective_start_year + i * step
        train_end_year = train_start_year + train_years
        test_end_year = train_end_year + test_years
        folds.append(
            {
                "fold_id": i,
                "train_start": f"{train_start_year}-01-01",
                "train_end": f"{train_end_year}-01-01",
                "test_start": f"{train_end_year}-01-01",
                "test_end": f"{test_end_year}-01-01",
            }
        )
    return folds


# ---------------------------------------------------------------------------
# Daily edge-feature dataset (§4 state features 0-3: log return, HL range,
# volume z-score, spread z-score, per pair, per bar)
# ---------------------------------------------------------------------------
def build_edge_feature_tensor(pairs=None, data_dir=None) -> tuple:
    """
    Returns (feature_tensor, timestamps):
      feature_tensor: (n_bars, n_pairs, EDGE_FEATURES) float32
      timestamps:     (n_bars,) aligned across all pairs (inner join on timestamp)
    """
    pairs = pairs or config.PAIRS
    frames = {p: load_raw_csv(p, data_dir) for p in pairs}

    common_ts = None
    for df in frames.values():
        ts = pd.Index(df["timestamp"])
        common_ts = ts if common_ts is None else common_ts.intersection(ts)
    common_ts = common_ts.sort_values()

    n_bars = len(common_ts)
    n_pairs = len(pairs)
    feats = np.zeros((n_bars, n_pairs, config.EDGE_FEATURES), dtype=np.float32)

    for j, pair in enumerate(pairs):
        df = frames[pair].set_index("timestamp").reindex(common_ts)
        log_ret = np.log(df["close"]).diff().fillna(0.0).to_numpy()
        hl_range = (df["high"] - df["low"]).to_numpy()
        vol = df["volume"].to_numpy() if "volume" in df else np.zeros(n_bars)
        spread = df["spread"].to_numpy() if "spread" in df else np.zeros(n_bars)

        def zscore(x):
            std = np.nanstd(x)
            return (x - np.nanmean(x)) / std if std > 1e-12 else np.zeros_like(x)

        feats[:, j, 0] = log_ret
        feats[:, j, 1] = zscore(hl_range)
        feats[:, j, 2] = zscore(vol)
        feats[:, j, 3] = zscore(spread)

    feats = np.nan_to_num(feats, nan=0.0)
    return feats, common_ts.to_numpy()


if __name__ == "__main__":
    # Running this file directly used to do nothing at all — no output, no
    # error, just exit — because it only ever defined functions. This gives
    # it something to actually do: a coverage sanity check across all pairs.
    report = validate_data_coverage()

    if not report.per_symbol_start:
        print(f"No raw CSVs found in {config.DATA_DIR}\n")
        print("Get real data (needs MT5 terminal running + logged in):")
        print("    python scripts/download_mt5_data.py --years 6")
        print("Or a synthetic smoke-test dataset:")
        print("    python scripts/generate_synthetic_data.py --years 6")
        raise SystemExit(1)

    print(f"Found data for {len(report.per_symbol_start)}/{len(config.PAIRS)} pairs in {config.DATA_DIR}\n")
    for pair in config.PAIRS:
        if pair in report.per_symbol_start:
            start = report.per_symbol_start[pair]
            cov = report.per_symbol_coverage_pct[pair]
            flag = "  <-- below coverage threshold" if cov < config.MIN_COVERAGE_PCT else ""
            print(f"  {pair}: from {start.date()}, {cov:.1%} coverage{flag}")
        else:
            print(f"  {pair}: MISSING")

    print(f"\nEffective start year (earliest year all pairs clear "
          f"{config.MIN_COVERAGE_PCT:.0%} coverage): {report.effective_start_year}")

    if report.dow_warning:
        print("\nDay-of-week distribution looks skewed — see warnings below.")

    if report.warnings:
        print(f"\n{len(report.warnings)} warning(s):")
        for w in report.warnings:
            print(f"  - {w}")
    else:
        print("\nNo coverage warnings.")
