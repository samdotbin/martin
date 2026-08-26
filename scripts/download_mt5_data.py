"""
scripts/download_mt5_data.py — pulls real H1 OHLCV+spread data for all
config.PAIRS straight from a running MetaTrader5 terminal via the official
MetaTrader5 Python package, and writes it in the exact CSV schema
data_pipeline.load_raw_csv() expects: timestamp, open, high, low, close,
volume, spread — one file per pair at config.DATA_DIR/{PAIR}.csv.

Requires:
  - The MT5 terminal application running on THIS machine and logged into a
    broker account that has all of config.PAIRS available.
  - `pip install MetaTrader5` (Windows only — the package talks to the
    terminal over its native API, so this script only runs on the same
    Windows box as the terminal; it can't run in a Linux sandbox).
  - It does NOT matter where this script/repo lives on disk (it does not
    need to be inside the terminal's MQL5 folder) — the package connects to
    the running terminal process, not to files on disk.

IMPORTANT — server time vs UTC:
MT5 bar timestamps are in the broker's SERVER time, which is usually NOT
UTC (commonly UTC+2/UTC+3, but it varies by broker and by daylight saving).
Every date boundary in this codebase (fold splits, day-of-week checks, the
64-bar lookback's day contiguity) assumes true UTC, so getting this offset
wrong will silently misalign all of it. This script auto-detects the offset
by comparing the terminal's latest tick time against this machine's system
clock and rounds to the nearest hour — treat that as a starting guess, not
a guarantee. Print it, sanity-check it against what your broker publishes
(most state it in their contract specs or account docs), and override with
--server-utc-offset-hours if it's wrong or if this machine's clock can't be
trusted.

Usage:
    python scripts/download_mt5_data.py --years 6
    python scripts/download_mt5_data.py --start 2015-01-01 --end 2026-08-01
    python scripts/download_mt5_data.py --pairs EURUSD GBPUSD   # subset, for a quick test
    python scripts/download_mt5_data.py --years 6 --suffix .a   # broker appends a suffix to symbol names
    python scripts/download_mt5_data.py --years 6 --server-utc-offset-hours 3  # override auto-detect
"""
import argparse
import os
import sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

import config as cfg

try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None


def _resolve_symbol(pair: str, suffix: str) -> str:
    """Some brokers append a suffix to standard names (EURUSD.a, EURUSDm,
    EURUSD_i, ...). If the plain name isn't found and no --suffix was given,
    scan the broker's symbol list for anything starting with the pair name
    and use the first match, so a broker-specific suffix doesn't silently
    produce 28 failures."""
    candidate = f"{pair}{suffix}"
    if mt5.symbol_info(candidate) is not None:
        return candidate
    if suffix:
        return candidate  # explicit suffix given but didn't match — fail loudly downstream, don't guess
    all_symbols = mt5.symbols_get()
    matches = [s.name for s in (all_symbols or []) if s.name.startswith(pair)]
    return matches[0] if matches else candidate


def detect_server_utc_offset_hours() -> int:
    """Best-effort auto-detect: latest tick's server time minus this
    machine's UTC clock, rounded to the nearest hour. See module docstring —
    verify this against your broker's documented server timezone."""
    symbols = mt5.symbols_get()
    if not symbols:
        return 0
    tick = mt5.symbol_info_tick(symbols[0].name)
    if tick is None or tick.time == 0:
        return 0
    server_time = datetime.fromtimestamp(tick.time, tz=timezone.utc)
    now_utc = datetime.now(timezone.utc)
    return round((server_time - now_utc).total_seconds() / 3600)


def download_pair(pair: str, utc_from: datetime, utc_to: datetime, out_dir: str,
                   suffix: str, offset_hours: int) -> int:
    symbol = _resolve_symbol(pair, suffix)
    if not mt5.symbol_select(symbol, True):
        raise RuntimeError(
            f"could not select symbol '{symbol}' in Market Watch — enable it "
            f"there (right-click -> Show All), or check --suffix"
        )

    # copy_rates_range takes naive datetimes interpreted as SERVER time.
    server_from = (utc_from + timedelta(hours=offset_hours)).replace(tzinfo=None)
    server_to = (utc_to + timedelta(hours=offset_hours)).replace(tzinfo=None)

    rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_H1, server_from, server_to)
    if rates is None or len(rates) == 0:
        raise RuntimeError(
            f"({symbol}) copy_rates_range returned no data for this range — "
            f"check the broker's history depth for this symbol, error={mt5.last_error()}"
        )

    df = pd.DataFrame(rates)
    # MT5 fields: time, open, high, low, close, tick_volume, spread, real_volume.
    # `time` is epoch seconds in SERVER time — convert back to true UTC.
    ts_utc = pd.to_datetime(df["time"], unit="s", utc=True) - pd.Timedelta(hours=offset_hours)

    out = pd.DataFrame({
        "timestamp": ts_utc,
        "open": df["open"],
        "high": df["high"],
        "low": df["low"],
        "close": df["close"],
        "volume": df["tick_volume"],   # tick volume; real_volume is usually 0 on FX feeds
        "spread": df["spread"],        # broker points, per MT5 convention for this symbol
    })
    out = out.sort_values("timestamp").drop_duplicates(subset="timestamp").reset_index(drop=True)
    out.to_csv(os.path.join(out_dir, f"{pair}.csv"), index=False)
    return len(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--years", type=int, default=6, help="years of history ending now (ignored if --start given)")
    parser.add_argument("--start", type=str, default=None, help="YYYY-MM-DD, UTC, overrides --years")
    parser.add_argument("--end", type=str, default=None, help="YYYY-MM-DD, UTC, defaults to now")
    parser.add_argument("--pairs", type=str, nargs="+", default=None, help="subset of config.PAIRS, for a quick test")
    parser.add_argument("--suffix", type=str, default="", help="broker symbol suffix, e.g. '.a' or 'm', if plain names like 'EURUSD' aren't found")
    parser.add_argument("--server-utc-offset-hours", type=int, default=None, help="override the auto-detected server-time-vs-UTC offset")
    parser.add_argument("--out-dir", type=str, default=cfg.DATA_DIR)
    args = parser.parse_args()

    if mt5 is None:
        print("ERROR: the MetaTrader5 package isn't installed here. Run:")
        print("    pip install MetaTrader5")
        sys.exit(1)

    if not mt5.initialize():
        print(f"ERROR: mt5.initialize() failed — is the MT5 terminal running and logged into an account? error={mt5.last_error()}")
        sys.exit(1)

    try:
        offset_hours = args.server_utc_offset_hours
        if offset_hours is None:
            offset_hours = detect_server_utc_offset_hours()
            print(f"Auto-detected server-time offset: UTC{offset_hours:+d}h "
                  f"(verify against your broker's docs; override with --server-utc-offset-hours if wrong)")
        else:
            print(f"Using server-time offset: UTC{offset_hours:+d}h (explicit override)")

        utc_to = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc) if args.end else datetime.now(timezone.utc)
        utc_from = (
            datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            if args.start else utc_to.replace(year=utc_to.year - args.years)
        )

        os.makedirs(args.out_dir, exist_ok=True)
        pairs = args.pairs or cfg.PAIRS

        print(f"Downloading {len(pairs)} pair(s), {utc_from.date()} -> {utc_to.date()} UTC, H1 bars, to {args.out_dir}")
        failures = []
        for pair in pairs:
            try:
                n = download_pair(pair, utc_from, utc_to, args.out_dir, args.suffix, offset_hours)
                print(f"  {pair}: {n} bars")
            except RuntimeError as e:
                print(f"  {pair}: FAILED — {e}")
                failures.append(pair)

        if failures:
            print(f"\n{len(failures)} pair(s) failed: {failures}")
            print("Common causes: symbol not visible in Market Watch (right-click -> Show All), "
                  "a broker-specific suffix on symbol names (try --suffix), or the broker's "
                  "history for that symbol doesn't go back this far (try a shorter --years).")
            sys.exit(1)
        else:
            print(f"\nAll {len(pairs)} pair(s) downloaded.")
            print("Next: python data_pipeline.py   (sanity-checks coverage across all pairs)")
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    main()
