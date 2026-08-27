"""
scripts/periodic.py — shared "run this on a schedule, tolerate failures"
loop. Extracted after having to manually re-apply the same
try/except-around-the-whole-body shape to colab_train.ipynb's background
loops more than once (checkpoint auto-push, then again for the heartbeat/
Telegram split) — and after an earlier version of one of those loops read
a required token BEFORE its own try block, so one missing-file error
raised uncaught and killed the whole background thread for the rest of
the session (see commit 8e59e56). Centralizing the mechanism here means
that mistake class can't recur silently: any new periodic background task
calls run_forever() and automatically gets "never let one bad cycle kill
the loop" for free, instead of a caller having to remember to write the
try/except themselves.
"""
import time


def run_forever(fn, interval_seconds: float, label: str = "task") -> None:
    """Calls fn() repeatedly, sleeping interval_seconds between calls. ANY
    exception fn() raises (a missing file, a network blip, a transient API
    error) is caught, printed, and the loop continues to the next cycle —
    it never propagates out and kills the thread. Meant to be the target
    of its own daemon thread; never returns."""
    while True:
        try:
            fn()
        except Exception as e:
            print(f'[{time.strftime("%H:%M:%S")}] {label} failed (will retry in '
                  f'{interval_seconds:.0f}s): {e}')
        time.sleep(interval_seconds)
