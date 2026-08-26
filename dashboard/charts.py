"""
dashboard/charts.py — shared Altair chart builders. Pure presentation, no
data access (that's lib.py) — kept separate so the charting code isn't
duplicated across the three pages that all show OHLC candlesticks.

Weekend gaps: a true temporal (:T) x-axis leaves dead whitespace over
Saturday/most of Sunday since no bars exist then, compressing the actual
trading hours into less room than they deserve. Every chart here instead
uses an ORDINAL axis over a formatted label column, in the data's own row
order — weekends (and any other gap) simply aren't categories, so they
don't take up space. Real timestamps still drive tooltips.
"""
import altair as alt
import numpy as np
import pandas as pd

UP_COLOR = "#26a69a"
DOWN_COLOR = "#ef5350"


def with_bar_labels(df: pd.DataFrame, fmt: str = "%b %d %H:%M") -> pd.DataFrame:
    """Sorts by timestamp and adds 'bar_label' (formatted string) — the
    ordinal x-axis field every chart below uses. Reuse the SAME label
    column (and fmt) across every chart on one page so they stay aligned."""
    df = df.sort_values("timestamp").reset_index(drop=True)
    df = df.copy()
    df["bar_label"] = df["timestamp"].dt.strftime(fmt)
    return df


def _ordinal_x(df: pd.DataFrame, label_col: str, title=None) -> alt.X:
    return alt.X(
        f"{label_col}:O",
        sort=df[label_col].tolist(),
        title=title,
        axis=alt.Axis(labelAngle=-45, labelOverlap=True, grid=False),
    )


def candlestick_layers(df: pd.DataFrame, label_col: str = "bar_label", extra_tooltip=None):
    """Gap-free OHLC candlestick as a single combined layer (wicks + bodies).
    extra_tooltip: list of Altair tooltip field specs (e.g. ["volume:Q"])
    appended after the standard OHLC fields."""
    df = df.copy()
    df["direction"] = np.where(df["close"] >= df["open"], "Up", "Down")
    color = alt.Color(
        "direction:N",
        scale=alt.Scale(domain=["Up", "Down"], range=[UP_COLOR, DOWN_COLOR]),
        legend=None,
    )
    x = _ordinal_x(df, label_col, title=None)
    base = alt.Chart(df).encode(x=x)
    tooltip = ["timestamp:T", "open:Q", "high:Q", "low:Q", "close:Q"] + (extra_tooltip or [])
    wicks = base.mark_rule(strokeWidth=1.5).encode(
        alt.Y("low:Q", title="Price", scale=alt.Scale(zero=False)), alt.Y2("high:Q"), color=color,
    )
    bodies = base.mark_bar(size=6).encode(
        alt.Y("open:Q"), alt.Y2("close:Q"), color=color, tooltip=tooltip,
    )
    return wicks + bodies


def area_chart(df: pd.DataFrame, y_field: str, label_col: str = "bar_label",
                y_title=None, color: str = "#4a90d9", height: int = 140):
    """Gap-free filled area chart (position, exposure, etc.), x-axis aligned
    with candlestick_layers() when given the same label_col."""
    x = _ordinal_x(df, label_col)
    return (
        alt.Chart(df)
        .mark_area(opacity=0.5, color=color)
        .encode(x=x, y=alt.Y(f"{y_field}:Q", title=y_title or y_field))
        .properties(height=height)
    )


def line_chart(df: pd.DataFrame, y_field: str, label_col: str = "bar_label",
                y_title=None, color_field: str = None, height: int = 140):
    """Gap-free line chart, optionally colored/grouped by color_field (e.g.
    multiple currencies or seeds on one chart)."""
    x = _ordinal_x(df, label_col)
    enc = {
        "x": x,
        "y": alt.Y(f"{y_field}:Q", title=y_title or y_field, scale=alt.Scale(zero=False)),
    }
    if color_field:
        enc["color"] = alt.Color(f"{color_field}:N")
        enc["tooltip"] = [color_field, "timestamp:T", y_field]
    chart = alt.Chart(df).mark_line().encode(**enc).properties(height=height)
    return chart


def bar_chart(df: pd.DataFrame, y_field: str, label_col: str = "bar_label",
              y_title=None, height: int = 120):
    """Gap-free bar chart (volume, etc.)."""
    x = _ordinal_x(df, label_col)
    return (
        alt.Chart(df)
        .mark_bar()
        .encode(x=x, y=alt.Y(f"{y_field}:Q", title=y_title or y_field))
        .properties(height=height)
    )
