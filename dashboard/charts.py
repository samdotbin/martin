"""
dashboard/charts.py — shared Altair chart builders. Pure presentation, no
data access (that's lib.py) — kept separate so the charting code isn't
duplicated across the pages that all show OHLC candlesticks.

Weekend gaps: a true temporal (:T) x-axis leaves dead whitespace over
Saturday/most of Sunday since no bars exist then, compressing the actual
trading hours into less room than they deserve. Every chart here instead
uses an ORDINAL axis over a formatted label column, in the data's own row
order — weekends (and any other gap) simply aren't categories, so they
don't take up space. Real timestamps still drive tooltips.

Styling: no top-level .configure_*() calls in any builder here — several
call sites layer more marks onto what these return (e.g. agent_replay.py
adds halt markers onto candlestick_layers()'s output), and Vega-Lite
rejects further composition once a chart carries a top-level config. All
theming here is mark- or encoding-level so results stay composable.
"""
import altair as alt
import numpy as np
import pandas as pd

UP_COLOR = "#00e5a0"
DOWN_COLOR = "#ff4d6d"
ACCENT = "#5b8dff"
GRID_COLOR = "#333a48"


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
        axis=alt.Axis(
            labelAngle=-45, labelOverlap=True, grid=False,
            domainColor=GRID_COLOR, tickColor=GRID_COLOR, labelColor="#9aa4b8",
        ),
    )


def _quant_y(field: str, title=None, zero: bool = True) -> alt.Y:
    return alt.Y(
        f"{field}:Q", title=title or field, scale=alt.Scale(zero=zero),
        axis=alt.Axis(gridColor=GRID_COLOR, domainColor=GRID_COLOR, tickColor=GRID_COLOR, labelColor="#9aa4b8"),
    )


def _fade(color: str, alpha_hex: str) -> str:
    """Appends an 8-digit-hex alpha channel to a #rrggbb color."""
    return f"{color}{alpha_hex}"


def candlestick_layers(df: pd.DataFrame, label_col: str = "bar_label", extra_tooltip=None):
    """Gap-free OHLC candlestick as a single combined layer (wicks + bodies)."""
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
    wicks = base.mark_rule(strokeWidth=1.6, opacity=0.85).encode(
        _quant_y("low", title="Price", zero=False), alt.Y2("high:Q"), color=color,
    )
    bodies = base.mark_bar(size=7, cornerRadius=2, stroke=None).encode(
        alt.Y("open:Q"), alt.Y2("close:Q"), color=color, tooltip=tooltip,
    )
    return wicks + bodies


def area_chart(df: pd.DataFrame, y_field: str, label_col: str = "bar_label",
                y_title=None, color: str = ACCENT, height: int = 160, zero: bool = True):
    """Gradient-filled area chart (position, exposure, etc.), x-axis aligned
    with candlestick_layers() when given the same label_col."""
    x = _ordinal_x(df, label_col)
    gradient = alt.Gradient(
        gradient="linear",
        stops=[
            alt.GradientStop(color=_fade(color, "b3"), offset=0),
            alt.GradientStop(color=_fade(color, "05"), offset=1),
        ],
        x1=1, y1=1, x2=1, y2=0,
    )
    return (
        alt.Chart(df)
        .mark_area(interpolate="monotone", line={"color": color, "strokeWidth": 2}, color=gradient)
        .encode(x=x, y=_quant_y(y_field, y_title, zero=zero))
        .properties(height=height)
    )


def line_chart(df: pd.DataFrame, y_field: str, label_col: str = "bar_label",
                y_title=None, color: str = ACCENT, color_field: str = None, height: int = 160):
    """Gap-free line chart. Single-series calls get a soft gradient glow
    under a crisp line with point markers — the 'trading terminal' look.
    Multi-series (color_field set) stays a plain multi-color line set, since
    layered glows from several overlapping series just muddy the chart."""
    x = _ordinal_x(df, label_col)
    y = _quant_y(y_field, y_title, zero=False)

    if color_field:
        enc = {
            "x": x, "y": y,
            "color": alt.Color(f"{color_field}:N", scale=alt.Scale(scheme="turbo")),
            "tooltip": [color_field, "timestamp:T", y_field],
        }
        return (
            alt.Chart(df).mark_line(strokeWidth=2.2, interpolate="monotone")
            .encode(**enc).properties(height=height)
        )

    base = alt.Chart(df).encode(x=x, y=y)
    gradient = alt.Gradient(
        gradient="linear",
        stops=[
            alt.GradientStop(color=_fade(color, "66"), offset=0),
            alt.GradientStop(color=_fade(color, "00"), offset=1),
        ],
        x1=1, y1=1, x2=1, y2=0,
    )
    glow = base.mark_area(interpolate="monotone", color=gradient, line=False)
    crisp = base.mark_line(
        interpolate="monotone", strokeWidth=2.6, color=color,
        point=alt.OverlayMarkDef(filled=True, size=26, color=color),
    )
    return (glow + crisp).properties(height=height)


def bar_chart(df: pd.DataFrame, y_field: str, label_col: str = "bar_label",
              y_title=None, color: str = ACCENT, height: int = 120):
    """Gap-free bar chart (volume, etc.)."""
    x = _ordinal_x(df, label_col)
    return (
        alt.Chart(df)
        .mark_bar(cornerRadiusTopLeft=2, cornerRadiusTopRight=2, color=color, opacity=0.85)
        .encode(x=x, y=_quant_y(y_field, y_title))
        .properties(height=height)
    )


def position_heatmap(pairs: list, values, height: int = 90):
    """One row of colored cells, one per pair — red/green intensity by
    position direction and size (bucket index, e.g. -2..+2). Used by the
    Arena 'Beat the AI' challenge to show all 28 pairs' current stance at a
    glance, the same way a trading terminal's book view would."""
    df = pd.DataFrame({"Pair": pairs, "Position": values})
    return (
        alt.Chart(df)
        .mark_rect(cornerRadius=3, stroke="#11151c", strokeWidth=2)
        .encode(
            x=alt.X("Pair:N", sort=pairs, title=None,
                    axis=alt.Axis(labelAngle=-60, domainColor=GRID_COLOR, tickColor=GRID_COLOR, labelColor="#9aa4b8")),
            color=alt.Color(
                "Position:Q",
                scale=alt.Scale(domain=[-2, 0, 2], range=[DOWN_COLOR, "#1c2330", UP_COLOR]),
                legend=None,
            ),
            tooltip=["Pair:N", "Position:Q"],
        )
        .properties(height=height)
    )
