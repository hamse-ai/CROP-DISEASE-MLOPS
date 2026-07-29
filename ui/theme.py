"""Chart theme: palette slots, Plotly defaults and small chart builders.

Colour is assigned by the *job* it does, not by rank or by taste:

* **categorical** -- identity (healthy vs diseased, before vs after). Hues are
  taken from a fixed slot order and never cycled or reassigned when a filter
  changes the series count.
* **sequential**  -- magnitude (a single blue ramp, light to dark).
* **status**      -- state (healthy / warning / critical), always paired with a
  label so colour never carries the meaning alone.

The palette below is a validated default: adjacent slots clear the
colour-vision-deficiency separation floor in both light and dark mode. Slots are
used in order, and charts here stay within the first three, which is the
all-pairs-safe prefix.
"""

from __future__ import annotations

from typing import Any

# --------------------------------------------------------------------------
# Palette
# --------------------------------------------------------------------------

CATEGORICAL_LIGHT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                     "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
CATEGORICAL_DARK = ["#3987e5", "#d95926", "#199e70", "#c98500",
                    "#d55181", "#008300", "#9085e9", "#e66767"]

# Single-hue ramp for magnitude. Never a rainbow.
SEQUENTIAL_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5",
                   "#256abf", "#184f95", "#0d366b"]

# Reserved for state. Never reused as a series colour.
STATUS = {"good": "#0ca30c", "warning": "#fab219",
          "serious": "#ec835a", "critical": "#d03b3b"}

CHROME = {
    "light": {
        "surface": "#fcfcfb", "plane": "#f9f9f7",
        "ink": "#0b0b0b", "ink_secondary": "#52514e", "muted": "#898781",
        "grid": "#e1e0d9", "axis": "#c3c2b7", "success_text": "#006300",
    },
    "dark": {
        "surface": "#1a1a19", "plane": "#0d0d0d",
        "ink": "#ffffff", "ink_secondary": "#c3c2b7", "muted": "#898781",
        "grid": "#2c2c2a", "axis": "#383835", "success_text": "#0ca30c",
    },
}

FONT_FAMILY = 'system-ui, -apple-system, "Segoe UI", sans-serif'


def current_mode() -> str:
    """Detect the viewer's theme so charts match the surface they sit on.

    Dark steps are a selected set validated against the dark surface, not an
    automatic inversion of the light ones.
    """
    try:
        import streamlit as st

        theme = getattr(st.context, "theme", None)
        if theme is not None and getattr(theme, "type", None) == "dark":
            return "dark"
        if st.get_option("theme.base") == "dark":
            return "dark"
    except Exception:
        pass
    return "light"


def palette(mode: str | None = None) -> list[str]:
    mode = mode or current_mode()
    return CATEGORICAL_DARK if mode == "dark" else CATEGORICAL_LIGHT


def chrome(mode: str | None = None) -> dict[str, str]:
    return CHROME[mode or current_mode()]


# --------------------------------------------------------------------------
# Plotly defaults
# --------------------------------------------------------------------------

def base_layout(mode: str | None = None, height: int = 320,
                show_legend: bool = False) -> dict[str, Any]:
    """Shared layout: recessive chrome, generous padding, hairline grid.

    Gridlines are solid hairlines one shade off the surface -- dashed grids read
    as a threshold or a projection when they are only a grid.
    """
    mode = mode or current_mode()
    c = chrome(mode)

    return {
        "height": height,
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "font": {"family": FONT_FAMILY, "size": 13, "color": c["ink_secondary"]},
        "margin": {"l": 8, "r": 16, "t": 32, "b": 8},
        "showlegend": show_legend,
        "legend": {
            "orientation": "h", "yanchor": "bottom", "y": 1.02,
            "xanchor": "left", "x": 0,
            "font": {"size": 12, "color": c["ink_secondary"]},
        },
        "hoverlabel": {
            "bgcolor": c["surface"], "bordercolor": c["axis"],
            "font": {"family": FONT_FAMILY, "size": 12, "color": c["ink"]},
        },
        "xaxis": {
            "gridcolor": c["grid"], "griddash": "solid", "zeroline": False,
            "linecolor": c["axis"], "tickfont": {"size": 11, "color": c["muted"]},
            "title": {"font": {"size": 12, "color": c["muted"]}},
        },
        "yaxis": {
            "gridcolor": c["grid"], "griddash": "solid", "zeroline": False,
            "linecolor": c["axis"], "tickfont": {"size": 11, "color": c["muted"]},
            "title": {"font": {"size": 12, "color": c["muted"]}},
        },
    }


def confidence_chart(top_k: list[dict[str, Any]], mode: str | None = None):
    """Horizontal bars for the top-k predictions.

    One series, so one hue -- with emphasis on the winning class and the
    runners-up in muted ink. Colouring every bar by its own rank would
    double-encode length as hue and burn the only free channel on information
    the bar length already carries.
    """
    import plotly.graph_objects as go

    mode = mode or current_mode()
    c, colours = chrome(mode), palette(mode)

    items = list(reversed(top_k))          # Plotly draws the first trace lowest
    labels = [f"{i['crop']} — {i['condition']}" for i in items]
    values = [i["probability"] * 100 for i in items]

    # Emphasis: the prediction in slot 1, the rest recessive.
    bar_colours = [colours[0] if i == len(items) - 1 else c["axis"]
                   for i in range(len(items))]

    figure = go.Figure(go.Bar(
        x=values, y=labels, orientation="h",
        marker={"color": bar_colours, "cornerradius": 4},
        text=[f"{v:.1f}%" for v in values],
        textposition="outside",
        textfont={"size": 12, "color": c["ink_secondary"]},
        hovertemplate="%{y}<br>%{x:.2f}%<extra></extra>",
        width=0.62,
    ))

    layout = base_layout(mode, height=60 + 46 * len(items))
    # Headroom so the outside labels are never clipped by the plot edge.
    layout["xaxis"].update({"range": [0, max(values) * 1.28], "showticklabels": False,
                            "showgrid": False})
    layout["yaxis"].update({"showgrid": False,
                            "tickfont": {"size": 12, "color": c["ink_secondary"]}})
    layout["margin"] = {"l": 8, "r": 60, "t": 8, "b": 8}
    figure.update_layout(**layout)
    return figure


def comparison_chart(labels: list[str], before: list[float], after: list[float],
                     mode: str | None = None):
    """Grouped bars comparing metrics before and after a retrain.

    Two series, so a legend is always present -- identity is never carried by
    colour alone.
    """
    import plotly.graph_objects as go

    mode = mode or current_mode()
    c, colours = chrome(mode), palette(mode)

    figure = go.Figure()
    for name, values, colour in (("Before", before, colours[0]),
                                 ("After", after, colours[1])):
        figure.add_trace(go.Bar(
            name=name, x=labels, y=values,
            marker={"color": colour, "cornerradius": 4},
            text=[f"{v:.3f}" if v is not None else "—" for v in values],
            textposition="outside",
            textfont={"size": 11, "color": c["ink_secondary"]},
            hovertemplate="%{x}<br>" + name + ": %{y:.4f}<extra></extra>",
        ))

    layout = base_layout(mode, height=340, show_legend=True)
    layout["yaxis"].update({"range": [0, 1.12], "title": {"text": "score"}})
    figure.update_layout(**layout, barmode="group", bargap=0.35, bargroupgap=0.08)
    return figure


def ranked_bar(labels: list[str], values: list[float],
               value_format: str = "{:.0f}",
               axis_title: str = "",
               mode: str | None = None,
               height: int | None = None):
    """Single-series horizontal bars, largest first."""
    import plotly.graph_objects as go

    mode = mode or current_mode()
    c, colours = chrome(mode), palette(mode)

    items = list(reversed(list(zip(labels, values))))
    names = [i[0] for i in items]
    amounts = [i[1] for i in items]

    figure = go.Figure(go.Bar(
        x=amounts, y=names, orientation="h",
        marker={"color": colours[0], "cornerradius": 4},
        text=[value_format.format(v) for v in amounts],
        textposition="outside",
        textfont={"size": 11, "color": c["ink_secondary"]},
        hovertemplate="%{y}<br>%{x}<extra></extra>",
        width=0.62,
    ))

    layout = base_layout(mode, height=height or (60 + 34 * len(items)))
    layout["xaxis"].update({"range": [0, max(amounts) * 1.25] if amounts else [0, 1],
                            "title": {"text": axis_title}})
    layout["yaxis"].update({"showgrid": False,
                            "tickfont": {"size": 11, "color": c["ink_secondary"]}})
    layout["margin"] = {"l": 8, "r": 56, "t": 16, "b": 32}
    figure.update_layout(**layout)
    return figure


def grouped_scatter(frame, x: str, y: str, colour_by: str,
                    x_title: str = "", y_title: str = "",
                    mode: str | None = None, height: int = 420):
    """Two-group scatter for the EDA feature separation plots.

    Capped at two groups on purpose: scatter compares every pair of colours at
    once, and only the first few palette slots clear the separation floor under
    that condition.
    """
    import plotly.graph_objects as go

    mode = mode or current_mode()
    c, colours = chrome(mode), palette(mode)

    figure = go.Figure()
    for i, (name, group) in enumerate(frame.groupby(colour_by, sort=True)):
        figure.add_trace(go.Scatter(
            x=group[x], y=group[y], mode="markers", name=str(name),
            marker={
                "size": 7, "color": colours[i % 3], "opacity": 0.55,
                # A 2px surface ring separates overlapping points -- never a
                # dark stroke, which would read as a border.
                "line": {"width": 1, "color": c["surface"]},
            },
            hovertemplate=f"{x}: %{{x:.3f}}<br>{y}: %{{y:.3f}}<extra>{name}</extra>",
        ))

    layout = base_layout(mode, height=height, show_legend=True)
    layout["xaxis"].update({"title": {"text": x_title or x}})
    layout["yaxis"].update({"title": {"text": y_title or y}})
    layout["margin"] = {"l": 8, "r": 16, "t": 40, "b": 48}
    figure.update_layout(**layout)
    return figure


def timeseries(x, y, name: str = "", y_title: str = "",
               mode: str | None = None, height: int = 260):
    """Single-series line with a crosshair hover."""
    import plotly.graph_objects as go

    mode = mode or current_mode()
    c, colours = chrome(mode), palette(mode)

    figure = go.Figure(go.Scatter(
        x=x, y=y, mode="lines", name=name,
        line={"width": 2, "color": colours[0], "shape": "spline", "smoothing": 0.4},
        fill="tozeroy", fillcolor=_translucent(colours[0], 0.10),
        hovertemplate="%{y:.2f}<extra></extra>",
    ))

    layout = base_layout(mode, height=height)
    layout["yaxis"].update({"title": {"text": y_title}})
    layout["xaxis"].update({"showspikes": True, "spikemode": "across",
                            "spikethickness": 1, "spikecolor": c["axis"],
                            "spikedash": "solid"})
    figure.update_layout(**layout, hovermode="x unified")
    return figure


def _translucent(hex_colour: str, alpha: float) -> str:
    hex_colour = hex_colour.lstrip("#")
    r, g, b = (int(hex_colour[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def status_colour(healthy: bool, confidence: float) -> tuple[str, str]:
    """Map a prediction to a status colour and an accompanying label.

    The label is returned alongside the colour, never instead of it: status
    colour must never carry meaning on its own.
    """
    if not healthy:
        return STATUS["critical"], "Disease detected"
    if confidence < 0.7:
        return STATUS["warning"], "Healthy — low confidence"
    return STATUS["good"], "Healthy"
