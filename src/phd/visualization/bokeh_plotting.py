import numpy as np

# Bokeh Libraries
from bokeh.models.widgets import Panel, Tabs
from bokeh.io import output_notebook
from bokeh.plotting import figure, show
from bokeh.models import (
    ColumnDataSource,
    CDSView,
    GroupFilter,
    ColorBar,
    HoverTool,
    BooleanFilter,
)
from bokeh.transform import linear_cmap, factor_mark

output_notebook()

from bokeh.io import curdoc

curdoc().theme = "dark_minimal"
from bokeh.palettes import Category10
from bokeh.layouts import row, column
from bokeh.core.properties import value
import itertools
from bokeh.palettes import Spectral6
from bokeh.plotting import output_file, save, reset_output


def color_gen():
    yield from itertools.cycle(Category10[10])


xs = {
    "resistance": "V",
    "Thrust variation": "thrust",
    "Thrust variation VCT": "thrust",
    "Thrust variation CT": "thrust",
    "Rudder angle resistance (no propeller)": "delta",
    "Rudder angle": "delta_deg",
    "Rudder angle VCT": "delta_deg",
    "Rudder angle CT": "delta_deg",
    "Drift angle": "beta_deg",
    "Circle": "r",
    "Circle + rudder angle": "delta",
    "Circle + Drift": "v*r",
    "Rudder and drift angle": "beta*delta",
    "Heel + Drift": "phi",
    "Heel and drift angle": "phi",
    "Heel": "phi",
    "Heel angle": "phi",
    "Force Test": "run",
    "Crash Stop": "rev",
}


def plot(
    x,
    y,
    test_type,
    mapper,
    groups,
    x_range,
    y_range,
    plot_width=800,
    plot_height=400,
    tool_tips_extras=[],
):

    tooltips = """
    <div>
        <span style="font-size: 10px;">
        @name<br>
        Ship:@model_name<br>
        V:@V<br>
        beta_deg:@beta_deg<br>
        r:@r<br>
        delta_deg:@delta_deg<br>
        %s:@%s<br>
        %s:@%s<br>
              
    """ % (
        x,
        x,
        y,
        y,
    )

    # Add some extra optional tool tips
    for tool_tip_key in tool_tips_extras:
        tooltips += "%s:@%s<br>" % (tool_tip_key, tool_tip_key)

    fig = figure(
        tooltips=tooltips,
        x_axis_label=x,
        y_axis_label=y,
        plot_width=plot_width,
        plot_height=plot_height,
        x_range=x_range,
        y_range=y_range,
    )

    # Axes:
    fig.line(x_range, [0, 0], line_width=2, color="blue")

    fig.line([0, 0], y_range, line_width=2, color="blue")

    markers = [
        "asterisk",
        "triangle",
        "x",
        "square",
        "circle",
        "cross",
        "circle_cross",
        "circle_x",
        "dash",
        "diamond",
        "diamond_cross",
        "hex",
        "inverted_triangle",
        "square_cross",
        "square_x",
    ]
    for group_name, group in groups:
        if len(markers) > 1:
            marker = markers.pop(0)
        else:
            marker = markers[0]

        vct_source = ColumnDataSource(group.sort_values(by=x))

        legend = ""
        for sub_group in group_name:
            legend += "%s " % sub_group

        fig.scatter(
            x=x,
            y=y,
            source=vct_source,
            # color = color.__next__(),
            line_color=mapper,
            color=None,
            size=12,
            line_width=2,
            legend_label=legend,
            marker=marker,
            alpha=1,
        )

        for V, df in group.groupby("V_round"):
            vct_source = ColumnDataSource(df.sort_values(by=x))
            fig.line(
                x,
                y,
                source=vct_source,
                line_alpha=0.5,
                legend_label=legend,
            )

    fig.legend.click_policy = "hide"

    return fig


def create_tab(
    df_VCT,
    by=["model_name"],
    ys=["fx", "fy", "mx", "mz"],
    plot_width=800,
    plot_height=400,
    do_save=False,
    save_name="plot.html",
    tool_tips_extras=[],
):

    df = df_VCT.copy()
    if not "name" in df:
        df["name"] = df.index

    df["v*r"] = df["v"] * df["r"]
    df["beta*delta"] = df["beta"] * df["delta"]

    df["beta_deg"] = np.rad2deg(df["beta"])
    df["delta_deg"] = np.rad2deg(df["delta"])
    df["V_round"] = df["V"].round(decimals=2)

    mapper = linear_cmap(
        field_name="V", palette=Spectral6, low=df["V"].min(), high=df["V"].max()
    )

    tabs = []

    for test_type, df_ in df.groupby("test type"):

        x = xs.get(test_type, "V")
        x_range = (df[x].min(), df[x].max())

        groups = df_.groupby(by=by)

        rows = []
        for y in ys:
            y_range = (df[y].min(), df[y].max())
            fig = plot(
                x=x,
                y=y,
                test_type=test_type,
                mapper=mapper,
                groups=groups,
                x_range=x_range,
                y_range=y_range,
                plot_width=plot_width,
                plot_height=plot_height,
                tool_tips_extras=tool_tips_extras,
            )
            rows.append(fig)

        tabs.append(Panel(child=column(rows), title=test_type))

    tabs = Tabs(tabs=tabs)

    if do_save:
        output_file(save_name, mode="inline", title="Evaluation")
        save(tabs)
        reset_output()
    else:
        show(tabs)

    return tabs
