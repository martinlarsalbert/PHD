from vessel_manoeuvring_models.visualization.plot import plot_ship

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

force_scale = 30


def plot_arrow(x, y, magnitude, angle, ax, **kwargs):
    dx = magnitude * np.sin(angle)
    dy = magnitude * np.cos(angle)

    if not "width" in kwargs:
        kwargs["width"] = 0.02

    ax.arrow(x=x, y=y, dx=dx, dy=dy, **kwargs)


def plot_force(x, y, force, angle, ax, **kwargs):
    plot_arrow(x=x, y=y, magnitude=force / force_scale, angle=angle, ax=ax, **kwargs)


def plot_force_xy(x, y, force_x, force_y, ax, **kwargs):
    force = np.sqrt(force_x**2 + force_y**2)
    angle = np.arctan2(force_y, force_x)
    plot_arrow(x=x, y=y, magnitude=force / force_scale, angle=angle, ax=ax, **kwargs)


def overview(state: pd.Series, ship_data):
    def plot_rudder(x_R, y_R, ax):
        c = ship_data["c"]
        l = 2 * np.array([0, -c])
        tranform = np.array([np.sin(state["delta"]), np.cos(state["delta"])])
        x = y_R + l * np.sin(state["delta"])
        y = x_R + l * np.cos(state["delta"])
        ax.plot(x, y, color="k", lw=3)

    def plot_fan(x_fan, y_fan, force, angle, ax, **kwargs):
        x = y_fan
        y = x_fan
        magnitude = force / force_scale

        if not "color" in kwargs:
            kwargs["color"] = "g"

        plot_arrow(
            x=x,
            y=y,
            magnitude=magnitude,
            angle=angle,
            ax=ax,
            **kwargs,
        )

    def plot_velocity(ax):
        x = 0
        y = 0
        magnitude = state["V"]
        angle = -state["beta"]

        plot_arrow(
            x=x,
            y=y,
            magnitude=magnitude,
            angle=angle,
            ax=ax,
            color="m",
            label=f"V",
        )

    def plot_propeller(x_p, y_p, thrust, ax, **kwargs):
        x = y_p
        y = x_p
        magnitude = state["thrust"] / force_scale
        angle = 0

        if not "label" in kwargs:
            kwargs["label"] = "Propeller"

        plot_arrow(
            x=x,
            y=y,
            magnitude=magnitude,
            angle=angle,
            ax=ax,
            **kwargs,
        )

    fig, ax = plt.subplots()
    plot_ship(
        x=0,
        y=0,
        psi=0,
        ax=ax,
        lpp=ship_data["L"],
        beam=ship_data["B"],
        color="lightblue",
        alpha=1,
        zorder=-100,
    )

    x_R = ship_data["x_r"]
    y_R = ship_data["y_p_port"]
    plot_rudder(x_R=x_R, y_R=y_R, ax=ax)
    y_R = ship_data["y_p_stbd"]
    plot_rudder(x_R=x_R, y_R=y_R, ax=ax)

    plot_fan(
        x_fan=ship_data["x_fan_aft"],
        y_fan=ship_data["y_fan_aft"],
        force=state["Fan/Aft/Fx"],
        angle=state["Fan/Aft/Angle"],
        ax=ax,
        label="Fans",
    )
    plot_fan(
        x_fan=ship_data["x_fan_fore"],
        y_fan=ship_data["y_fan_fore"],
        force=state["Fan/Fore/Fx"],
        angle=state["Fan/Fore/Angle"],
        ax=ax,
        label="__none__",
    )

    plot_propeller(
        x_p=ship_data["x_p"],
        y_p=ship_data["y_p_port"],
        thrust=state["thrust_port"],
        color="r",
        label="Propellers",
        ax=ax,
        zorder=20,
    )
    plot_propeller(
        x_p=ship_data["x_p"],
        y_p=ship_data["y_p_stbd"],
        thrust=state["thrust_stbd"],
        color="r",
        label="__none__",
        ax=ax,
        zorder=20,
    )

    plot_velocity(ax=ax)

    ax.legend()
    ax.axis("equal")
    return ax
