from vessel_manoeuvring_models.visualization.plot import plot_ship

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

force_scale = 15


def plot_arrow(x, y, magnitude, angle, ax, **kwargs):
    dx = magnitude * np.sin(angle)
    dy = magnitude * np.cos(angle)

    if not "width" in kwargs:
        kwargs["width"] = 0.02 * force_scale

    ax.arrow(
        x=x * force_scale,
        y=y * force_scale,
        dx=dx,
        dy=dy,
        length_includes_head=True,
        **kwargs,
    )


def plot_force(x, y, force, angle, ax, **kwargs):
    plot_arrow(x=x, y=y, magnitude=force, angle=angle, ax=ax, **kwargs)


def plot_force_xy(x, y, force_x, force_y, ax, **kwargs):
    force = np.sqrt(force_x**2 + force_y**2)
    angle = np.arctan2(force_y, force_x)
    plot_arrow(x=x, y=y, magnitude=force, angle=angle, ax=ax, **kwargs)


def quiver_line(phi, ax, l=1, N=5, **kwargs):
    da = l / (N - 1)
    a = np.arange(da, l, da)

    xs = a * np.sin(phi)
    ys = a * np.cos(phi)
    l_ = 0.7 * l / (N - 1)
    u_ = l_ * np.sin(phi)
    v_ = l_ * np.cos(phi)
    ax.quiver(xs, ys, u_, v_, scale=1, width=0.7, units="xy")

    s = np.linspace(0, 1.3 * l, 2)
    xs = s * np.sin(phi)
    ys = s * np.cos(phi)
    ax.plot(xs, ys, ":", **kwargs, zorder=-100)

    ax.axis("equal")


def overview(state: pd.Series, ship_data, ax=None, **kwargs):
    def plot_rudder(x_R, y_R, ax, suffix: str, **kwargs):
        # c = ship_data["c"]
        # l = 2 * np.array([0, -c])
        # tranform = np.array([np.sin(state["delta"]), np.cos(state["delta"])])
        # x = y_R + l * np.sin(state["delta"])
        # y = x_R + l * np.cos(state["delta"])
        # ax.plot(force_scale * x, force_scale * y, color="k", lw=3)

        plot_force_xy(
            x=y_R,
            y=x_R,
            force_x=state[f"X_R_{suffix}"],
            force_y=state[f"Y_R_{suffix}"],
            ax=ax,
            **kwargs,
        )

    def plot_velocity(ax):
        x = 0
        y = 0
        magnitude = force_scale * state["V"]
        angle = -state["beta"]

        # plot_arrow(
        #    x=x,
        #    y=y,
        #    magnitude=magnitude,
        #    angle=angle,
        #    ax=ax,
        #    color="m",
        #    label=f"V",
        # )

        l = ship_data["L"] / 2 * force_scale
        quiver_line(phi=angle, ax=ax, l=l, N=5, color="black", label="V")

        a = -np.linspace(0, 1.3 * l, 2)
        xs = a * np.sin(angle)
        ys = a * np.cos(angle)
        ax.plot(xs, ys, "k:", label="__none__", zorder=-100)

    def plot_wind(ax):
        X_W = state["X_W"]
        Y_W = state["Y_W"]
        N_W = state["N_W"]
        x_A = -0.2 * ship_data["L"]
        x_F = 0.2 * ship_data["L"]
        # factor = (N_W - x_F * Y_W) / (x_A - x_F)
        # Y_W_A = Y_W * factor
        # X_W_A = X_W * factor
        # Y_W_F = Y_W - Y_W_A
        # X_W_F = X_W - X_W_A

        X_W_A = 0
        Y_W_A = N_W / 2 / x_A

        X_W_F = 0
        Y_W_F = N_W / 2 / x_F

        plot_force_xy(
            x=0,
            y=0,
            force_x=X_W,
            force_y=Y_W,
            ax=ax,
            label="Wind",
            color="blue",
        )

        plot_force_xy(
            x=0,
            y=x_A,
            force_x=X_W_A,
            force_y=Y_W_A,
            ax=ax,
            label="__none__",
            color="blue",
        )

        plot_force_xy(
            x=0,
            y=x_F,
            force_x=X_W_F,
            force_y=Y_W_F,
            ax=ax,
            label="__none__",
            color="blue",
        )

    def plot_propeller(x_p, y_p, thrust, ax, **kwargs):
        x = y_p
        y = x_p
        magnitude = thrust
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

    if ax is None:
        fig, ax = plt.subplots()
    plot_ship(
        x=0,
        y=0,
        psi=0,
        ax=ax,
        lpp=force_scale * ship_data["L"],
        beam=force_scale * ship_data["B"],
        color="lightblue",
        alpha=1,
        zorder=-1000,
    )

    x_R = ship_data["x_r"]
    y_R = ship_data["y_p_port"]
    plot_rudder(
        x_R=x_R,
        y_R=y_R,
        suffix="port",
        ax=ax,
        label="Rudders",
        color="green",
    )
    y_R = ship_data["y_p_stbd"]
    plot_rudder(
        x_R=x_R,
        y_R=y_R,
        suffix="stbd",
        ax=ax,
        label="__none__",
        color="green",
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
    plot_wind(ax=ax)

    ax.legend()
    ax.set_xlabel(f"$Y$ $[N]$")
    ax.set_ylabel(f"$X$ $[N]$")

    ax.axis("equal")
    xlims = ax.get_xlim()
    ylims = ax.get_ylim()

    plot_velocity(ax=ax)

    l = 10 * force_scale
    ax.plot([-l, l], [0, 0], "k-", lw=0.3, zorder=-200)
    ax.plot([0, 0], [-l, l], "k-", lw=0.3, zorder=-200)

    ax.set_xlim(xlims)
    ax.set_ylim(ylims)

    return ax
