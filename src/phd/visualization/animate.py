import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.gridspec import GridSpec
from . import plot_ship
import imageio

from ipywidgets import Layout, interact, IntSlider, interactive, Play
import ipywidgets as widgets
import os


def plot_thruster(ax, x, y, delta, power, kW_per_meter=1.0):

    l = power * kW_per_meter
    ax.arrow(
        x=y,
        y=x,
        dx=l * np.sin(delta),
        dy=l * np.cos(delta),
        head_width=l / 5,
        head_length=l / 5,
    )
    # ax.arrow(x=y, y=x, dx=l*sin, dy=-l*cos, head_width=l/5, head_length=l/5)


def plot_thrusters(
    ax,
    row,
    lpp=50,
    beam=20,
    extra_space=30,
    positions_xy=[[-1, 0], [1, 0]],
    kW_per_meter=1.0,
):

    plot_ship.plot(0, 0, 0, lpp=lpp, beam=beam, ax=ax, color="b", alpha=0.5)

    positions_xy = np.array(positions_xy)

    coordinates = np.array(
        [positions_xy[:, 0] * lpp / 2, positions_xy[:, 1] * beam / 2]
    ).T

    for i, position in enumerate(coordinates):

        n = i + 1
        delta_key = f"delta{n}"
        power_key = f"ME{n} Load [kW]"

        plot_thruster(
            ax=ax,
            x=position[0],
            y=position[1],
            delta=np.deg2rad(row[delta_key]),
            power=row[power_key],
            kW_per_meter=kW_per_meter,
        )
        ax.text(position[1], position[0], " Thruster %i" % n)

    # velocity
    l = 3 * row["sog"]
    # if row["reversing"]:
    #    direction = np.deg2rad(row["cog"] - row["heading"] - 180)
    # else:
    #    direction = np.deg2rad(row["cog"] - row["heading"])

    direction = np.deg2rad(row["cog"] - row["heading"])

    dx = l * np.cos(direction)
    dy = l * np.sin(direction)
    ax.arrow(x=0, y=0, dx=dy, dy=dx, head_width=l / 5, head_length=l / 5, color="green")
    ax.annotate(xy=(0, 0), xytext=(1.3 * dy, 1.3 * dx), text="velocity")

    ax.set_xlim(
        np.min(coordinates[:, 1]) - extra_space, np.max(coordinates[:, 1]) + extra_space
    )
    ax.set_ylim(
        np.min(coordinates[:, 0]) - extra_space, np.max(coordinates[:, 0]) + extra_space
    )

    # if row["reversing"]:
    #    # Rotate the view
    #    ax.invert_yaxis()
    #    ax.invert_xaxis()


def plot_thrust_allocation(
    row, trip, lpp=50, beam=20, extra_space=30, positions_xy=[[-1, 0], [1, 0]]
):

    ## First plot

    # fig,axes=plt.subplots(ncols=2)
    fig = plt.figure(constrained_layout=True)
    fig.set_size_inches(19, 8)

    gs = GridSpec(2, 2, figure=fig)
    ax1 = fig.add_subplot(gs[:, 0])
    # identical to ax1 = plt.subplot(gs.new_subplotspec((0, 0), colspan=3))
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1:, 1])

    ax = ax1

    max_kW = np.max(
        [trip[f"ME{i} Load [kW]"].max() for i in range(1, len(positions_xy) + 1)]
    )
    kW_per_meter = max_kW / extra_space / 50

    plot_thrusters(
        ax=ax,
        row=row,
        lpp=lpp,
        beam=beam,
        extra_space=extra_space,
        positions_xy=positions_xy,
        kW_per_meter=kW_per_meter,
    )

    ## Second plot:
    ax = ax2
    trip.plot(x="longitude", y="latitude", ax=ax, style="k--")
    # ax.plot(row['longitude'], row['latitude'], 'o', ms=15)
    x = row["latitude"]
    y = row["longitude"]
    psi = np.deg2rad(row["heading"])

    # Define a ship size in lat/lon:
    N_scale = 20
    extra_space = (
        1
        / lpp
        / N_scale
        * np.sqrt(
            (trip["latitude"].max() - trip["latitude"].min()) ** 2
            + (trip["longitude"].max() - trip["longitude"].min()) ** 2
        )
    )
    lpp_ = lpp * extra_space
    beam_ = beam * extra_space

    plot_ship.plot(x, y, psi, lpp=lpp_, beam=beam_, ax=ax, color="b", alpha=0.5)

    ax.set_ylim(trip["latitude"].min() - 0.005, trip["latitude"].max() + 0.005)
    ax.set_xlim(trip["longitude"].min() - 0.005, trip["longitude"].max() + 0.005)

    ax.set_aspect("equal")

    ## Third plot:
    trip.plot(x="trip_time", y="sog", ax=ax3, style="k--")
    ax3.plot(row["trip_time"], row["sog"], "o", ms=15)
    ax3.set_xlabel("Time [s]")
    ax3.set_ylabel("Ship speed [m/s]")


def create_animator(trip, positions_xy=[[-1, 0], [1, 0]]):
    trip = trip.copy()

    def animate(i=0):

        index = int(i)
        row = trip.iloc[index]
        plot_thrust_allocation(row=row, trip=trip, positions_xy=positions_xy)

    return animate


def normalize_power(trip, positions_xy=[[-1, 0], [1, 0]]):
    trip = trip.copy()
    power_columns = [f"ME{i} Load [kW]" for i in range(1, len(positions_xy) + 1)]
    trip[power_columns] /= trip["ME load [kW]"].max() / 4
    return trip


def widget(
    trip: pd.DataFrame, positions_xy=[[-1, 0], [1, 0]], downsample="1S"
) -> widgets.VBox:
    """ipywidget widget stepping in the animation

    Parameters
    ----------
    trip : pd.DataFrame
        [description]

    Returns
    -------
    widget
        [description]
    """

    trip = trip.copy()

    ## Preprocess:
    trip["trip_time_s"] = pd.TimedeltaIndex(trip["trip_time"]).total_seconds()

    ## Normalizing:
    # trip = normalize_power(trip=trip, positions_xy=positions_xy)

    ## Resample:
    if downsample is not None:
        trip = trip.resample(downsample).mean().dropna()

    animator = create_animator(trip=trip, positions_xy=positions_xy)

    play = Play(
        value=0,
        min=0,
        max=len(trip) - 1,
        step=1,
        interval=100,
        description="Press play",
        disabled=False,
    )

    slider = IntSlider(0, 0, len(trip) - 1, 1, layout=Layout(width="70%"))
    widgets.jslink((play, "value"), (slider, "value"))
    animation = interactive(animator, i=slider)
    return widgets.VBox([play, slider, animation])


def create_gif(trip: pd.DataFrame, animation_dir=None):
    """Create a GIF animation for a trip, showing the thrusters etc.

    Parameters
    ----------
    trip : pd.DataFrame
        [description]
    animation_dir : [type], optional
        [description], by default None
    """

    ## Where should the animation be placed?
    if animation_dir is None:
        animation_dir = "animations"
        if not os.path.exists(animation_dir):
            os.mkdir(animation_dir)

    file_paths = []
    trip_anim = trip.resample("5S").mean()

    ## Animate:
    for index, row in trip_anim.iterrows():
        plot_thrust_allocation(row=row)
        file_name = "%i.png" % row["trip_time_s"]
        file_path = os.path.join(animation_dir, file_name)
        file_paths.append(file_path)
        plt.savefig(file_path)
        plt.close()

    ## Convert to GIF:
    trip_no = trip.iloc[0]["trip_no"]
    save_file_path = os.path.join(animation_dir, "trip_%i.gif" % trip_no)
    with imageio.get_writer(save_file_path, mode="I") as writer:
        for file_path in file_paths:
            image = imageio.imread(file_path)
            writer.append_data(image)

    ## Clean up:
    for file_path in file_paths:
        os.remove(file_path)
