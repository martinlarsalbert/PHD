import numpy as np
import matplotlib.pyplot as plt
import vessel_manoeuvring_models.visualization.plot as plot
from vessel_manoeuvring_models.angles import mean_angle


def track_plots(
    dataframes: dict,
    lpp: float,
    beam: float,
    ax=None,
    N: int = None,
    x_dataset="x0",
    y_dataset="y0",
    psi_dataset="psi",
    plot_boats=True,
    styles: dict = {},
    flip=False,
    time_window=[0, np.inf],
) -> plt.axes:

    ax = plot.track_plots(
        dataframes=dataframes,
        lpp=lpp,
        beam=beam,
        ax=ax,
        N=N,
        x_dataset=x_dataset,
        y_dataset=y_dataset,
        psi_dataset=psi_dataset,
        plot_boats=plot_boats,
        styles=styles,
        flip=flip,
        time_window=time_window,
    )

    for df in dataframes.values():

        if "twa" in df:
            twa = df["twa"]
            x0 = df["x0"]
            y0 = df["y0"]

            ## Plot wind:
            twa_mean = mean_angle(twa)
            R = np.sqrt((x0 - x0.iloc[0]) ** 2 + (y0 - y0.iloc[0]) ** 2).max()
            x = y0.mean() + 0.35 * R * np.cos(twa_mean)
            y = x0.mean() + 0.35 * R * np.sin(twa_mean)
            dx = -0.1 * R * np.cos(twa_mean)
            dy = -0.1 * R * np.sin(twa_mean)
            ax.arrow(
                x=x, y=y, dx=dx, dy=dy, width=10**-2 * R, color="m", label="Wind"
            )
            ax.legend(loc="best")

            break

    return ax
