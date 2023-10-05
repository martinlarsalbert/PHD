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
    include_wind=True,
) -> plt.axes:
    styles_ = styles.copy()

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
        styles=styles_,
        flip=flip,
        time_window=time_window,
    )

    if not include_wind:
        return ax

    for df in dataframes.values():
        if "twa" in df:
            twa = df["twa"]
            tws = df["tws"]
            x0 = df["x0"]
            y0 = df["y0"]

            ## Plot wind:

            # True wind:
            twa_mean = mean_angle(twa)

            # x = -50
            # y = 0

            x = y0.mean() - 10
            y = x0.mean()

            mean_wind = tws.mean()
            l = 2 * lpp * mean_wind / 10
            dx = -l * lpp * np.sin(twa_mean)
            dy = -l * lpp * np.cos(twa_mean)

            ax.arrow(
                x=x,
                y=y,
                dx=dx,
                dy=dy,
                width=1.0 * beam,
                color="m",
                label=f"True wind {np.round(mean_wind,1)} m/s",
                zorder=10,
            )

        if "awa" in df:
            # Apparent wind:

            awa = (
                df["awa"] + df["psi"]
            )  # Apparent wind angle in Earth fixed coordinates
            awa_mean = mean_angle(awa)

            # x = 50
            # y = 0
            x0 = df["x0"]
            y0 = df["y0"]
            x = y0.median() + 10
            y = x0.mean()

            mean_wind = df["aws"].mean()
            l = 2 * lpp * mean_wind / 10

            dx = -l * lpp * np.sin(awa_mean)
            dy = -l * lpp * np.cos(awa_mean)

            ax.arrow(
                x=x,
                y=y,
                dx=dx,
                dy=dy,
                width=1.0 * beam,
                color="c",
                label=f"Apparent {np.round(mean_wind,1)} m/s",
                zorder=10,
            )

            ax.legend(loc="lower right")
            break

    return ax
