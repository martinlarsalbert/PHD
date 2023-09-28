import pandas as pd
from vessel_manoeuvring_models.models.modular_simulator import ModularVesselSimulator
import matplotlib.pyplot as plt
from vessel_manoeuvring_models.substitute_dynamic_symbols import get_function_subs
import numpy as np


def predict(model: ModularVesselSimulator, data: pd.DataFrame) -> pd.DataFrame:
    df_force_predicted = pd.DataFrame(
        model.calculate_forces(data[model.states_str], control=data[model.control_keys])
    )
    return df_force_predicted


def plot_total_force(model: ModularVesselSimulator, data: pd.DataFrame, window=None):
    if not isinstance(model, dict):
        models = {"Prediction": model}
    else:
        models = model

    model = models[list(models.keys())[0]]
    df_force = model.forces_from_motions(data=data)

    if window is None:
        df_force_plot = df_force
    else:
        df_force_plot = (
            df_force.select_dtypes(exclude="object").rolling(window=window).mean()
        )

    fig, axes = plt.subplots(nrows=3)
    ax = axes[0]
    ax.set_title("X force")
    df_force_plot.plot(y="fx", label="Experiment", ax=ax)
    ax.set_xticklabels([])
    ax.set_xlabel("")

    ax = axes[1]
    ax.set_title("Y force")

    df_force_plot.plot(y="fy", label="Experiment", ax=ax)
    ax.set_xticklabels([])
    ax.set_xlabel("")

    ax = axes[2]
    ax.set_title("N Moment")

    df_force_plot.plot(y="mz", label="Experiment", ax=ax)

    for name, model in models.items():
        df_force_predicted = pd.DataFrame(
            model.calculate_forces(
                data[model.states_str], control=data[model.control_keys]
            )
        )
        df_force_predicted["fx"] = df_force_predicted["X_D"]
        df_force_predicted["fy"] = df_force_predicted["Y_D"]
        df_force_predicted["mz"] = df_force_predicted["N_D"]

        if window is None:
            df_force_predicted_plot = df_force_predicted
        else:
            df_force_predicted_plot = (
                df_force_predicted.select_dtypes(exclude="object")
                .rolling(window=window)
                .mean()
            )

        ax = axes[0]
        ax.set_title("X force")
        df_force_predicted_plot.plot(y="fx", label=name, ax=ax)
        ax.set_xticklabels([])
        ax.set_xlabel("")

        ax = axes[1]
        ax.set_title("Y force")
        df_force_predicted_plot.plot(y="fy", label=name, ax=ax)
        ax.set_xticklabels([])
        ax.set_xlabel("")

        ax = axes[2]
        ax.set_title("N Moment")
        df_force_predicted_plot.plot(y="mz", label=name, ax=ax)

    for ax in axes:
        make_y_axis_symmetrical(ax)

    return fig


def make_y_axis_symmetrical(ax):
    ylims = ax.get_ylim()
    ylim = np.max(np.abs(ylims))
    ax.set_ylim(-ylim, ylim)


def plot_force_components(model, data, window=None):
    df_force_predicted = pd.DataFrame(
        model.calculate_forces(data[model.states_str], control=data[model.control_keys])
    )

    if window is None:
        df_plot = df_force_predicted
    else:
        df_plot = df_force_predicted.rolling(window=window).mean()

    keys = list(get_function_subs(model.X_D_eq).values())
    fig, axes = plt.subplots(nrows=3)

    ax = axes[0]
    ax.set_title("X forces")
    for key in keys:
        df_plot.plot(y=key, ax=ax)

        ax.set_xlabel("time [s]")
    ax.set_xticklabels([])
    ax.set_xlabel("")

    keys = list(get_function_subs(model.Y_D_eq).values())
    ax = axes[1]
    ax.set_title("Y forces")

    for key in keys:
        df_plot.plot(y=key, ax=ax)
        ax.set_xlabel("time [s]")
    ax.set_xticklabels([])
    ax.set_xlabel("")

    keys = list(get_function_subs(model.N_D_eq).values())
    ax = axes[2]
    ax.set_title("N moments")

    for key in keys:
        df_plot.plot(y=key, ax=ax)
        ax.set_xlabel("time [s]")

    for ax in axes:
        make_y_axis_symmetrical(ax)

    return fig
