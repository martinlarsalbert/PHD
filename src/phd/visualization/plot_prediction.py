import pandas as pd
from vessel_manoeuvring_models.models.modular_simulator import ModularVesselSimulator
import matplotlib.pyplot as plt
from vessel_manoeuvring_models.substitute_dynamic_symbols import get_function_subs
)


def plot_total_force(model: ModularVesselSimulator, data: pd.DataFrame, window=100):

    df_force_predicted = pd.DataFrame(
        model.calculate_forces(data[model.states_str], control=data[model.control_keys])
    )
    df_force_predicted["fx"] = df_force_predicted["X_D"]
    df_force_predicted["fy"] = df_force_predicted["Y_D"]
    df_force_predicted["mz"] = df_force_predicted["N_D"]

    df_force = model.forces_from_motions(data=data)

    fig, axes = plt.subplots(nrows=3)
    ax = axes[0]
    ax.set_title("X force")
    df_force.rolling(window=window).mean().plot(y="fx", label="Experiment", ax=ax)
    df_force_predicted.rolling(window=window).mean().plot(
        y="fx", label="Prediction", ax=ax
    )

    ax = axes[1]
    ax.set_title("Y force")

    df_force.rolling(window=window).mean().plot(y="fy", label="Experiment", ax=ax)
    df_force_predicted.rolling(window=window).mean().plot(
        y="fy", label="Prediction", ax=ax
    )

    ax = axes[2]
    ax.set_title("N Moment")

    df_force.rolling(window=window).mean().plot(y="mz", label="Experiment", ax=ax)
    df_force_predicted.rolling(window=window).mean().plot(
        y="mz", label="Prediction", ax=ax
    )

    return fig


def plot_force_components(model, data, window = 10):

    df_force_predicted = pd.DataFrame(
        model.calculate_forces(data[model.states_str], control=data[model.control_keys])
    )

    keys = list(get_function_subs(model.X_D_eq).values())
    fig, axes = plt.subplots(nrows=3)
    
    ax=axes[0]
    ax.set_title("X forces")
    for key in keys:
        
        df_force_predicted.rolling(window=window).mean().plot(y=key, ax=ax)
        ax.set_xlabel("time [s]")

    keys = list(get_function_subs(model.Y_D_eq).values())
    ax=axes[1]
    fig.set_title("Y forces")

    for key in keys:
        df_force_predicted.rolling(window=window).mean().plot(y=key, ax=ax)
        ax.set_xlabel("time [s]")

    keys = list(get_function_subs(model.N_D_eq).values())
    ax=axes[2]
    fig.set_title("N moments")

    for key in keys:
        df_force_predicted.rolling(window=window).mean().plot(y=key, ax=ax)
        ax.set_xlabel("time [s]")

    return fig