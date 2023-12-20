import pandas as pd
from vessel_manoeuvring_models.models.modular_simulator import ModularVesselSimulator
from vessel_manoeuvring_models.substitute_dynamic_symbols import run
from scipy.optimize import least_squares
import numpy as np
from phd.visualization.plot_prediction import predict


def calculate(
    model: ModularVesselSimulator, data: pd.DataFrame, l_R: float, kappa: float
):
    control = data[model.control_keys]
    states = data[["x0", "y0", "psi", "u", "v", "r"]]

    model.parameters["l_R"] = l_R
    model.parameters["kappa_outer"] = kappa
    model.parameters["kappa_inner"] = kappa

    df_force_predicted = pd.DataFrame(
        model.calculate_forces(states_dict=states, control=control)
    )
    df_force_predicted = predict(model=model, data=data)

    return df_force_predicted


def predict_residual(x, model: ModularVesselSimulator, data: pd.DataFrame):
    df_force_predicted = calculate(model=model, data=data, l_R=x[0], kappa=x[1])

    scaler = x[2]
    scaler2 = x[3]

    # 1)
    residual1 = data["N_D"] - scaler * df_force_predicted["N_D"]
    residual2 = data["Y_D"] - scaler * df_force_predicted["Y_D"]
    residual = np.concatenate((residual1.values,residual2.values))
    
    return residual


def fit(model: ModularVesselSimulator, data: pd.DataFrame):
    model = model.copy()

    if not "rev" in data:
        data["rev"] = data[["Prop/SB/Rpm", "Prop/PS/Rpm"]].mean(axis=1)

    if not "twa" in data:
        data["twa"] = 0

    if not "tws" in data:
        data["tws"] = 0

    df_force = model.forces_from_motions(data=data)

    kwargs = {"model": model, "data": df_force}

    x_r = model.ship_parameters["x_r"]
    if "l_R" in model.parameters:
        l_R_0 = model.parameters["l_R"]
    else:
        l_R_0 = x_r

    if "kappa" in model.parameters:
        kappa_0 = model.parameters["kappa"]
    else:
        kappa_0 = 0.5

    x0 = [l_R_0, kappa_0, 1.0, 1.0]

    return least_squares(
        fun=predict_residual,
        x0=x0,
        # bounds=((3 * x_r, 0.1), (0.25 * x_r, 10)),
        kwargs=kwargs,
    )
