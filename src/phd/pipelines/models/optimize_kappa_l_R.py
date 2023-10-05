import pandas as pd
from vessel_manoeuvring_models.models.modular_simulator import ModularVesselSimulator
from vessel_manoeuvring_models.substitute_dynamic_symbols import run
from scipy.optimize import least_squares
import numpy as np


def calculate(
    model: ModularVesselSimulator, data: pd.DataFrame, kappa: float, l_R: float
):
    control = data[model.control_keys]
    states = data[["x0", "y0", "psi", "u", "v", "r"]]

    model.parameters["kappa"] = kappa
    model.parameters["l_R"] = l_R

    df_force_predicted = pd.DataFrame(
        model.calculate_forces(states_dict=states, control=control)
    )
    df_force_predicted["fx"] = run(model.lambda_X_D, inputs=df_force_predicted)
    df_force_predicted["fy"] = run(model.lambda_Y_D, inputs=df_force_predicted)
    df_force_predicted["mz"] = run(model.lambda_N_D, inputs=df_force_predicted)

    return df_force_predicted


def predict_mz_kappa_l_R(x, model: ModularVesselSimulator, data: pd.DataFrame):
    df_force_predicted = calculate(model=model, data=data, kappa=x[0], l_R=x[1])

    # 1)
    residual = data["mz"] - df_force_predicted["mz"]

    # 2)
    # mz_y = data["mz"] / model.ship_parameters["x_r"]
    # mz_y_pred = df_force_predicted["mz"] / model.ship_parameters["x_r"]
    # residual_mz = mz_y - mz_y_pred
    # residual_fy = data["fy"] - df_force_predicted["fy"]
    # residual = np.concatenate((residual_mz.values, residual_fy.values))

    # 3)
    # error_fx = data["fx"] - df_force_predicted["fx"]
    # error_mz = data["mz"] - df_force_predicted["mz"]
    # error_fy = data["fy"] - df_force_predicted["fy"]
    # residual = [
    #    # np.mean(np.abs(error_fx**2)),
    #    np.mean(np.abs(error_mz**2)),
    #    np.mean(np.abs(error_fy**2)),
    # ]

    return residual


def fit_kappa_l_R(model: ModularVesselSimulator, data: pd.DataFrame):
    if not "rev" in data:
        data["rev"] = data[["Prop/SB/Rpm", "Prop/PS/Rpm"]].mean(axis=1)

    if not "twa" in data:
        data["twa"] = 0

    if not "tws" in data:
        data["tws"] = 0

    df_force = model.forces_from_motions(data=data)

    kwargs = {"model": model, "data": df_force}

    if "kappa" in model.parameters:
        kappa_0 = model.parameters["kappa"]
    else:
        kappa_0 = 0.5

    x_r = model.ship_parameters["x_r"]
    if "l_R" in model.parameters:
        l_R_0 = model.parameters["l_R"]
    else:
        l_R_0 = x_r

    x0 = [kappa_0, l_R_0]

    return least_squares(
        fun=predict_mz_kappa_l_R,
        x0=x0,
        bounds=((0, 3 * x_r), (1, -3 * x_r)),
        kwargs=kwargs,
    )
