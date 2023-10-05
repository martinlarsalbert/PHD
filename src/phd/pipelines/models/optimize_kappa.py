import pandas as pd
from vessel_manoeuvring_models.models.modular_simulator import ModularVesselSimulator
from vessel_manoeuvring_models.substitute_dynamic_symbols import run
from scipy.optimize import least_squares


def calculate_mz(model: ModularVesselSimulator, data: pd.DataFrame, kappa: float):
    control = data[model.control_keys]
    states = data[["x0", "y0", "psi", "u", "v", "r"]]

    model.parameters["kappa"] = kappa

    df_force_predicted = pd.DataFrame(
        model.calculate_forces(states_dict=states, control=control)
    )
    df_force_predicted["mz"] = run(model.lambda_N_D, inputs=df_force_predicted)
    return df_force_predicted


def predict_mz_kappa(x, model: ModularVesselSimulator, data: pd.DataFrame):
    df_force_predicted = calculate_mz(model=model, data=data, kappa=x[0])
    residual = data["mz"] - df_force_predicted["mz"]
    return residual


def fit_kappa(model: ModularVesselSimulator, data: pd.DataFrame):
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

    x0 = [kappa_0]

    return least_squares(fun=predict_mz_kappa, x0=x0, bounds=(0, 1), kwargs=kwargs)
