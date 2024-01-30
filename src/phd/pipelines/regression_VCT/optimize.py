import pandas as pd
from vessel_manoeuvring_models.models.modular_simulator import ModularVesselSimulator
from vessel_manoeuvring_models.substitute_dynamic_symbols import run
from scipy.optimize import least_squares
import numpy as np
from phd.visualization.plot_prediction import predict


def update_model(model, x, parameters):
    changes = {}

    for i, parameter in enumerate(parameters):
        if isinstance(parameter, str):
            model.parameters[parameter] = changes[parameter] = x[i]
        elif isinstance(parameter, list) or isinstance(parameter, tuple):
            for sub_parameter in parameter:
                model.parameters[sub_parameter] = changes[sub_parameter] = x[i]

    return changes


def calculate(model: ModularVesselSimulator, data: pd.DataFrame, x, parameters: list):
    control = data[model.control_keys]
    states = data[["x0", "y0", "psi", "u", "v", "r"]]

    update_model(model=model, x=x, parameters=parameters)

    ## Calculate only rudder forces:
    calculation = {}
    calculation = model.subsystems["rudder_port"].calculate_forces(
        states_dict=states, control=control, calculation=calculation
    )
    calculation = model.subsystems["rudder_stbd"].calculate_forces(
        states_dict=states, control=control, calculation=calculation
    )
    calculation = model.subsystems["rudders"].calculate_forces(
        states_dict=states, control=control, calculation=calculation
    )
    df_force_predicted = pd.DataFrame(calculation)
    # df_force_predicted = pd.DataFrame(
    #    model.calculate_forces(states_dict=states, control=control)
    # )
    # df_force_predicted = predict(model=model, data=data)

    return df_force_predicted


def predict_residual(
    x,
    model: ModularVesselSimulator,
    data: pd.DataFrame,
    parameters: list,
    residual_keys=["N_R", "Y_R"],
):
    df_force_predicted = calculate(model=model, data=data, x=x, parameters=parameters)

    # 1)

    # residual1 = data["N_R"] - df_force_predicted["N_R"]
    # residual2 = data["Y_R"] - df_force_predicted["Y_R"]

    residuals = [data[key] - df_force_predicted[key] for key in residual_keys]

    residual = np.concatenate(residuals)

    return residual


def fit(
    model: ModularVesselSimulator,
    data: pd.DataFrame,
    parameters: list,
    residual_keys=["N_R", "Y_R"],
):
    model = model.copy()

    if not "rev" in data:
        data["rev"] = data[["Prop/SB/Rpm", "Prop/PS/Rpm"]].mean(axis=1)

    if not "twa" in data:
        data["twa"] = 0

    if not "tws" in data:
        data["tws"] = 0

    # df_force = model.forces_from_motions(data=data)

    kwargs = {
        "model": model,
        "data": data,
        "parameters": parameters,
        "residual_keys": residual_keys,
    }

    x0 = [
        model.parameters[parameter]
        if isinstance(parameter, str)
        else np.mean([model.parameters[sub_parameter] for sub_parameter in parameter])
        for parameter in parameters
    ]

    result = least_squares(
        fun=predict_residual,
        x0=x0,
        # bounds=((3 * x_r, 0.1), (0.25 * x_r, 10)),
        kwargs=kwargs,
    )
    assert result.success

    changes = update_model(model=model, x=result.x, parameters=parameters)

    return model, changes
