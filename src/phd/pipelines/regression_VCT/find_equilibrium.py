import pandas as pd
from vessel_manoeuvring_models.models.modular_simulator import ModularVesselSimulator
from vessel_manoeuvring_models.substitute_dynamic_symbols import run
from scipy.optimize import least_squares
import numpy as np
from phd.visualization.plot_prediction import predict


def unpack_x(
    model: ModularVesselSimulator,
    data: pd.DataFrame,
    x,
    states_optimize: list,
    controls_optimize: list,
):
    i = 0
    states = pd.DataFrame(index=data.index)
    for key in model.states_str:
        if key in states_optimize:
            states[key] = x[i]
            i += 1
        else:
            states[key] = data[key]

    controls = pd.DataFrame(index=data.index)
    for key in model.control_keys:
        if key in controls_optimize:
            controls[key] = x[i]
            i += 1
        else:
            controls[key] = data[key]

    return states, controls


def calculate(
    model: ModularVesselSimulator,
    data: pd.DataFrame,
    x,
    states_optimize: list,
    controls_optimize: list,
):
    states, controls = unpack_x(
        model=model,
        data=data,
        x=x,
        states_optimize=states_optimize,
        controls_optimize=controls_optimize,
    )

    df_force_predicted = pd.DataFrame(
        model.calculate_forces(states_dict=states, control=controls)
    )
    # df_force_predicted = predict(model=model, data=data)

    return df_force_predicted


def predict_residual(
    x,
    model: ModularVesselSimulator,
    data: pd.DataFrame,
    states_optimize: list,
    controls_optimize: list,
    dofs=["X_D", "Y_D", "N_D"],
):
    df_force_predicted = calculate(
        model=model,
        data=data,
        x=x,
        states_optimize=states_optimize,
        controls_optimize=controls_optimize,
    )

    # 1)
    # residual1 = data["N_R"] - df_force_predicted["N_R"]
    # residual2 = data["Y_R"] - df_force_predicted["Y_R"]
    # residual = np.concatenate((residual1.values,residual2.values))

    residual = (df_force_predicted[dofs]).iloc[0]

    return residual


def fit(
    model: ModularVesselSimulator,
    data: pd.DataFrame,
    states_optimize=[],
    controls_optimize=[],
    dofs=["X_D", "Y_D", "N_D"],
):
    """Fit states and controls to find the force equilibrium

    Parameters
    ----------
    model : ModularVesselSimulator
        _description_
    data : pd.DataFrame
        _description_
    states : list, optional
        states that should be optimized, by default []
    controls : list, optional
        controls that should be optimized, by default []

    Returns
    -------
    _type_
        _description_
    """

    # model = model.copy()

    ## x0
    # states_optimize = list(set(model.states_str) - set(states))
    x0_states = [data[state].iloc[0] for state in states_optimize]

    # controls_optimize = list(set(model.control_keys) - set(controls))
    x0_controls = [data[control_key].iloc[0] for control_key in controls_optimize]

    x0 = x0_states + x0_controls

    ## kwargs:
    kwargs = {
        "model": model,
        "data": data,
        "states_optimize": states_optimize,
        "controls_optimize": controls_optimize,
        "dofs": dofs,
    }

    result = least_squares(
        fun=predict_residual,
        x0=x0,
        # bounds=((3 * x_r, 0.1), (0.25 * x_r, 10)),
        kwargs=kwargs,
        method="lm",
        ftol=1e-10,
        xtol=1e-10,
        gtol=1e-10,
    )
    assert result.success

    states, controls = unpack_x(
        model=model,
        data=data,
        x=result.x,
        states_optimize=states_optimize,
        controls_optimize=controls_optimize,
    )

    return states.iloc[0], controls.iloc[0]


def equilibrium(
    model: ModularVesselSimulator,
    data: pd.DataFrame,
    states_optimize=[],
    controls_optimize=[],
    dofs=["X_D", "Y_D", "N_D"],
) -> dict:
    states, controls = fit(
        model=model,
        data=data,
        states_optimize=states_optimize,
        controls_optimize=controls_optimize,
        dofs=dofs,
    )

    result = model.calculate_forces(states_dict=states, control=controls)

    state = dict(data.iloc[0])
    state.update(dict(states))
    state.update(dict(controls))
    state.update(result)
    state["V"] = np.sqrt(state["u"] ** 2 + state["v"] ** 2)
    state["beta"] = np.arctan2(-state["v"], state["u"])
    state["psi"] = state["beta"]

    return state
