"""
This is a boilerplate pipeline 'filter'
generated using Kedro 0.18.7
"""
import pandas as pd
from vessel_manoeuvring_models.models.modular_simulator import ModularVesselSimulator
from vessel_manoeuvring_models.extended_kalman_vmm import ExtendedKalmanModular
import logging
import numpy as np
from phd.pipelines.load_7m.nodes import calculated_signals, divide_into_tests

log = logging.getLogger(__name__)


def guess_covariance_matrixes_many(ek_covariance_input: dict, datas: dict) -> dict:

    covariance_matrixes_many = {}
    for name, loader in datas.items():
        data = loader()
        covariance_matrixes = guess_covariance_matrixes(
            ek_covariance_input=ek_covariance_input, data=data
        )
        covariance_matrixes_many[name] = covariance_matrixes

    return covariance_matrixes_many


def guess_covariance_matrixes(ek_covariance_input: dict, data: pd.DataFrame) -> dict:

    process_variance = ek_covariance_input["process_variance"]
    variance_u = process_variance["u"]
    variance_v = process_variance["v"]
    variance_r = np.deg2rad(process_variance["r"])

    h = np.mean(np.diff(data.index))

    Qd = np.diag([variance_u, variance_v, variance_r]) * h  # process variances: u,v,r

    measurement_error_max = ek_covariance_input["measurement_error_max"]
    error_max_pos = measurement_error_max["positions"]
    sigma_pos = error_max_pos / 3
    variance_pos = sigma_pos**2

    error_max_psi = np.deg2rad(measurement_error_max["psi"])
    sigma_psi = error_max_psi / 3
    variance_psi = sigma_psi**2

    Rd = np.diag([variance_pos, variance_pos, variance_psi])

    P_prd = np.diag(
        [
            variance_pos,
            variance_pos,
            variance_psi,
            variance_u * h,
            variance_v * h,
            variance_r * h,
        ]
    )

    covariance_matrixes = {
        "P_prd": P_prd.tolist(),
        "Qd": Qd.tolist(),
        "Rd": Rd.tolist(),
    }

    return covariance_matrixes


def initial_state_many(
    datas: dict, state_columns=["x0", "y0", "psi", "u", "v", "r"]
) -> np.ndarray:

    x0_many = {}
    for name, loader in datas.items():
        data = loader()
        x0 = initial_state(data=data, state_columns=state_columns)
        x0_many[name] = x0

    return x0_many


def initial_state(
    data: pd.DataFrame, state_columns=["x0", "y0", "psi", "u", "v", "r"]
) -> np.ndarray:
    # x0 = data.iloc[0][state_columns].values
    x0 = data.iloc[0:5][state_columns].mean()

    return {key: float(value) for key, value in x0.items()}


def filter_many(
    datas: dict,
    models: dict,
    covariance_matrixes: dict,
    x0: dict,
    filter_model_name: str,
) -> dict:

    ek_many = {}
    time_steps_many = {}
    df_kalman_many = {}

    for name, loader in datas.items():
        data = loader()
        log.info(f"Filtering: {name}")
        ek_many[name], time_steps_many[name], df_kalman_many[name] = filter(
            data=data,
            models=models,
            covariance_matrixes=covariance_matrixes[name],
            x0=x0[name],
            filter_model_name=filter_model_name,
        )

    return ek_many, time_steps_many, df_kalman_many


def filter(
    data: pd.DataFrame,
    models: dict,
    covariance_matrixes: dict,
    x0: list,
    filter_model_name: str,
) -> pd.DataFrame:

    if not filter_model_name in models:
        raise ValueError(f"model: {filter_model_name} does not exist.")

    model = models[filter_model_name]()
    assert isinstance(model, ModularVesselSimulator)

    for name, subsystem in model.subsystems.items():
        if not hasattr(subsystem, "partial_derivatives"):
            log.info(
                f"The {name} system misses partial derivatives for the jacobi matrix. Creating new ones...(takes a while)"
            )
            subsystem.create_partial_derivatives()

    if not hasattr(subsystem, "lambda_jacobian"):
        f"The model misses the jacobi matrix. Creating a new one...(takes a while)"
        model.create_predictor_and_jacobian()

    Cd = np.array(
        [
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
        ]
    )

    E = np.array(
        [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ],
    )

    ek = ExtendedKalmanModular(model=model)
    x0_ = pd.Series(x0)[["x0", "y0", "psi", "u", "v", "r"]].values
    log.info("Running Kalman filter")
    time_steps = ek.filter(
        data=data,
        **covariance_matrixes,
        E=E,
        Cd=Cd,
        input_columns=model.control_keys,
        x0_=x0_,
    )

    df = ek.df_kalman.copy()
    calculated_signals(df)  # Update beta, V, true wind speed etc.

    return ek, time_steps, df


def smoother_many(
    ek: dict,
    datas: dict,
    time_steps: dict,
    covariance_matrixes: dict,
):
    ek_many = {}
    df_smooth_many = {}

    for name, loader in datas.items():
        data = loader()
        log.info(f"Smoothing: {name}")
        ek_many[name], df_smooth_many[name] = smoother(
            ek=ek[name],
            data=data,
            time_steps=time_steps[name],
            covariance_matrixes=covariance_matrixes[name],
        )

    return ek_many, df_smooth_many


def smoother(
    ek: ExtendedKalmanModular,
    data: pd.DataFrame,
    time_steps,
    covariance_matrixes: dict,
):

    ## Update parameters
    ek = ek.copy()

    E = np.array(
        [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ],
    )

    ek.Qd = np.array(covariance_matrixes["Qd"])
    ek.E = E

    ek.smoother(time_steps=time_steps)
    ek.data = data

    df = ek.df_smooth.copy()
    if "thrust" in data:
        df["thrust"] = data["thrust"].values

    calculated_signals(df)  # Update beta, V, true wind speed etc.

    return ek, df


def estimate_propeller_speed_many(
    datas: dict, models: dict, filter_model_name: str
) -> pd.DataFrame:
    df_all = {}
    for name, loader in datas.items():
        data = loader()
        df_all[name] = estimate_propeller_speed(
            data=data, models=models, filter_model_name=filter_model_name
        )

    return df_all


def estimate_propeller_speed(
    data: pd.DataFrame, models: dict, filter_model_name: str
) -> pd.DataFrame:

    if not filter_model_name in models:
        raise ValueError(f"model: {filter_model_name} does not exist.")

    model = models[filter_model_name]()
    assert isinstance(model, ModularVesselSimulator)

    # data["rev"] = 10  # Very simple estimation... ;)
    data["rev"] = (
        -0.1766 + 0.2074 * data["thrusterTarget"]
    )  # Simple propeller speed predictor
    # mask = data["rev"] < 0
    # data.loc[mask, "rev"] = float(data["rev"].mean())

    mask = data["rev"] <= 0
    data.loc[mask, "rev"] = 5.0
    mask = data["rev"] >= 15.5
    data.loc[mask, "rev"] = 15.5
    # mask = data["rev"] >= 21
    # data.loc[mask, "rev"] = 21

    return data


def divide_into_tests_filtered(data: pd.DataFrame, units: dict) -> pd.DataFrame:
    tests, time_series_meta_data = divide_into_tests(data=data, units=units)
    return tests
