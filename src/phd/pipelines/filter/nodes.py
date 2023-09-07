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
from vessel_manoeuvring_models.data.lowpass_filter import lowpass_filter
from numpy import cos as cos
from numpy import sin as sin

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
    variance_r = process_variance["r"]

    h = np.mean(np.diff(data.index))

    Qd = np.diag([variance_u, variance_v, variance_r]) * h  # process variances: u,v,r

    measurement_error_max = ek_covariance_input["measurement_error_max"]
    error_max_pos = measurement_error_max["positions"]
    sigma_pos = error_max_pos / 3
    variance_pos = sigma_pos**2

    error_max_psi = np.deg2rad(measurement_error_max["psi"])
    sigma_psi = error_max_psi / 3
    variance_psi = sigma_psi**2

    diag = [variance_pos, variance_pos, variance_psi]

    if "r" in measurement_error_max:
        # Yaw rate is also a measurement!
        error_max_r = measurement_error_max["r"]
        sigma_r = error_max_r / 3
        variance_r = sigma_r**2
        diag.append(variance_r)

    Rd = np.diag(diag)

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
    accelerometer_position: dict,
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
            accelerometer_position=accelerometer_position,
        )

    return ek_many, time_steps_many, df_kalman_many


def filter(
    data: pd.DataFrame,
    models: dict,
    covariance_matrixes: dict,
    x0: list,
    filter_model_name: str,
    accelerometer_position: dict,
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
            [1, 0, 0, 0, 0, 0],  # x0 is measured
            [0, 1, 0, 0, 0, 0],  # y0 is measured
            [0, 0, 1, 0, 0, 0],  # psi is measured
            [0, 0, 0, 0, 0, 1],  # r is measured (GyroZ)
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
        measurement_columns=["x0", "y0", "psi", "GyroZ"],
        do_checks=False,
    )

    df = ek.df_kalman.copy()
    calculated_signals(
        df, accelerometer_position=accelerometer_position
    )  # Update beta, V, true wind speed etc.

    return ek, time_steps, df


def smoother_many(
    ek: dict,
    datas: dict,
    time_steps: dict,
    covariance_matrixes: dict,
    accelerometer_position: dict,
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
            accelerometer_position=accelerometer_position,
        )

    return ek_many, df_smooth_many


def smoother(
    ek: ExtendedKalmanModular,
    data: pd.DataFrame,
    time_steps,
    covariance_matrixes: dict,
    accelerometer_position: dict,
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

    calculated_signals(
        df, accelerometer_position=accelerometer_position
    )  # Update beta, V, true wind speed etc.

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
    return tests, time_series_meta_data


def derivative(df, key):
    d = np.diff(df[key]) / np.diff(df.index)
    d = np.concatenate((d, [d[-1]]))
    return d


def lowpass(df: pd.DataFrame, cutoff: float = 1.0, order=1) -> pd.DataFrame:
    """Lowpass filter and calculate velocities and accelerations with numeric differentiation

    Parameters
    ----------
    df : pd.DataFrame
        [description]
    cutoff : float
        Cut off frequency of the lowpass filter [Hz]
    order : int, optional
        order of the filter, by default 1

    Returns
    -------
    pd.DataFrame
        lowpassfiltered positions, velocities and accelerations
    """

    df_lowpass = df.copy()
    t = df_lowpass.index
    ts = np.mean(np.diff(t))
    fs = 1 / ts

    position_keys = ["x0", "y0", "psi"]
    for key in position_keys:
        df_lowpass[key] = lowpass_filter(
            data=df_lowpass[key], fs=fs, cutoff=cutoff, order=order
        )

    df_lowpass["x01d_gradient"] = x1d_ = derivative(df_lowpass, "x0")
    df_lowpass["y01d_gradient"] = y1d_ = derivative(df_lowpass, "y0")
    df_lowpass["r"] = r_ = derivative(df_lowpass, "psi")

    psi_ = df_lowpass["psi"]

    df_lowpass["u"] = x1d_ * cos(psi_) + y1d_ * sin(psi_)
    df_lowpass["v"] = -x1d_ * sin(psi_) + y1d_ * cos(psi_)

    velocity_keys = ["u", "v", "r"]
    for key in velocity_keys:
        df_lowpass[key] = lowpass_filter(
            data=df_lowpass[key], fs=fs, cutoff=cutoff, order=order
        )

    df_lowpass["u1d"] = r_ = derivative(df_lowpass, "u")
    df_lowpass["v1d"] = r_ = derivative(df_lowpass, "v")
    df_lowpass["r1d"] = r_ = derivative(df_lowpass, "r")

    return df_lowpass
