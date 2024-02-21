"""
This is a boilerplate pipeline 'resimulate_with_autopilot'
generated using Kedro 0.18.7
"""

import pandas as pd
import numpy as np
from vessel_manoeuvring_models.models.modular_simulator import ModularVesselSimulator
import vessel_manoeuvring_models.models.IMO_simulations
from vessel_manoeuvring_models.substitute_dynamic_symbols import run
from vessel_manoeuvring_models.angles import mean_angle
from scipy.optimize import least_squares
import logging

log = logging.getLogger(__name__)


def resimulate_all(
    datas: dict,
    model: ModularVesselSimulator,
    exclude=["wpcc.updated.joined.ek_smooth"],
) -> pd.DataFrame:
    results = {}
    log.info("Resimulating:")
    for key, loader in datas.items():
        if key in exclude:
            log.info(f"Exluded: {key}")
            continue
        else:
            log.info(f"Simulate: {key}")
        data = loader()
        results[key] = resimulate(data=data, model=model)

    return results


def resimulate(
    data: pd.DataFrame,
    model: ModularVesselSimulator,
    angle: float = None,
    neutral_rudder_angle: float = None,
    **kwargs,
) -> pd.DataFrame:
    data = data.copy()
    data["V"] = data["U"] = np.sqrt(data["u"] ** 2 + data["v"] ** 2)
    if not "rev" in data:
        data["rev"] = data[["Prop/SB/Rpm", "Prop/PS/Rpm"]].mean(axis=1)

    if not "twa" in data:
        data["twa"] = 0

    if not "tws" in data:
        data["tws"] = 0

    u0 = data.iloc[0]["u"]
    # rev=float(data['rev'].mean())

    result_optimization = find_initial_equilibrium_rev(data=data, model=model)
    if result_optimization.success:
        rev = result_optimization.x[0]
    else:
        log.error("Could not find rev (find_initial_equilibrium_rev)")
        rev = data["rev"]

    rudder_rate = 2.32 * np.sqrt(model.ship_parameters["scale_factor"])
    if angle is None:
        log.info("finding the angle")
        angle_abs = np.round(np.rad2deg(data["delta"].abs().max()), 0)
        i = (data["delta"].abs() > np.deg2rad(2)).idxmax()
        sign = np.sign(data["delta"].loc[i])
        angle = angle_abs if sign == 1 else -angle_abs

    psi0 = data.iloc[0]["psi"].copy()
    # twa = mean_angle(data["twa"]) - psi0
    twa = data["twa"] - psi0  # Note twa is rotated!

    # tws = data["tws"].mean() - psi0
    tws = data["tws"]
    if neutral_rudder_angle is None:
        neutral_rudder_angle = np.rad2deg(data["delta"].max() + data["delta"].min()) / 2
        log.info("Finding the neutral rudder angle")

    result = vessel_manoeuvring_models.models.IMO_simulations.zigzag(
        model=model,
        u0=u0,
        rev=rev,
        rudder_rate=rudder_rate,
        angle=angle,
        heading_deviation=angle,
        twa=twa,
        tws=tws,
        neutral_rudder_angle=neutral_rudder_angle,
        **kwargs,
    )
    return result

def turning_circle(
    data: pd.DataFrame,
    model: ModularVesselSimulator,
    angle: float = None,
    neutral_rudder_angle: float = None,
    r_limit = 0.1,
    **kwargs,
) -> pd.DataFrame:
    data = data.copy()
    data["V"] = data["U"] = np.sqrt(data["u"] ** 2 + data["v"] ** 2)
    if not "rev" in data:
        data["rev"] = data[["Prop/SB/Rpm", "Prop/PS/Rpm"]].mean(axis=1)

    if not "twa" in data:
        data["twa"] = 0

    if not "tws" in data:
        data["tws"] = 0

    u0 = data.iloc[0]["u"]
    # rev=float(data['rev'].mean())

    result_optimization = find_initial_equilibrium_rev(data=data, model=model)
    if result_optimization.success:
        rev = result_optimization.x[0]
    else:
        log.error("Could not find rev (find_initial_equilibrium_rev)")
        rev = data["rev"]

    rudder_rate = 2.32 * np.sqrt(model.ship_parameters["scale_factor"])
    if angle is None:
        log.info("finding the angle")
        angle_abs = np.round(np.rad2deg(data["delta"].abs().max()), 0)
        i = (data["delta"].abs() > np.deg2rad(2)).idxmax()
        sign = np.sign(data["delta"].loc[i])
        angle = angle_abs if sign == 1 else -angle_abs

    psi0 = data.iloc[0]["psi"].copy()
    # twa = mean_angle(data["twa"]) - psi0
    twa = data["twa"] - psi0  # Note twa is rotated!

    # tws = data["tws"].mean() - psi0
    tws = data["tws"]
    if neutral_rudder_angle is None:
        neutral_rudder_angle = np.rad2deg(data["delta"].max() + data["delta"].min()) / 2
        log.info("Finding the neutral rudder angle")

    result = vessel_manoeuvring_models.models.IMO_simulations.turning_circle(
        model=model,
        u0=u0,
        rev=rev,
        rudder_rate=rudder_rate,
        angle=angle,
        twa=twa,
        tws=tws,
        neutral_rudder_angle=neutral_rudder_angle,
        r_limit = r_limit,
        **kwargs,
    )
    return result

def calculate_fx(data: pd.DataFrame, rev: float, model: ModularVesselSimulator):
    control = data[model.control_keys].copy()
    states = data[["x0", "y0", "psi", "u", "v", "r"]]
    control["rev"] = rev
    df_force_predicted = pd.DataFrame(
        model.calculate_forces(states_dict=states, control=control)
    )
    df_force_predicted["fx"] = model.lambda_X_D(**df_force_predicted)
    return df_force_predicted


def predict_fx_rev(
    x, data: pd.DataFrame, df_force: pd.DataFrame, model: ModularVesselSimulator
):
    df_force_predicted = calculate_fx(data=data, rev=x[0], model=model)

    residual = df_force["fx"] - df_force_predicted["fx"]
    return residual


def find_initial_equilibrium_rev(
    data: pd.DataFrame, model: ModularVesselSimulator, i_stop=10
) -> float:
    data = data.iloc[0:i_stop]
    df_force = model.forces_from_motions(data=data)
    df_force["fx"] = 0

    kwargs = {"data": data, "df_force": df_force, "model": model}

    return least_squares(
        fun=predict_fx_rev, x0=[data.iloc[0]["rev"]], bounds=(0, np.inf), kwargs=kwargs
    )

def postprocess(df):
    df['V'] = df['U'] = np.sqrt(df['u']**2 + df['v']**2)
    df['beta'] = -np.arctan2(df['v'],df['u'])
    df['beta_deg'] = np.rad2deg(df['beta'])
    df['psi_deg'] = np.rad2deg(df['psi'])

def simulate_closed_loop(model:ModularVesselSimulator, data:pd.DataFrame):
    u0 = data.iloc[0]['u']
    rev = None
    rudder_rate = 2.32 * np.sqrt(model.ship_parameters["scale_factor"])
    angle_abs = np.rad2deg(data['delta'].abs().max())
    angle_sign = np.sign(data.iloc[0:10]['delta'].mean())
    angle = np.round(angle_sign*angle_abs)
    twa=None
    tws=None
    neutral_rudder_angle=0
    
    if 'thrust_port' in data:
        thrust_port = data['thrust_port']
        thrust_stbd = data['thrust_stbd']
        thrust = None
    else:
        thrust_port = None
        thrust_stbd = None
        thrust = data['thrust']
    
    result_closed_loop = vessel_manoeuvring_models.models.IMO_simulations.zigzag(
            model=model,
            u0=u0,
            rev=rev,
            thrust_port=thrust_port,
            thrust_stbd=thrust_stbd,
            thrust=thrust,
            rudder_rate=rudder_rate,
            angle=angle,
            heading_deviation=angle,
            twa=twa,
            tws=tws,
            neutral_rudder_angle=neutral_rudder_angle,
            t_max=data.index[-1],
        )
    postprocess(result_closed_loop)
    return result_closed_loop