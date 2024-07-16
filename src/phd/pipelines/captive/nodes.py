"""
This is a boilerplate pipeline 'captive'
generated using Kedro 0.18.14
"""

from vessel_manoeuvring_models.models.modular_simulator import ModularVesselSimulator
import pandas as pd
import numpy as np
from vessel_manoeuvring_models.models.diff_eq_to_matrix import DiffEqToMatrix
from vessel_manoeuvring_models.symbols import *
from vessel_manoeuvring_models.prime_system import PrimeSystem
from vessel_manoeuvring_models.parameters import df_parameters
import logging

log = logging.getLogger(__name__)

p = df_parameters["symbol"]

scale_factor_TT = 25


def load(captive: pd.DataFrame, ship_data: dict) -> pd.DataFrame:
    log.info("Loading captive tests")

    captive.rename(
        columns={
            "eta0_round\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t": "eta0_round"
        },
        inplace=True,
    )
    captive["X_D"] = captive["fx"]
    captive["Y_D"] = captive["fy"]
    captive["N_D"] = captive["mz"]

    # Assuming hull = total:
    captive["X_H"] = captive["X_D"]
    captive["Y_H"] = captive["Y_D"]
    captive["N_H"] = captive["N_D"]

    captive["speed"] = np.round(
        captive["V"] * np.sqrt(scale_factor_TT) * 3.6 / 1.852, decimals=0
    )
    L_TT = ship_data["L"] * ship_data["scale_factor"] / scale_factor_TT
    g = 9.81
    captive["Fn"] = np.round(captive["V"] / np.sqrt(L_TT * g), decimals=2)

    mask = captive["test type"] == "Drift angle"
    captive.loc[mask, "test type"] = "Unknown"
    mask = (
        (captive["delta"] == 0)
        & (captive["beta"].abs() >= 0)
        & (captive["phi"] == 0)
        & (captive["V"] > 0.5)
        & (captive['r'] == 0)
        & pd.notnull((captive["V"]))
    )
    captive.loc[mask, "test type"] = "Drift angle"

    #captive.drop(index="40199079006k_06", inplace=True)

    return captive


def prime(df_CMT: pd.DataFrame, ship_data: dict, rho=1000) -> pd.DataFrame:
    log.info("Convert captive tests to prime system")

    L_TT = ship_data["L"] * ship_data["scale_factor"] / scale_factor_TT
    prime_system_TT = PrimeSystem(L=L_TT, rho=rho)
    units = {
        "Fn": "-",
    }

    mask = df_CMT["V"] > 0  # Does not work in prime system
    df_CMT = df_CMT.loc[mask].copy()

    df_CMT_prime = prime_system_TT.prime(
        df_CMT, U=df_CMT["V"], only_with_defined_units=True, units=units
    )
    df_CMT_prime["speed"] = df_CMT["speed"]

    return df_CMT_prime


def fit_and_correct(
    df_CMT_prime: pd.DataFrame,
    data_predict=None,
    alpha=0.05,
    symmetry=True,
):

    df_CMT_corrected = df_CMT_prime.copy()

    for (model_name, speed), group in df_CMT_prime.groupby(by=["model_name", "speed"]):

        log.info(f"_________________________________________________________________\n")

        df = group.groupby("test type").get_group("Drift angle")

        if len(group) < 2:
            log.info(f"Skipping: {model_name} at {speed}")
            continue

        log.info(
            f"Fitting a model to correct the drift angle captive tests for: {model_name} at {speed}"
        )

        df_CMT_corrected_ = fit_and_correct_one_model_one_speed(
            df_CMT_prime=df,
            data_predict=data_predict,
            alpha=alpha,
            symmetry=symmetry,
        )
        df_CMT_corrected.loc[df.index] = df_CMT_corrected_

    return df_CMT_corrected


def fit_and_correct_one_model_one_speed(
    df_CMT_prime: pd.DataFrame,
    data_predict=None,
    alpha=0.05,
    symmetry=True,
    full_output=False,
):

    dofs = ["Y_H", "N_H"]
    eqs = {}
    exclude_parameters = {}
    eq_to_matrices = {}
    ols_fits = {}

    equations = {
        "Y_H": sp.Eq(
            Y_H,
            p.Yv * v
            + p.Yr * r
            + p.Yvvv * v**3
            + p.Yvvr * v**2 * r
            + p.Yrrr * r**3
            + p.Yvrr * v * r**2
            + p.Y0,  # Very important!
        ),
        "N_H": sp.Eq(
            N_H,
            p.Nv * v
            + p.Nr * r
            + p.Nvvv * v**3
            + p.Nvvr * v**2 * r
            + p.Nrrr * r**3
            + p.Nvrr * v * r**2  # This one is very important to not get the drift...
            + p.N0,  # Very important !
        ),
    }

    df_predict = pd.DataFrame()

    if data_predict is None:
        data_predict = df_CMT_prime
        df_predict["beta"] = np.linspace(
            data_predict["beta"].min(), data_predict["beta"].max(), 30
        )
    else:
        df_predict = data_predict.copy()

    df_predict["V"] = 1
    df_predict["v"] = -df_predict["V"] * np.sin(df_predict["beta"])

    for dof in dofs:

        eq = equations[dof].subs(r, 0)  ## Towing tank
        log.info(f"Fitting{eq}")

        if len(np.round(df_CMT_prime["v"], decimals=2).unique()) < 3:
            eq = eq.subs(v**3, 0)

        eqs[dof] = eq

        label = eq.lhs
        eq_to_matrices[dof] = eq_to_matrix = DiffEqToMatrix(
            eq,
            label=label,
            base_features=[
                u,
                v,
                r,
                thrust,
                delta,
                thrust_port,
                thrust_stbd,
                y_p_port,
                y_p_stbd,
                Y_R,
                N_R,
            ],
            exclude_parameters=exclude_parameters,
        )

        ols_fits[dof] = ols_fit = eq_to_matrix.fit(
            data=df_CMT_prime, y=df_CMT_prime[dof]
        )

        log.info(ols_fit.summary2())

        X = eq_to_matrix.calculate_features(df_predict)

        if symmetry:
            if dof[0] == "Y":
                X["Y0"] = 0  # ship should be symmetric
            elif dof[0] == "N":
                X["N0"] = 0

        # Assuming hull = total:
        dof_save = f"{dof[0]}_D"
        df_predict[dof_save] = ols_fit.predict(X)

        prediction = ols_fit.get_prediction(X)

        confidence_intervalls = prediction.conf_int(alpha=alpha)
        df_predict[f"{dof_save}_iv_l"] = confidence_intervalls[:, 0]
        df_predict[f"{dof_save}_iv_u"] = confidence_intervalls[:, 1]

    df_CMT_corrected = df_CMT_prime.copy()
    df_CMT_corrected["Y_D"] = df_CMT_prime["Y_D"] - ols_fits["Y_H"].params["Y0"]
    df_CMT_corrected["N_D"] = df_CMT_prime["N_D"] - ols_fits["N_H"].params["N0"]

    df_predict["beta_deg"] = np.rad2deg(df_predict["beta"])

    if full_output:
        return df_CMT_corrected, df_predict
    else:    
        return df_CMT_corrected
