"""
This is a boilerplate pipeline 'regression_VCT_7m'
generated using Kedro 0.18.14
"""
import pandas as pd
from vessel_manoeuvring_models.models.modular_simulator import ModularVesselSimulator
from vessel_manoeuvring_models.prime_system import PrimeSystem
import numpy as np
from vessel_manoeuvring_models.symbols import *
from vessel_manoeuvring_models.models.diff_eq_to_matrix import DiffEqToMatrix
import statsmodels.api as sm
from vessel_manoeuvring_models.substitute_dynamic_symbols import (
    run,
    lambdify,
    remove_functions,
    equation_to_python_method,
)
from phd.visualization.plot_prediction import predict

# from vct.read_shipflow import mirror_x_z
from phd.pipelines.models import optimize_l_R_shape
from sklearn.metrics import r2_score
from vessel_manoeuvring_models.parameters import df_parameters
import phd.pipelines.regression_VCT.optimize

p = df_parameters["symbol"]

from pyfiglet import figlet_format


import logging

log = logging.getLogger(__name__)

import pandas as pd

def load_VCT(df_VCT_all_raw: dict) -> dict:
    """_summary_

    Parameters
    ----------
    df_VCT : dict
        partitioned dataset with VCT results from different sources

    Returns
    -------
    pd.DataFrame
        concatinated selection of VCT data for this ship
    """
    files = {}
    for key, loader in df_VCT_all_raw.items():
        #if not isinstance(df, pd.DataFrame):
        
        df = loader()
        extra_columns(df)
        files[key] = df

    return files

def extra_columns(df):
    df["X_D"] = df["fx"]
    df["Y_D"] = df["fy"]
    df["N_D"] = df["mz"]
    df["X_H"] = df["fx_hull"]
    df["Y_H"] = df["fy_hull"]
    df["N_H"] = df["mz_hull"]

    if "fx_rudders" in df:
        df["X_R"] = df["fx_rudders"]
        df["Y_R"] = df["fy_rudders"]
        df["N_R"] = df["mz_rudders"]

    if "fx_rudder_port" in df:
        df["X_R_port"] = df["fx_rudder_port"]
        df["Y_R_port"] = df["fy_rudder_port"]
        df["N_R_port"] = df["mz_rudder_port"]
        df["X_R_stbd"] = df["fx_rudder_stb"]
        df["Y_R_stbd"] = df["fy_rudder_stb"]
        df["N_R_stbd"] = df["mz_rudder_stb"]

    df["x0"] = 0
    df["y0"] = 0
    df["psi"] = 0
    df["twa"] = 0
    df["tws"] = 0

    df["thrust_port"] = df["thrust"] / 2
    df["thrust_stbd"] = df["thrust"] / 2
    df['V'] = df['U'] = np.sqrt(df['u']**2 + df['v']**2)

    return df

def select(df_VCT_all_raw: dict) -> pd.DataFrame:
    """_summary_

    Parameters
    ----------
    df_VCT_all_raw : dict
        _description_

    Returns
    -------
    pd.DataFrame
        concatinated selection of VCT data for this ship
    """
    selection = [
        df_VCT_all_raw["7m_MS.df_VCT_all"](),
        #df_VCT_all_raw["V2_3_R2_MDL_additional.df_VCT"](),
    ]
    df_VCT_raw = pd.concat(selection, axis=0)
    
    
    return df_VCT_raw

def add_extra_circle_drift(df_VCT: pd.DataFrame) -> pd.DataFrame:
    log.info("Add extra Circle + Drift")

    df_VCT = add_extra_points_with_multiple_test_types(
        df_VCT=df_VCT,
        new_test_type="Circle + Drift",
        old_test_type="Circle",
        by=["V_round", "r_round"],
    )
    df_VCT = add_extra_points_with_multiple_test_types(
        df_VCT=df_VCT,
        new_test_type="Circle + Drift",
        old_test_type="Drift angle",
        by=["V_round", "beta_round"],
    )
    df_VCT.fillna(0, inplace=True)

    df_VCT = mirror_circle_drift(df_VCT=df_VCT)

    return df_VCT

def add_extra_points_with_multiple_test_types(
    df_VCT: pd.DataFrame,
    new_test_type="Circle + Drift",
    old_test_type="Circle",
    by=["V_round", "r_round"],
):
    df_VCT_extra = df_VCT.copy()
    df_VCT_extra["V_round"] = df_VCT_extra["V"].round(decimals=2)
    df_VCT_extra["r_round"] = df_VCT_extra["r"].round(decimals=3)
    df_VCT_extra["beta_round"] = df_VCT_extra["beta"].round(decimals=3)

    df = df_VCT_extra.groupby("test type").get_group(new_test_type)
    groups = df.groupby(by=by)

    key = by[1]
    for (V, r), group in groups:
        mask = (df_VCT_extra["test type"] == old_test_type) & (
            df_VCT_extra["V_round"] == V
        )
        df_circle = df_VCT_extra.loc[mask]
        rs = list(set(df_circle[key]) & set(df[key]))
        mask = df_circle[key].isin(rs)
        df_extra = df_circle.loc[mask].copy()
        if len(df_extra) > 0:
            df_extra["test type"] = new_test_type
            print(f"adding: {old_test_type}")
            df_VCT_extra = pd.concat(
                (df_VCT_extra, df_extra), axis=0, ignore_index=True
            )

    return df_VCT_extra

def mirror_circle_drift(df_VCT) -> pd.DataFrame:
    log.info("Mirror the Circle + Drift")

    df = df_VCT.groupby(by="test type").get_group("Circle + Drift")

    # mask = ((df['beta'].abs() > 0) & (df['r'].abs() > 0))
    df_mirror = df.copy()
    keys_fy = [key for key in df_mirror.columns if "fy" in key or "Y_" in key]
    keys_mz = [key for key in df_mirror.columns if "mz" in key or "N_" in key]
    keys_other = [
        "beta",
        "r",
        "v",
    ]
    keys = keys_other + keys_fy + keys_mz
    df_mirror[keys] *= -1

    # swap port/stbd:
    df_mirror_swap = df_mirror.copy()
    columns_port = [column for column in df_VCT.columns if "_port" in column]
    columns_stbd = [
        column for column in df_VCT.columns if "_stbd" in column or "_stb" in column
    ]
    assert len(columns_port) == len(columns_stbd)
    df_mirror_swap[columns_port] = df_mirror[columns_stbd]
    df_mirror_swap[columns_stbd] = df_mirror[columns_port]

    df_mirror_swap["mirror"] = True

    df_VCT = pd.concat((df_VCT, df_mirror_swap), axis=0, ignore_index=True)
    df_VCT["mirror"] = df_VCT["mirror"].fillna(False)

    return df_VCT

def prime(df_VCT: pd.DataFrame, models: dict= None, model=None) -> pd.DataFrame:
    log.info("Scaling VCT results to prime scale")
    
    if model is None:
        model = models['semiempirical_covered']()
    
    df_VCT_u0 = df_VCT.copy()
    # df_VCT_u0["u"] -= U0_ * np.sqrt(model.ship_parameters["scale_factor"])
    df_VCT_u0["u"] -= model.U0
    
    prime_system_ship = PrimeSystem(
        L=model.ship_parameters["L"] * model.ship_parameters["scale_factor"], rho=df_VCT.iloc[0]["rho"]
    )

    df_VCT_prime = prime_system_ship.prime(
        df_VCT_u0, U=df_VCT_u0["U"], only_with_defined_units=True
    )
    return df_VCT_prime

def scale(df_VCT: pd.DataFrame, ship_data: dict) -> pd.DataFrame:
    log.info("Scaling VCT results to model scale")
    prime_system_model = PrimeSystem(L=ship_data["L"], rho=df_VCT.iloc[0]["rho"])
    prime_system_ship = PrimeSystem(
        L=ship_data["L"] * ship_data["scale_factor"], rho=df_VCT.iloc[0]["rho"]
    )

    df_VCT_prime = prime_system_ship.prime(
        df_VCT, U=df_VCT["U"], only_with_defined_units=True
    )
    df_VCT_scaled = prime_system_model.unprime(
        df_VCT_prime,
        U=df_VCT["U"] / np.sqrt(ship_data["scale_factor"]),
        only_with_defined_units=True,
    )

    return df_VCT_scaled

def regress_hull_VCT(
    base_models: dict,
    df_VCT: pd.DataFrame,
    exclude_parameters: dict = {},
):
    #    log.info("""
    # ____________________________________
    # |                                    |
    # |       Regressing hull VCT          |
    # |____________________________________|
    # """)
    log.info(figlet_format("Hull VCT", font="starwars"))

    df_VCT = df_VCT.copy()

    models = {}

    for name, loader in base_models.items():
        base_model = loader()

        log.info(f"regressing VCT hull forces for:{name}")
        model, fits = _regress_hull_VCT(
            model=base_model,
            df_VCT=df_VCT,
            full_output=True,
            exclude_parameters=exclude_parameters,
        )

        models[name] = model
        # Also return fits?

    return models

def _regress_hull_VCT(
    model: ModularVesselSimulator,
    df_VCT: pd.DataFrame,
    full_output=False,
    exclude_parameters: dict = {},
):
    # log.info("Regressing hull VCT")
    from .regression_pipeline import pipeline, pipeline_RHI

    # First manual regression on the rudder parameters
    #model = manual_regression(model=model)

    log.info(
        "Optimization of rudder coefficients that are not so easy to regress with linear regression:"
    )
    parameters = ["kappa_v", "kappa_r", "kappa_v_gamma_g", "kappa_r_gamma_g"]
    log.info(f"Optimizing parameters:{parameters}")
    model, changes = phd.pipelines.regression_VCT.optimize.fit(
        model=model, data=df_VCT, parameters=parameters
    )
    log.info(f"Optimized the following parameters to:{changes}")

    parameters = ["C_D_tune", "C_D0_tune"]
    log.info(f"Optimizing parameters:{parameters}")
    model, changes = phd.pipelines.regression_VCT.optimize.fit(
        model=model, data=df_VCT, parameters=parameters, residual_keys=["X_R"]
    )
    log.info(f"Optimized the following parameters to:{changes}")

    model, fits_RHI = regress_VCT(
        model=model,
        df_VCT=df_VCT,
        pipeline=pipeline_RHI,
        exclude_parameters=exclude_parameters,
    )
    # Separating the rudder hull interaction coefficients:
    model.parameters["a_H"] = model.parameters["aH"] - 1
    model.parameters["x_H"] = model.parameters["xH"] - 1
    model.parameters.pop("aH")
    model.parameters.pop("xH")
    log.info(f"a_H is:{model.parameters['a_H']}")
    log.info(f"x_H is:{model.parameters['x_H']}")

    # Subtracting the rudder hull interaction from the hull forces:
    calculation = {}
    control = df_VCT[model.control_keys]
    states = df_VCT[model.states_str]
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
    df_VCT["Y_H"] -= model.parameters["a_H"] * df_force_predicted["Y_R"]
    df_VCT["N_H"] -= model.parameters["x_H"] * df_force_predicted["N_R"]
    # df_VCT["Y_H"] -= model.parameters["a_H"] * df_VCT["Y_R"]
    # df_VCT["N_H"] -= model.parameters["x_H"] * df_VCT["N_R"]

    model, fits = regress_VCT(
        model=model,
        df_VCT=df_VCT,
        pipeline=pipeline,
        exclude_parameters=exclude_parameters,
    )

    fits_all = fits_RHI
    fits_all.update(fits)

    if full_output:
        return model, fits_all
    else:
        return model
    
def regress_VCT(
    model: ModularVesselSimulator,
    df_VCT: pd.DataFrame,
    pipeline: dict,
    exclude_parameters: dict = {},
):
    from phd.pipelines.regression_VCT.regression_pipeline import fit

    df_VCT["U"] = df_VCT["V"]

    replacements = {
        "X_D": "fx",
        "Y_D": "fy",
        "N_D": "mz",
        "X_H": "fx_hull",
        "Y_H": "fy_hull",
        "N_H": "mz_hull",
        "X_R": "fx_rudders",
        "Y_R": "fy_rudders",
        "N_R": "mz_rudders",
    }
    for key, replacement in replacements.items():
        if not key in df_VCT:
            df_VCT[key] = df_VCT[replacement]

    zeros = [
        "x0",
        "y0",
        "psi",
        "twa",
        "tws",
    ]
    for zero in zeros:
        if not zero in df_VCT:
            df_VCT[zero] = 0

    df_VCT_prime = prime(df_VCT=df_VCT, model=model)
    
    ## Regression:
    regression_pipeline = pipeline(df_VCT_prime=df_VCT_prime, model=model)
    models, new_parameters = fit(
        regression_pipeline=regression_pipeline,
        model=model,
        exclude_parameters=exclude_parameters,
    )
    model.parameters.update(new_parameters)

    for key, fit in models.items():
        log.info(f"Regression:{key}")
        log.info(sp.pprint(fit.eq))
        try:
            log.info(fit.summary2().as_text())
        except Exception as e:
            raise ValueError(key)

    return model, models