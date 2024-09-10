"""
This is a boilerplate pipeline 'regression_VCT'
generated using Kedro 0.18.7
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
    ### MDL:
    # df_VCT_MDL = df_VCT["V2_3_R2_MDL.df_VCT"]()
    # df_VCT_MDL["fx_rudders"] = (
    #    df_VCT_MDL["fx_rudder_port"] + df_VCT_MDL["fx_rudder_stb"]
    # )
    # df_VCT_MDL["fy_rudders"] = (
    #    df_VCT_MDL["fy_rudder_port"] + df_VCT_MDL["fy_rudder_stb"]
    # )
    # df_VCT_MDL["mz_rudders"] = (
    #    df_VCT_MDL["mz_rudder_port"] + df_VCT_MDL["mz_rudder_stb"]
    # )
    #
    ### Additional:
    df_VCT_MDL_additional = df_VCT_all_raw["V2_3_R2_MDL_additional.df_VCT"]()
    df_VCT_MDL_additional["fx_rudders"] = (
        df_VCT_MDL_additional["fx_rudder_port"] + df_VCT_MDL_additional["fx_rudder_stb"]
    )
    df_VCT_MDL_additional["fy_rudders"] = (
        df_VCT_MDL_additional["fy_rudder_port"] + df_VCT_MDL_additional["fy_rudder_stb"]
    )
    df_VCT_MDL_additional["mz_rudders"] = (
        df_VCT_MDL_additional["mz_rudder_port"] + df_VCT_MDL_additional["mz_rudder_stb"]
    )
    df_VCT_all_raw["V2_3_R2_MDL_additional.df_VCT"] = df_VCT_MDL_additional

    ## M5139-02-A_MS:
    df_VCT_MDL_M5139 = df_VCT_all_raw["M5139-02-A_MS.df_VCT"]()
    renames = {
        "fx_Rudder_PS": "fx_rudder_port",
        "fy_Rudder_PS": "fy_rudder_port",
        "mz_Rudder_PS": "mz_rudder_port",
        "fx_Rudder_SB": "fx_rudder_stb",
        "fy_Rudder_SB": "fy_rudder_stb",
        "mz_Rudder_SB": "mz_rudder_stb",
    }
    df_VCT_MDL_M5139.rename(columns=renames, inplace=True)
    # df_mirror = mirror_x_z(df_VCT_MDL_M5139.copy())
    df_VCT_all_raw["M5139-02-A_MS.df_VCT"] = df_VCT_MDL_M5139

    df = df_VCT_all_raw["M5139-02-A_straightening_MS.df_VCT"]()  # (model scale)
    df.rename(columns=renames, inplace=True)
    df_VCT_all_raw["M5139-02-A_straightening_MS.df_VCT"] = df

    df = df_VCT_all_raw["M5139-02-A_straightening.df_VCT"]()  # (full scale)
    df.rename(columns=renames, inplace=True)
    df_VCT_all_raw["M5139-02-A_straightening.df_VCT"] = df

    df = df_VCT_all_raw["V2_3_R2_power.df_VCT"]()  # (full scale)
    df.rename(columns=renames, inplace=True)
    df_VCT_all_raw["V2_3_R2_power.df_VCT"] = df

    for key, df in df_VCT_all_raw.items():
        if not isinstance(df, pd.DataFrame):
            df = df()

        extra_columns(df)

    return df_VCT_all_raw


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
    df["U"] = df["V"]
    df["twa"] = 0
    df["tws"] = 0

    df["thrust_port"] = df["thrust"] / 2
    df["thrust_stbd"] = df["thrust"] / 2

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
        df_VCT_all_raw["M5139-02-A_MS.df_VCT"](),
        #df_VCT_all_raw["V2_3_R2_MDL_additional.df_VCT"](),
    ]
    df_VCT_raw = pd.concat(selection, axis=0)
    
    
    return df_VCT_raw

def load_MDL(tests:dict, exclude=[])->pd.DataFrame:
    
    _ = []
    for name, loader in tests.items():
        if name in exclude:
            continue
        
        df = loader()
        df = df.rolling(window=100).mean().dropna()  # simple filtering
        df['id'] = name
        _.append(df)
    
    data_MDL = pd.concat(_,axis=0)
    return data_MDL

def limit_states_for_regression(df_VCT_scaled:pd.DataFrame,tests:dict, exclude=[])->pd.DataFrame:
    
    exclude = list([str(key) for key in exclude])
    
    #Only pick states that are relevant for the MDL tests:
    data_MDL = load_MDL(tests=tests, exclude=exclude)
    data_MDL['beta'] = np.arctan2(-data_MDL['v'],data_MDL['u'])
    
    keys = ['beta','r']
    log.info(f"Only using VCT data with {keys} with states of MDL tests:{data_MDL['id'].unique()}")
    for key in keys:
        max_value = data_MDL[key].abs().max()
        log.info(f"max({key})={max_value}")
        mask = df_VCT_scaled[key].abs() <= max_value
        df_VCT_scaled=df_VCT_scaled.loc[mask].copy()
    
    log.info(f"Limiting states in VCT data: {df_VCT_scaled[keys].abs().max()}")
    
    return df_VCT_scaled
    

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
    log.info(f"Exclude parameters: {exclude_parameters}")

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
    model = manual_regression(model=model)

    # log.info("Precalculate the rudders, propellers and wind_force")
    # calculation = {}
    # model.parameters.update(exclude_parameters)
    # for system_name, system in model.subsystems.items():
    #    if system_name == "hull":
    #        continue
    #    try:
    #        system.calculate_forces(
    #            states_dict=df_VCT[model.states_str],
    #            control=df_VCT[model.control_keys],
    #            calculation=calculation,
    #        )
    #    except KeyError as e:
    #        raise KeyError(f"Failed in subsystem:{system_name}")
    # df_calculation = pd.DataFrame(calculation)

    # df_calculation_prime = model.prime_system.prime(df_calculation[['Y_R','N_R']],U=df_VCT['V'])
    # prime_system_ship = PrimeSystem(
    #    L=model.ship_parameters["L"] * model.ship_parameters["scale_factor"],
    #    rho=df_VCT.iloc[0]["rho"],
    # )
    # df_calculation_fullscale = prime_system_ship.unprime(df_calculation_prime, U=df_VCT['V'])
    # for key,value in df_calculation.items():
    #    if not key in df_VCT:
    #        log.info(f"Adding calculated:{key}")
    #    else:
    #        log.info(f"Replacing with calculated:{key}")
    #
    #    df_VCT[key] = value

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


def regress_hull_rudder_VCT(
    base_models: dict,
    df_VCT: pd.DataFrame,
    exclude_parameters: dict = {},
):
    log.info(figlet_format("Hull + Rudder VCT", font="starwars"))

    df_VCT = df_VCT.copy()

    models = {}

    for name, loader in base_models.items():
        base_model = loader()

        log.info(f"regressing VCT hull and rudder forces for:{name}")
        model, fits = _regress_hull_rudder_VCT(
            model=base_model,
            df_VCT=df_VCT,
            full_output=True,
            exclude_parameters=exclude_parameters,
        )

        models[name] = model
        # Also return fits?

    return models


def _regress_hull_rudder_VCT(
    model: ModularVesselSimulator,
    df_VCT: pd.DataFrame,
    full_output=False,
    exclude_parameters: dict = {},
):
    log.info("Regressing hull and rudder VCT")
    from .regression_pipeline import pipeline_with_rudder, fit

    model, fits = regress_VCT(
        model=model,
        df_VCT=df_VCT,
        pipeline=pipeline_with_rudder,
        exclude_parameters=exclude_parameters,
    )

    model = manual_regression(model=model)

    if full_output:
        return model, fits
    else:
        return model


def manual_regression(model: ModularVesselSimulator) -> ModularVesselSimulator:
    """Manual regression based on visual inspection."""

    covered = model.ship_parameters["D"] / model.ship_parameters["b_R"] * 0.65
    model.ship_parameters["A_R_C"] = model.ship_parameters["A_R"] * covered
    model.ship_parameters["A_R_U"] = model.ship_parameters["A_R"] * (1 - covered)
    # model.parameters['kappa_outer']=0.94
    # model.parameters['kappa_inner']=0.94
    model.parameters["kappa_v"] = 0.94
    model.parameters["kappa_r"] = 0.94
    model.parameters["kappa_v_gamma_g"] = 0
    model.parameters["kappa_r_gamma_g"] = 0
    # model.parameters['kappa_gamma']=0.0

    # model.parameters['l_R']=1.27*model.ship_parameters['x_r']
    c_ = (model.ship_parameters["c_t"] + model.ship_parameters["c_r"]) / 2
    model.ship_parameters["c_t"] = 1.30 * 0.1529126213592233
    model.ship_parameters["c_r"] = c_ * 2 - model.ship_parameters["c_t"]

    gamma_0_ = 0.044
    model.parameters["gamma_0_port"] = gamma_0_
    model.parameters["gamma_0_stbd"] = -gamma_0_

    model.parameters["C_D_tune"] = 1.25
    model.parameters["C_D0_tune"] = 4.8
    
    model.parameters['delta_lim'] = np.deg2rad(15)
    model.parameters['s'] = 0
    model.ship_parameters['w_f']=0.297

    return model


def df_VCT_to_prime(
    model: ModularVesselSimulator, df_VCT: pd.DataFrame
) -> pd.DataFrame:
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

    U0_ = model.U0
    df_VCT_u0 = df_VCT.copy()
    df_VCT_u0["u"] -= U0_ * np.sqrt(model.ship_parameters["scale_factor"])

    keys = (
        model.states_str
        + ["beta", "V", "U"]
        + model.control_keys
        + ["X_D", "Y_D", "N_D", "X_H", "Y_H", "N_H", "X_R", "Y_R", "N_R"]
        + ["test type", "model_name","thrust"]
    )

    df_VCT_prime = model.prime_system.prime(df_VCT_u0[keys], U=df_VCT["U"])
    return df_VCT_prime


def regress_VCT(
    model: ModularVesselSimulator,
    df_VCT: pd.DataFrame,
    pipeline: dict,
    exclude_parameters: dict = {},
):
    from phd.pipelines.regression_VCT.regression_pipeline import fit

    ## scale VCT data
    # df_VCT = vct_scaling(data=df_VCT, ship_data=model.ship_parameters)
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

    ## Prime system
    # U0_ = df_VCT["V"].min()
    U0_ = model.U0
    df_VCT_u0 = df_VCT.copy()
    # df_VCT_u0["u"] -= U0_ * np.sqrt(model.ship_parameters["scale_factor"])
    df_VCT_u0["u"] -= U0_

    keys = (
        model.states_str
        + ["beta", "V", "U"]
        + model.control_keys
        + ["X_D", "Y_D", "N_D", "X_H", "Y_H", "N_H", "X_R", "Y_R", "N_R"]
        + ["test type", "model_name"]
    )
    # prime_system_ship = PrimeSystem(
    #    L=model.ship_parameters["L"] * model.ship_parameters["scale_factor"],
    #    rho=df_VCT.iloc[0]["rho"],
    # )
    # df_VCT_prime = prime_system_ship.prime(df_VCT_u0[keys], U=df_VCT["U"])

    df_VCT_prime = model.prime_system.prime(df_VCT_u0[keys], U=df_VCT["U"])

    # for name, subsystem in model.subsystems.items():
    #    if isinstance(subsystem, PrimeEquationSubSystem):
    #        # subsystem.U0 = U0_ / np.sqrt(
    #        #    model.ship_parameters["scale_factor"]
    #        # )  # Important!
    #        subsystem.U0 = U0_  # Important!

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


def adopting_to_MDL(
    models_VCT: dict, resistance_MDL: pd.DataFrame, #tests_ek: dict
) -> dict:
    log.info(figlet_format("Adopting to MDL", font="starwars"))

    #data_MDL_many = gather_data(tests_ek=tests_ek)

    models = {}
    for name, loader in models_VCT.items():
        model = loader()
        add_MDL_resistance(model=model, resistance=resistance_MDL)

        model.parameters["delta_alpha_s"] = 0  # Delayed stall
        
        #model.parameters['Nr']*=2
        #model.parameters['Nrdot']*=0.25

        models[name] = model

    return models


def adopting_nonlinear_to_MDL(
    models_VCT: dict, resistance_MDL: pd.DataFrame, #tests_ek: dict
) -> dict:
    log.info(figlet_format("Adopting nonlinear to MDL", font="starwars"))

    #data_MDL_many = gather_data(tests_ek=tests_ek)

    models = {}
    for name, loader in models_VCT.items():
        model = loader()
        add_MDL_resistance(model=model, resistance=resistance_MDL)

        model.parameters["delta_alpha_s"] = 0  # Delayed stall

        model.parameters["s"] = 0
        
        # CMT corrections (most likely wave generation):
        factor = 1.30
        model.parameters['Yv']*=factor
        model.parameters['Yvvv']*=factor

        factor = 1.05
        model.parameters['Nv']*=factor
        model.parameters['Nvvv']*=factor
        
        # Rudder yawing moment:
        model.ship_parameters['x_R']=1.15*-2.45  # This seems to be necessary to get the right yawing moment
                
        ## This correction was also necessary. (probably wave generation) (see: 148.01_selective_IDR.ipynb)
        model.parameters['Nr']*=2.2536644742604537
        model.parameters['Nrrr']*=1.1095856121752687
        
        #model.parameters['Nr']*=2
        #model.parameters['Nrdot']*=0.25

        models[name] = model

    return models

def adopting_hull_rudder_to_MDL(
    models_rudder_VCT: dict, resistance_MDL: pd.DataFrame, models_VCT: dict
) -> dict:
    model_VCT = models_VCT["semiempirical_covered"]()

    models = {}
    for name, loader in models_rudder_VCT.items():
        model = loader()
        add_MDL_resistance(model=model, resistance=resistance_MDL)

        model.parameters["Nrdot"] = model_VCT.parameters["Nrdot"]
        model.parameters["Yvdot"] = model_VCT.parameters["Yvdot"]

        models[name] = model

    return models


def fit_Nrdot(
    model: ModularVesselSimulator,
    data_MDL_many: pd.DataFrame,
):
    log.info("Fitting the Nrdot...")
    fit = fit_added_mass(
        model=model,
        data_MDL_many=data_MDL_many,
        eq=model.N_eq,
        move=p.Nrdot * r1d,
        x="r1d",
    )
    denominator = run(
        lambdify(df_parameters.loc["Nrdot", "denominator"]),
        inputs=model.ship_parameters,
    )
    new_value = fit.params["r1d"] / denominator

    change = new_value / model.parameters["Nrdot"]
    # if ((change < 0.2) or (change>3)):
    #    raise ValueError(f"Fitted Nrdot is {np.round(change,2)} times the old one, which is too large difference!")

    log.info(f"Fitted Nrdot is {np.round(change,2)} times the old one")

    model.parameters["Nrdot"] = new_value
    return model


def fit_Yvdot(
    model: ModularVesselSimulator,
    data_MDL_many: pd.DataFrame,
):
    log.info("Fitting the Yvdot...")
    fit = fit_added_mass(
        model=model,
        data_MDL_many=data_MDL_many,
        eq=model.Y_eq,
        move=p.Yvdot * v1d,
        x="v1d",
    )
    denominator = run(
        lambdify(df_parameters.loc["Yvdot", "denominator"]),
        inputs=model.ship_parameters,
    )
    new_value = fit.params["v1d"] / denominator

    change = new_value / model.parameters["Yvdot"]
    # if ((change < 0.2) or (change>3)):
    #    raise ValueError(f"Fitted Yvdot is {np.round(change,2)} times the old one, which is too large difference!")

    log.info(f"Fitted Yvdot is {np.round(change,2)} times the old one")
    model.parameters["Yvdot"] = new_value
    return model


def fit_added_mass(
    model: ModularVesselSimulator, data_MDL_many: pd.DataFrame, eq, move, x
):
    prediction = predict(model=model, data=data_MDL_many)

    mask = prediction[x].abs() > prediction[x].abs().quantile(0.4)
    prediction_ = prediction.loc[mask]

    eq_added_masses = sp.Eq(sp.solve(eq, move)[0], move)
    eq_added_masses

    eq_ = eq_added_masses.subs(
        [
            (v1d, "v1d"),
            (r1d, "r1d"),
            (p.Yrdot, "Yrdot"),
            (p.Nvdot, "Nvdot"),
        ]
    )
    lambda_y = lambdify(eq_.lhs, substitute_functions=True)

    y = run(
        lambda_y,
        inputs=prediction_,
        Nvdot=model.parameters["Nvdot"],
        Yrdot=model.parameters["Yrdot"],
        **model.ship_parameters,
    )
    # y = run(lambda_y, inputs=prediction_, Nvdot=0, Yrdot=0, **model.ship_parameters)

    X = prediction_[[x]]

    ols_model = sm.OLS(y, X)
    fit = ols_model.fit()
    log.info(fit.summary2())
    return fit


def gather_data(tests_ek: dict):
    ids = [
        22773,
        22772,
        22770,
        22764,
    ]

    data_MDL_many = []
    for id in ids:
        data_MDL = tests_ek[f"{id}"]()
        data_MDL["V"] = data_MDL["U"] = np.sqrt(data_MDL["u"] ** 2 + data_MDL["v"] ** 2)
        data_MDL["beta"] = -np.arctan2(data_MDL["v"], data_MDL["u"])
        
        if "Prop/PS/Rpm" in data_MDL:
            data_MDL["rev"] = data_MDL[["Prop/PS/Rpm", "Prop/SB/Rpm"]].mean(axis=1)
        
        data_MDL["twa"] = 0
        data_MDL["tws"] = 0
        data_MDL["theta"] = 0
        data_MDL["q"] = 0
        
        if not 'phi' in data_MDL:
            data_MDL["phi"] = data_MDL["roll"]
        
        data_MDL["p"] = 0
        data_MDL["q1d"] = 0
        data_MDL["thrust_port"] = data_MDL["Prop/PS/Thrust"]
        data_MDL["thrust_stbd"] = data_MDL["Prop/SB/Thrust"]
        data_MDL_many.append(data_MDL)

    data_MDL_many = pd.concat(data_MDL_many, axis=0)
    return data_MDL_many


def shape_optimization(
    models_VCT: dict, resistance_MDL: pd.DataFrame, tests_ek_smooth_joined: pd.DataFrame
) -> dict:
    models = {}
    for name, loader in models_VCT.items():
        model = loader()
        model = optimize(model=model, tests_ek_smooth_joined=tests_ek_smooth_joined)
        add_MDL_resistance(model=model, resistance=resistance_MDL)

        models[name] = model

    return models


def optimize(
    model: ModularVesselSimulator, tests_ek_smooth_joined
) -> ModularVesselSimulator:
    ids = [
        22773,
        22772,
        22770,
        22764,
    ]

    mask = tests_ek_smooth_joined["id"].isin(ids)
    data = tests_ek_smooth_joined.loc[mask].copy()

    data["thrust_port"] = data["Prop/PS/Thrust"]
    data["thrust_stbd"] = data["Prop/SB/Thrust"]

    optimization = optimize_l_R_shape.fit(model=model, data=data)

    model.parameters["l_R"] = optimization.x[0]
    model.parameters["kappa_outer"] = optimization.x[1]
    model.parameters["kappa_inner"] = optimization.x[1]

    log.info(
        f"The shape oprimizer gave: l_R={optimization.x[0]}, kappa={optimization.x[1]}, scaler N ={optimization.x[2]}, scaler Y = {optimization.x[3]}"
    )

    model = fit_Nrdot(model=model, data_MDL_many=data)
    model = fit_Yvdot(model=model, data_MDL_many=data)

    return model


def add_MDL_resistance(
    model: ModularVesselSimulator, resistance: pd.DataFrame
) -> ModularVesselSimulator:
    # Extrapolate down to minimum speed U0:
    hull = model.subsystems["hull"]
    resistance_extrapolation = pd.DataFrame(columns=["u", "thrust"])
    resistance_extrapolation["u"] = resistance_extrapolation["U"] = np.linspace(
        1.1 * hull.U0, resistance["u"].min(), 3
    )

    i = resistance["u"].idxmin()
    first = resistance.loc[i]
    a = first["thrust"] / (first["u"] ** 2)
    resistance_extrapolation["thrust"] = a * resistance_extrapolation["u"] ** 2
    df_resistance = pd.concat((resistance_extrapolation, resistance), axis=0).fillna(0)
    df_resistance["thrust_port"] = df_resistance["thrust_stbd"] = (
        df_resistance["thrust"] / 2
    )

    # Later for weighted regression:
    mask = df_resistance["u"] > 0.6
    df_resistance.loc[mask, "weight"] = 1
    df_resistance.loc[~mask, "weight"] = 0.01

    # Precalculate rudder and propeller etc:
    df_calculation = model.calculate_subsystems(
        states_dict=df_resistance[model.states_str],
        control=df_resistance[model.control_keys],
    )
    df = pd.merge(
        left=df_calculation,
        right=df_resistance,
        how="left",
        left_index=True,
        right_index=True,
        suffixes=("", "_MDL"),
    )

    # ______________  Define the regression ____________:
    # X_D = X_RHI + X_H(u, v, r, δ) + X_P(u, v, r, rev) + X_R(u, v, r, δ, thrust) + X_W(Cₓ₀, awa, Cₓ₁, A_XV, aws, Cₓ₃, Cₓ₂, Cₓ₅, ρ_A, Cₓ₄)
    eq = remove_functions(model.X_D_eq)
    # -> X_D = X_H + X_P + X_R + X_RHI + X_W

    eq_X_H = sp.Eq(X_H, sp.solve(eq, X_H)[0])
    # -> X_H = X_D - X_P - X_R - X_RHI - X_W
    lambda_X_H = equation_to_python_method(eq_X_H)
    # Calculate what the hull resistance must be:
    df["X_H"] = lambda_X_H(**df)

    eq_regression = hull.equations["X_H"].subs([(v, 0), (r, 0)])
    # -> X_H = X_{0} + X_{u}⋅u

    exclude_parameters = {}
    eq_to_matrix_X_H = DiffEqToMatrix(
        eq_regression,
        label=X_H,
        base_features=[u, v, r, thrust],
        exclude_parameters=exclude_parameters,
    )

    # Convert to prime system:
    df_u0 = df.copy()
    df_u0["u"] -= model.U0
    df_prime = model.prime_system.prime(
        df_u0, U=df_u0["U"], only_with_defined_units=True
    )

    X, y = eq_to_matrix_X_H.calculate_features_and_label(
        data=df_prime, y=df_prime["X_H"]
    )
    ols = sm.WLS(y, X, weights=df["weight"])
    ols_fit = ols.fit()
    model.parameters.update(ols_fit.params)
    log.info(ols_fit.summary2())
    r2_SI = r2_score(y_true=df["X_H"], y_pred=predict(model, df)["X_H"])
    log.info(f"In SI units r2 is {r2_SI}")
    return model

def get_eq_X_PMM(model:ModularVesselSimulator):
    X_PMM = symbols("X_PMM")
    eq = model.X_eq.subs(model.X_D_eq.rhs,model.X_D_eq.lhs)
    eq_X_PMM = Eq(X_PMM,eq.rhs-eq.lhs)
    return eq_X_PMM

def get_eq_Y_PMM(model:ModularVesselSimulator):
    Y_PMM = symbols("Y_PMM")
    eq = model.Y_eq.subs(model.Y_D_eq.rhs,model.Y_D_eq.lhs)
    eq_Y_PMM = Eq(Y_PMM,eq.rhs-eq.lhs)
    return eq_Y_PMM

def get_eq_N_PMM(model:ModularVesselSimulator):
    N_PMM = symbols("N_PMM")
    eq = model.N_eq.subs(model.N_D_eq.rhs,model.N_D_eq.lhs)
    eq_N_PMM = Eq(N_PMM,eq.rhs-eq.lhs)
    return eq_N_PMM


def subtract_centrifugal_and_centrepetal_forces(df_VCT:pd.DataFrame, model:ModularVesselSimulator)->pd.DataFrame:
    
    X_PMM,Y_PMM,N_PMM = symbols("X_PMM,Y_PMM,N_PMM")
    X_VCT,Y_VCT,N_VCT = symbols("X_VCT,Y_VCT,N_VCT")    
    
    eq_X_PMM = get_eq_X_PMM(model=model)
    eq_Y_PMM = get_eq_Y_PMM(model=model)
    eq_N_PMM = get_eq_N_PMM(model=model)
    
    subs_steady_state = {
    X_PMM:X_VCT,
    Y_PMM:Y_VCT,
    N_PMM:N_VCT,
    u1d:0,
    v1d:0,
    r1d:0,
    m:0,
    }

    eq_X_VCT = eq_X_PMM.subs(subs_steady_state)
    eq_Y_VCT = eq_Y_PMM.subs(subs_steady_state)
    eq_N_VCT = eq_N_PMM.subs(subs_steady_state)
    
    eq_X_D = Eq(X_D_,sp.solve(eq_X_VCT, X_D_)[0])
    eq_Y_D = Eq(Y_D_,sp.solve(eq_Y_VCT, Y_D_)[0])
    eq_N_D = Eq(N_D_,sp.solve(eq_N_VCT, N_D_)[0])
    
    X_H_VCT,Y_H_VCT,N_H_VCT = symbols("X_H_VCT,Y_H_VCT,N_H_VCT")
    subs_hull = {
        X_D_:X_H,
        X_VCT:X_H_VCT,

        Y_D_:Y_H,
        Y_VCT:Y_H_VCT,

        N_D_:N_H,
        N_VCT:N_H_VCT,

    }
    eq_X_H = eq_X_D.subs(subs_hull)
    eq_Y_H = eq_Y_D.subs(subs_hull)
    eq_N_H = eq_N_D.subs(subs_hull)
    
    subs = {value:key for key,value in p.items()}

    lambda_X_D = lambdify(eq_X_D.rhs.subs(subs))
    lambda_Y_D = lambdify(eq_Y_D.rhs.subs(subs))
    lambda_N_D = lambdify(eq_N_D.rhs.subs(subs))

    lambda_X_H = lambdify(eq_X_H.rhs.subs(subs))
    lambda_Y_H = lambdify(eq_Y_H.rhs.subs(subs))
    lambda_N_H = lambdify(eq_N_H.rhs.subs(subs))

    lambda_X_VCT = lambdify(eq_X_VCT.rhs.subs(subs))
    lambda_Y_VCT = lambdify(eq_Y_VCT.rhs.subs(subs))
    lambda_N_VCT = lambdify(eq_N_VCT.rhs.subs(subs))
    
    df_VCT_viscous = df_VCT.copy()

    df_VCT_viscous['X_VCT'] = df_VCT_viscous['X_D']
    df_VCT_viscous['Y_VCT'] = df_VCT_viscous['Y_D']
    df_VCT_viscous['N_VCT'] = df_VCT_viscous['N_D']

    df_VCT_viscous['X_H_VCT'] = df_VCT_viscous['X_H']
    df_VCT_viscous['Y_H_VCT'] = df_VCT_viscous['Y_H']
    df_VCT_viscous['N_H_VCT'] = df_VCT_viscous['N_H']

    df_VCT_viscous['X_H'] = run(lambda_X_H, inputs=df_VCT_viscous, **added_masses)
    df_VCT_viscous['Y_H'] = run(lambda_Y_H, inputs=df_VCT_viscous, **added_masses)
    df_VCT_viscous['N_H'] = run(lambda_N_H, inputs=df_VCT_viscous, **added_masses)

    df_VCT_viscous['X_D'] = run(lambda_X_D, inputs=df_VCT_viscous, **added_masses)
    df_VCT_viscous['Y_D'] = run(lambda_Y_D, inputs=df_VCT_viscous, **added_masses)
    df_VCT_viscous['N_D'] = run(lambda_N_D, inputs=df_VCT_viscous, **added_masses)
    
    return df_VCT_viscous