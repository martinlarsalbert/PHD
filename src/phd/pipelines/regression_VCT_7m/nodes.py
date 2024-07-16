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
from vessel_manoeuvring_models.sympy_helpers import eq_move_to_LHS, eq_remove
from vessel_manoeuvring_models.substitute_dynamic_symbols import get_function_subs
from sympy import Eq,symbols

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

    mask = df_VCT['test type'].isin([
    'Cirlce',
    'Drift angle',
    'Circle + Drift',
    ])

    df_raw = df_VCT.loc[mask]
    
    df_mirror = df_raw.copy()
    keys_fy = [key for key in df_mirror.columns if "fy" in key or "Y_" in key]
    keys_mz = [key for key in df_mirror.columns if "mz" in key or "N_" in key]
    keys_other = [
        "beta",
        "r",
        "v",
    ]
    keys = keys_other + keys_fy + keys_mz
    df_mirror[keys] *= -1
    mask = df_mirror.duplicated(['beta','r'])
    df_mirror = df_mirror.loc[~mask].copy()
    
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

def prime(df_VCT: pd.DataFrame, models: dict= None, model=None, fullscale=True) -> pd.DataFrame:
    log.info("Scaling VCT results to prime scale")
    
    if model is None:
        model = models['semiempirical_covered']()
    
    df_VCT_u0 = df_VCT.copy()
    # df_VCT_u0["u"] -= U0_ * np.sqrt(model.ship_parameters["scale_factor"])
    df_VCT_u0["u"] -= model.U0
    
    if fullscale:
        prime_system_ship = PrimeSystem(
            L=model.ship_parameters["L"] * model.ship_parameters["scale_factor"], rho=df_VCT.iloc[0]["rho"]
        )
    else:
        prime_system_ship = PrimeSystem(
            L=model.ship_parameters["L"], rho=df_VCT.iloc[0]["rho"]
        )
    
    
    df_VCT_prime = prime_system_ship.prime(
           df_VCT_u0, U=df_VCT_u0["U"], only_with_defined_units=True
    )
    
    df_VCT_prime['speed_kts'] = df_VCT['V']*3.6/1.852
    
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
    models: dict,
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

    for name, loader in models.items():
        model = loader()

        log.info(f"regressing VCT hull forces for:{name}")
        model, fits = _regress_hull_VCT(
            model=model,
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
    
    #To slice the calculation up till the rudders system:
    precalculate_subsystems=model.find_providing_subsystems_recursive(model.subsystems['rudders'])
    precalculate_subsystems = list(np.flipud(precalculate_subsystems))  # calculation order...
    precalculate_subsystems.append('rudders')
    
    for precalculate_subsystem in precalculate_subsystems:
        calculation = model.subsystems[precalculate_subsystem].calculate_forces(
            states_dict=states, control=control, calculation=calculation)
    
    #calculation = model.subsystems["rudder_port"].calculate_forces(
    #    states_dict=states, control=control, calculation=calculation
    #)
    #calculation = model.subsystems["rudder_stbd"].calculate_forces(
    #    states_dict=states, control=control, calculation=calculation
    #)
    #calculation = model.subsystems["rudders"].calculate_forces(
    #    states_dict=states, control=control, calculation=calculation
    #)
    
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
    from .regression_pipeline import fit

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

    df_VCT_prime = prime(df_VCT=df_VCT, model=model, fullscale=False)
    
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
    model.parameters['s'] = -10
    model.ship_parameters['w_f']=0.297

    return model

def scale_resistance(ship_data:dict, df_resistance_TT:pd.DataFrame)->pd.DataFrame:
    alpha_TT = 25
    df_resistance_TT['U'] = df_resistance_TT['VS [kn]']*1.852/3.6/np.sqrt(alpha_TT)
    df_resistance_TT['X_D'] = -df_resistance_TT['RTm [N]']
    df_resistance_TT['u'] = df_resistance_TT['U']
    
    ## Scale to 7m
    L = ship_data['L']
    scale_factor = ship_data['scale_factor']
    prime_system = PrimeSystem(L, rho=1000)
        
    L_TT = L*scale_factor/alpha_TT
    prime_system_TT = PrimeSystem(L_TT, rho=1000)
    
    df_resistance_prime = prime_system_TT.prime(df_resistance_TT, U=df_resistance_TT['U'], only_with_defined_units=True)
    df_resistance_SI = prime_system.unprime(df_resistance_prime, U=df_resistance_TT['U']*np.sqrt(alpha_TT)/np.sqrt(scale_factor), only_with_defined_units=True)
    
    return df_resistance_SI

def wave_generation_correction(models:dict, df_resistance_SI:pd.DataFrame, exclude_parameters={})-> dict:
    
    corrected_models = {}
    
    log.info("From the TT captive tests and the MOTIONS calculations it has been concluded that the VCT underpredicts the forces in the drift angles. This methods applied a correction on the hydrodynamic derivatives.")
    
    for name, loader in models.items():
        model = loader()
        
        model = _wave_generation_correction(model=model, df_resistance_SI=df_resistance_SI, exclude_parameters=exclude_parameters)
        
        corrected_models[name] = model
        
    return corrected_models

def _wave_generation_correction(model:ModularVesselSimulator, df_resistance_SI:pd.DataFrame, exclude_parameters={})-> ModularVesselSimulator:
    """From the TT captive tests and the MOTIONS calculations it has been concluded that the VCT underpredicts the forces in the drift angles.
    This methods applied a correction on the hydrodynamic derivatives.    

    Args:
        model (ModularVesselSimulator): _description_

    Returns:
        ModularVesselSimulator: _description_
    """
    
    factor = 1.30
    model.parameters['Yv']*=factor
    model.parameters['Yvvv']*=factor
    
    factor = 1.05
    model.parameters['Nv']*=factor
    model.parameters['Nvvv']*=factor
    
    ## Adding the wave resistance from the TT test...
    model = _TT_resistance(model=model, df_resistance_SI=df_resistance_SI, exclude_parameters=exclude_parameters)
    
    return model
    

def _TT_resistance(model:ModularVesselSimulator, df_resistance_SI:pd.DataFrame, exclude_parameters={})->ModularVesselSimulator:
    
    model_no_prop = model.copy()  # Model without propeller...
    ## Remove the propeller:
    model_no_prop.subsystems.pop('propeller_port')
    model_no_prop.subsystems.pop('propeller_stbd')
    model_no_prop.subsystems.pop('propellers')
    model_no_prop.control_keys+=['thrust_port','thrust_stbd']

    df_resistance_SI[model_no_prop.states_str] = 0
    df_resistance_SI[model_no_prop.control_keys] = 0  # thrust_port/stbd = 0 --> V=(1-w)*u
    df_resistance_SI['u'] = df_resistance_SI['U']


    ## Define the regression:
    X_D_eq = model_no_prop.X_D_eq.subs(get_function_subs(model_no_prop.X_D_eq))
    
    X_P, X_RHI, X_W = symbols("X_P, X_RHI, X_W")
    eq_precalulate_X = eq_remove(eq_move_to_LHS(eq=X_D_eq, symbol=X_R), [X_P,X_RHI,X_W])
    log.info(f"Regression: {eq_precalulate_X}")
    
    precalculated_subsystems=model_no_prop.find_precalculated_subsystems(eq=eq_precalulate_X)
    
    log.info(f"precalculating: {precalculated_subsystems}")
    df_calculation = model_no_prop.precalculate_subsystems(data=df_resistance_SI, precalculated_subsystems=precalculated_subsystems)
    
    data_with_precalculation = pd.concat((df_resistance_SI,df_calculation), axis=1)
    
    eq_regression_RHS = model_no_prop.expand_subsystemequations(eq_precalulate_X.rhs)
    eq_regression_RHS = eq_regression_RHS.subs([
        (v,0),
        (r,0),
    ]
    )
    
    eq_regression_LHS = eq_precalulate_X.lhs
    eq_regression = Eq(eq_regression_LHS, eq_regression_RHS)

    dof = 'X'
    label = symbols(f"{dof}_label")
    eq_regression_subs = Eq(label,eq_regression.rhs)

    label_name = str(eq_regression_subs.lhs)
    data_with_precalculation[label_name] = run(lambdify(eq_regression.lhs), data_with_precalculation)
    
    
    eq_to_matrix = DiffEqToMatrix(
        eq_regression_subs,
        label=label,
        base_features=[
            u,
        ],
        exclude_parameters=exclude_parameters,
    )
    
    label_name = str(eq_to_matrix.label)
    
    units = {label_name:"force",}
    data_with_precalculation_prime = model.prime(data_with_precalculation, units=units)
    
    ols_fit = eq_to_matrix.fit(data=data_with_precalculation_prime,
                y=data_with_precalculation_prime[label_name],
                parameters=model.ship_parameters,
                simplify_names=True,                
                )
    
    log.info(ols_fit.summary2())
    
    model.parameters.update(ols_fit.params)
    
    return model