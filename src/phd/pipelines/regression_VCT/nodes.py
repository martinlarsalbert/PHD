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
from vessel_manoeuvring_models.substitute_dynamic_symbols import run, lambdify
from phd.visualization.plot_prediction import predict
#from vct.read_shipflow import mirror_x_z
from phd.pipelines.models import optimize_l_R_shape

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
    #df_mirror = mirror_x_z(df_VCT_MDL_M5139.copy())
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
    
    
    for key,df in df_VCT_all_raw.items():
        if not isinstance(df,pd.DataFrame):
            df = df()

        extra_columns(df)        
        
    return df_VCT_all_raw
    
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

def select(df_VCT_all_raw:dict)->pd.DataFrame:
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
        df_VCT_all_raw["V2_3_R2_MDL_additional.df_VCT"](),
                
    ]
    df_VCT_raw = pd.concat(selection, axis=0)
    return df_VCT_raw
    

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
    
    df = df_VCT.groupby(by='test type').get_group('Circle + Drift')
    
    #mask = ((df['beta'].abs() > 0) & (df['r'].abs() > 0))
    df_mirror = df.copy()
    keys_fy = [key for key in df_mirror.columns if "fy" in key or "Y_" in key]
    keys_mz = [key for key in df_mirror.columns if "mz" in key or "N_" in key]
    keys_other = [
        'beta',
        'r',
        'v',
    ]
    keys = keys_other + keys_fy + keys_mz
    df_mirror[keys]*=-1
    
    #swap port/stbd:
    df_mirror_swap = df_mirror.copy()
    columns_port = [column for column in df_VCT.columns if '_port' in column]
    columns_stbd = [column for column in df_VCT.columns if '_stbd' in column or '_stb' in column]
    assert len(columns_port) == len(columns_stbd)
    df_mirror_swap[columns_port]=df_mirror[columns_stbd]
    df_mirror_swap[columns_stbd]=df_mirror[columns_port]
    
    df_mirror_swap["mirror"]=True

    df_VCT=pd.concat((df_VCT,df_mirror_swap),axis=0, ignore_index=True)
    df_VCT['mirror']=df_VCT['mirror'].fillna(False)
    
    return df_VCT

def regress_hull_VCT(
    base_models: dict,
    df_VCT: pd.DataFrame,
    exclude_parameters: dict = {},
):
    log.info("_______________  Regressing hull VCT _______________")

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
    log.info("Regressing hull VCT")
    from .regression_pipeline import pipeline, fit

    log.info("Precalculate the rudders, propellers and wind_force")
    calculation = {}
    model.parameters.update(exclude_parameters)
    for system_name, system in model.subsystems.items():
        if system_name == "hull":
            continue
        try:
            system.calculate_forces(
                states_dict=df_VCT[model.states_str],
                control=df_VCT[model.control_keys],
                calculation=calculation,
            )
        except KeyError as e:
            raise KeyError(f"Failed in subsystem:{system_name}")
    
    df_calculation = pd.DataFrame(calculation)
    df_calculation_prime = model.prime_system.prime(df_calculation[['Y_R','N_R']],U=df_VCT['V'])
    prime_system_ship = PrimeSystem(
        L=model.ship_parameters["L"] * model.ship_parameters["scale_factor"],
        rho=df_VCT.iloc[0]["rho"],
    )
    df_calculation_fullscale = prime_system_ship.unprime(df_calculation_prime, U=df_VCT['V'])
    
    for key,value in df_calculation_fullscale.items():
        if not key in df_VCT:
            log.info(f"Adding calculated:{key}")
        else:
            log.info(f"Replacing with calculated:{key}")

        df_VCT[key] = value
    
    model, fits = regress_VCT(
        model=model,
        df_VCT=df_VCT,
        pipeline=pipeline,
        exclude_parameters=exclude_parameters,
    )

    model.parameters['a_H'] = model.parameters.pop('aH')
    model.parameters['x_H'] = model.parameters["axH"]/model.parameters['a_H']
    
    model = manual_regression(model=model)

    if full_output:
        return model, fits
    else:
        return model

def regress_hull_rudder_VCT(
    base_models: dict,
    df_VCT: pd.DataFrame,
    exclude_parameters: dict = {},
):
    log.info("______________________ Regressing hull+rudder VCT ______________________")

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
    
    model.parameters["delta_lim"] = np.deg2rad(90)
    covered = model.ship_parameters["D"] / model.ship_parameters["b_R"] * 0.65
    model.ship_parameters["A_R_C"] = model.ship_parameters["A_R"] * covered
    model.ship_parameters["A_R_U"] = model.ship_parameters["A_R"] * (1 - covered)
    #model.parameters['kappa_outer']=1.0
    #model.parameters['kappa_inner']=0.75
    #model.parameters['kappa_gamma']=0.3
    model.parameters['kappa_outer']=0.94
    model.parameters['kappa_inner']=0.94
    model.parameters['kappa_gamma']=0.0
    #model.parameters['delta_lim']=10000
    model.ship_parameters['w_f']=0.27
    #model.parameters['l_R']=1.2*model.ship_parameters['x_r']
    #model.parameters['l_R']=1.38*model.ship_parameters['x_r']
    model.parameters['l_R']=1.27*model.ship_parameters['x_r']
    c_ = (model.ship_parameters['c_t'] + model.ship_parameters['c_r'])/2
    model.ship_parameters['c_t']=1.30*0.1529126213592233
    model.ship_parameters['c_r'] = (c_*2-model.ship_parameters['c_t'])

    gamma_0_ = 0.044
    model.parameters['gamma_0_port']=gamma_0_
    model.parameters['gamma_0_stbd']=-gamma_0_
    

    return model


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
        "X_D" : "fx",
        "Y_D" : "fy",
        "N_D" : "mz",
        "X_H" : "fx_hull",
        "Y_H" : "fy_hull",
        "N_H" : "mz_hull",
        "X_R" : "fx_rudders",
        "Y_R" : "fy_rudders",
        "N_R" : "mz_rudders",       
    }
    for key,replacement in replacements.items():
        if not key in df_VCT:
            df_VCT[key] = df_VCT[replacement]
    
    zeros = ["x0","y0","psi","twa","tws",]
    for zero in zeros:
        if not zero in df_VCT:
            df_VCT[zero]= 0

    ## Prime system
    # U0_ = df_VCT["V"].min()
    U0_ = model.U0
    df_VCT_u0 = df_VCT.copy()
    df_VCT_u0["u"] -= U0_ * np.sqrt(model.ship_parameters["scale_factor"])

    keys = (
        model.states_str
        + ["beta", "V", "U"]
        + model.control_keys
        + ["X_D", "Y_D", "N_D", "X_H", "Y_H", "N_H", "X_R", "Y_R", "N_R"]
        + ["test type", "model_name"]
    )
    prime_system_ship = PrimeSystem(
        L=model.ship_parameters["L"] * model.ship_parameters["scale_factor"],
        rho=df_VCT.iloc[0]["rho"],
    )
    df_VCT_prime = prime_system_ship.prime(df_VCT_u0[keys], U=df_VCT["U"])

    # for name, subsystem in model.subsystems.items():
    #    if isinstance(subsystem, PrimeEquationSubSystem):
    #        # subsystem.U0 = U0_ / np.sqrt(
    #        #    model.ship_parameters["scale_factor"]
    #        # )  # Important!
    #        subsystem.U0 = U0_  # Important!

    ## Regression:
    regression_pipeline = pipeline(df_VCT_prime=df_VCT_prime, model=model)
    models, new_parameters = fit(
        regression_pipeline=regression_pipeline, model=model, exclude_parameters=exclude_parameters
    )
    model.parameters.update(new_parameters)

    for key, fit in models.items():
        log.info(f"Regression:{key}")
        try:
            log.info(fit.summary2().as_text())
        except Exception as e:
            raise ValueError(key)

    return model, models


def adopting_to_MDL(models_VCT: dict, resistance_MDL: pd.DataFrame) -> dict:
    models = {}
    for name, loader in models_VCT.items():
        model = loader()
        add_MDL_resistance(model=model, resistance=resistance_MDL)

        model.parameters["delta_alpha_s"] = 0  # Delayed stall

        factor = 0.80
        model.parameters["X0"] *= factor
        model.parameters["Xu"] *= factor

        # model.parameters["kappa"] = 0.85
        model.parameters["l_R"] = 1.5 * model.parameters["l_R"]
        # model.parameters['Yvdot']*=0.5

        model.parameters["Yvdot"] *= 0.55
        model.parameters["Nrdot"] *= 0.7

        models[name] = model

    return models


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

    return model


def add_MDL_resistance(
    model: ModularVesselSimulator, resistance: pd.DataFrame
) -> ModularVesselSimulator:
    zero = pd.DataFrame({"u": [0.0], "X_D": [0]}, index=[0])
    df_ = pd.concat((zero, resistance), axis=0).fillna(0)
    coeffs = np.polyfit(x=df_["u"], y=df_["X_D"], deg=3)
    resistance_prediction = pd.DataFrame(columns=["u"])
    resistance_prediction["u"] = np.linspace(0, resistance["u"].max(), 100)
    resistance_prediction["X_D"] = np.polyval(coeffs, x=resistance_prediction["u"])

    hull = model.subsystems["hull"]
    # hull.U0=0.1
    resistance_extrapolation = pd.DataFrame(columns=["u", "X_D"])
    resistance_extrapolation["u"] = np.linspace(hull.U0, resistance["u"].min(), 3)
    resistance_extrapolation["X_D"] = np.polyval(
        coeffs, x=resistance_extrapolation["u"]
    )
    columns = model.states_str + model.control_keys
    columns.remove("u")
    resistance_extrapolation[columns] = 0
    resistance_extrapolation["V"] = resistance_extrapolation["u"]

    df_resistance_ = pd.concat((resistance_extrapolation, resistance), axis=0).fillna(0)

    df_resistance_[["thrust_port", "thrust_stbd"]] = 0
    subs = {value: value.name for value in model.X_D_eq.rhs.args}
    columns = list(subs.values())
    columns.remove("X_H")
    prediction = predict(model, data=df_resistance_)
    df_resistance_[columns] = prediction[columns]

    df_resistance_["u"] -= hull.U0
    df_resistance_prime = model.prime_system.prime(
        df_resistance_[model.states_str + model.control_keys + ["X_D"] + columns],
        U=df_resistance_["V"],
    )

    ## Regress the resistance:
    # eq_X_D = model.expand_subsystemequations(model.X_D_eq)
    # eq = eq_X_D.subs([(v, 0), (r, 0), (thrust_stbd, 0), (thrust_port, 0), (X_RHI, 0)])

    eq_X_D = model.X_D_eq.subs(subs)
    eq_X_H = sp.Eq(X_H, sp.solve(eq_X_D, X_H)[0])
    lambda_X_H = lambdify(eq_X_H.rhs)
    df_resistance_prime["X_H"] = run(lambda_X_H, inputs=df_resistance_prime)

    eq = (
        model.subsystems["hull"]
        .equations["X_H"]
        .subs(
            [
                (v, 0),
                (r, 0),
            ]
        )
    )

    eq_to_matrix = DiffEqToMatrix(
        eq,
        label=X_H,
        base_features=[
            u,
            v,
            r,
            delta,
            thrust,
            thrust_port,
            thrust_stbd,
            y_p_port,
            y_p_stbd,
        ],
        exclude_parameters={
            "Xuuu": 0,
            #'Xuu':0,
            #'X0':0
        },
    )

    X, y = eq_to_matrix.calculate_features_and_label(
        data=df_resistance_prime,
        y=df_resistance_prime["X_H"],
        parameters=model.ship_parameters,
    )

    ols = sm.OLS(y, X)
    ols_fit = ols.fit()
    model.parameters.update(ols_fit.params)
    
    ## Manual tweaking:
    factor = 0.8
    model.parameters['X0']*=factor
    model.parameters['Xu']*=factor
    
