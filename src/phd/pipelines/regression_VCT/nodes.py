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
from vct.read_shipflow import mirror_x_z
from phd.pipelines.models import optimize_l_R_shape

import logging

log = logging.getLogger(__name__)


def load_VCT(df_VCT: dict) -> pd.DataFrame:
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
    df_VCT_MDL_additional = df_VCT["V2_3_R2_MDL_additional.df_VCT"]()
    df_VCT_MDL_additional["fx_rudders"] = (
        df_VCT_MDL_additional["fx_rudder_port"] + df_VCT_MDL_additional["fx_rudder_stb"]
    )
    df_VCT_MDL_additional["fy_rudders"] = (
        df_VCT_MDL_additional["fy_rudder_port"] + df_VCT_MDL_additional["fy_rudder_stb"]
    )
    df_VCT_MDL_additional["mz_rudders"] = (
        df_VCT_MDL_additional["mz_rudder_port"] + df_VCT_MDL_additional["mz_rudder_stb"]
    )

    ## M5139-02-A_MS:
    df_VCT_MDL_M5139 = df_VCT["M5139-02-A_MS.df_VCT"]()
    renames = {
        "fx_Rudder_PS": "fx_rudder_port",
        "fy_Rudder_PS": "fy_rudder_port",
        "mz_Rudder_PS": "mz_rudder_port",
        "fx_Rudder_SB": "fx_rudder_stb",
        "fy_Rudder_SB": "fy_rudder_stb",
        "mz_Rudder_SB": "mz_rudder_stb",
    }
    df_VCT_MDL_M5139.rename(columns=renames, inplace=True)
    df_mirror = mirror_x_z(df_VCT_MDL_M5139.copy())

    # df_VCT = pd.concat((df_VCT_MDL, df_VCT_MDL_additional, df_VCT_MDL_M5139), axis=0)
    df_VCT = pd.concat((df_VCT_MDL_additional, df_VCT_MDL_M5139, df_mirror), axis=0)
    # df_VCT = df_VCT_MDL_M5139

    df_VCT["X_D"] = df_VCT["fx"]
    df_VCT["Y_D"] = df_VCT["fy"]
    df_VCT["N_D"] = df_VCT["mz"]
    df_VCT["X_H"] = df_VCT["fx_hull"]
    df_VCT["Y_H"] = df_VCT["fy_hull"]
    df_VCT["N_H"] = df_VCT["mz_hull"]
    df_VCT["X_R"] = df_VCT["fx_rudders"]
    df_VCT["Y_R"] = df_VCT["fy_rudders"]
    df_VCT["N_R"] = df_VCT["mz_rudders"]
    df_VCT["X_R_port"] = df_VCT["fx_rudder_port"]
    df_VCT["Y_R_port"] = df_VCT["fy_rudder_port"]
    df_VCT["N_R_port"] = df_VCT["mz_rudder_port"]
    df_VCT["X_R_stbd"] = df_VCT["fx_rudder_stb"]
    df_VCT["Y_R_stbd"] = df_VCT["fy_rudder_stb"]
    df_VCT["N_R_stbd"] = df_VCT["mz_rudder_stb"]
    df_VCT["x0"] = 0
    df_VCT["y0"] = 0
    df_VCT["psi"] = 0
    df_VCT["U"] = df_VCT["V"]
    df_VCT["twa"] = 0
    df_VCT["tws"] = 0

    df_VCT["thrust_port"] = df_VCT["thrust"] / 2
    df_VCT["thrust_stbd"] = df_VCT["thrust"] / 2

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


def add_extra_circle_drift(df_VCT: pd.DataFrame) -> pd.DataFrame:
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

    return df_VCT


def regress_hull_VCT(
    base_models: dict,
    df_VCT: pd.DataFrame,
    exclude_parameters: dict = {},
):
    log.info("Regressing hull VCT")

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

    model, fits = regress_VCT(
        model=model,
        df_VCT=df_VCT,
        pipeline=pipeline,
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

    model.parameters["kappa_outer"] = 0.94
    model.parameters["kappa_inner"] = 0.94
    model.parameters["kappa_gamma"] = 0.0

    model.parameters["l_R"] = 1.27 * model.ship_parameters["x_r"]
    c_ = (model.ship_parameters["c_t"] + model.ship_parameters["c_r"]) / 2
    model.ship_parameters["c_t"] = 1.30 * 0.1529126213592233
    model.ship_parameters["c_r"] = c_ * 2 - model.ship_parameters["c_t"]

    gamma_0_ = 0.044
    model.parameters["gamma_0_port"] = -gamma_0_
    model.parameters["gamma_0_stbd"] = gamma_0_

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

    df_VCT["X_D"] = df_VCT["fx"]
    df_VCT["Y_D"] = df_VCT["fy"]
    df_VCT["N_D"] = df_VCT["mz"]
    df_VCT["X_H"] = df_VCT["fx_hull"]
    df_VCT["Y_H"] = df_VCT["fy_hull"]
    df_VCT["N_H"] = df_VCT["mz_hull"]
    df_VCT["X_R"] = df_VCT["fx_rudders"]
    df_VCT["Y_R"] = df_VCT["fy_rudders"]
    df_VCT["N_R"] = df_VCT["mz_rudders"]
    df_VCT["x0"] = 0
    df_VCT["y0"] = 0
    df_VCT["psi"] = 0
    df_VCT["twa"] = 0
    df_VCT["tws"] = 0

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
        regression_pipeline=regression_pipeline, exclude_parameters=exclude_parameters
    )
    model.parameters.update(new_parameters)

    for key, fit in models.items():
        log.info(f"Regression:{key}")
        log.info(fit.summary2().as_text())

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
