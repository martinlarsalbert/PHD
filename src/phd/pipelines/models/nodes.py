"""
This is a boilerplate pipeline 'models'
generated using Kedro 0.18.7
"""
from vessel_manoeuvring_models.symbols import *
from vessel_manoeuvring_models.nonlinear_vmm_equations import (
    X_eom,
    Y_eom,
    N_eom,
    fx_eq,
    fy_eq,
    mz_eq,
)
from vessel_manoeuvring_models.models.modular_simulator import (
    ModularVesselSimulator,
    function_eq,
)
from vessel_manoeuvring_models.models.wind_force import eq_X_W, eq_Y_W, eq_N_W
from vessel_manoeuvring_models.parameters import df_parameters

p = df_parameters["symbol"]

from vessel_manoeuvring_models.models.subsystem import PrimeEquationSubSystem
from vessel_manoeuvring_models.models.wind_force import (
    WindForceSystem,
    DummyWindForceSystem,
)
from vessel_manoeuvring_models.substitute_dynamic_symbols import run
from wPCC_pipeline.pipelines.vct_data.nodes import vct_scaling
from vessel_manoeuvring_models.models.propeller import PropellersSystem
from vessel_manoeuvring_models.models.semiempirical_rudder import (
    SemiempiricalRudderSystem,
)
from vessel_manoeuvring_models.models.diff_eq_to_matrix import DiffEqToMatrix
import statsmodels.api as sm

import logging

log = logging.getLogger(__name__)


def main_model() -> ModularVesselSimulator:
    """General model that is used to create all other models

    Returns
    -------
    ModularVesselSimulator
        _description_
    """
    log.info("Creating the general model")

    f_X_H = sp.Function("X_H")(u, v, r, delta)
    f_Y_H = sp.Function("Y_H")(u, v, r, delta)
    f_N_H = sp.Function("N_H")(u, v, r, delta)

    f_X_R = sp.Function("X_R")(u, v, r, delta, thrust)
    f_Y_R = sp.Function("Y_R")(u, v, r, delta, thrust)
    f_N_R = sp.Function("N_R")(u, v, r, delta, thrust)

    f_X_W = function_eq(eq_X_W).lhs
    f_Y_W = function_eq(eq_Y_W).lhs
    f_N_W = function_eq(eq_N_W).lhs

    f_X_P = sp.Function("X_P")(u, v, r, rev)

    eq_X_force = fx_eq.subs(X_D, f_X_H + f_X_R + f_X_P + f_X_W)
    eq_Y_force = fy_eq.subs(Y_D, f_Y_H + f_Y_R + f_Y_W)
    eq_N_force = mz_eq.subs(N_D, f_N_H + f_N_R + f_N_W)

    X_eq = X_eom.subs(X_force, eq_X_force.rhs)
    Y_eq = Y_eom.subs(Y_force, eq_Y_force.rhs)
    N_eq = N_eom.subs(N_force, eq_N_force.rhs)
    subs = [
        (p.Xvdot, 0),
        (p.Xrdot, 0),
        (p.Yudot, 0),
        # (p.Yrdot,0),  # this is probably not true
        (p.Nudot, 0),
        # (p.Nvdot,0),# this is probably not true
    ]
    X_eq = X_eq.subs(subs)
    Y_eq = Y_eq.subs(subs)
    N_eq = N_eq.subs(subs)

    main_model = ModularVesselSimulator(
        X_eq=X_eq,
        Y_eq=Y_eq,
        N_eq=N_eq,
        ship_parameters={},
        parameters={},
        control_keys=["delta", "rev"],
        do_create_jacobian=False,
    )
    return main_model


def add_added_mass(model: ModularVesselSimulator):

    mask = df_parameters["state"] == "dot"
    lambdas_added_mass = df_parameters.loc[mask, "brix_lambda"].dropna()
    added_masses = {
        key: run(lambda_, **model.ship_parameters)
        for key, lambda_ in lambdas_added_mass.items()
    }
    model.parameters.update(added_masses)


def main_model_with_scale(
    main_model: ModularVesselSimulator, ship_data: dict
) -> ModularVesselSimulator:

    model = main_model.copy()
    model.set_ship_parameters(ship_data)
    add_added_mass(model)

    return model


def regress_hull_VCT(main_model: ModularVesselSimulator, df_VCT: pd.DataFrame):

    log.info("Regressing VCT")
    from vessel_manoeuvring_models.models.vmm_7m_vct import eq_X_H, eq_Y_H, eq_N_H
    from .regression_pipeline import pipeline, fit

    model = main_model.copy()

    ## scale VCT data
    df_VCT = vct_scaling(data=df_VCT, ship_data=model.ship_parameters)
    df_VCT["U"] = df_VCT["V"]

    ## Prime system
    V0_ = df_VCT["V"].min()
    df_VCT_u0 = df_VCT.copy()
    df_VCT_u0["u"] -= V0_

    units = {
        "fx_hull": "force",
        "fy_hull": "force",
        "mz_hull": "moment",
    }
    df_VCT_prime = model.prime_system.prime(
        df_VCT_u0[["u", "v", "r", "fx_hull", "fy_hull", "mz_hull", "test type"]],
        U=df_VCT["U"],
        units=units,
    )
    df_VCT_prime["X_H"] = df_VCT_prime["fx_hull"]
    df_VCT_prime["Y_H"] = df_VCT_prime["fy_hull"]
    df_VCT_prime["N_H"] = df_VCT_prime["mz_hull"]

    equations_hull = [eq_X_H, eq_Y_H, eq_N_H]
    hull = PrimeEquationSubSystem(
        ship=model, equations=equations_hull, create_jacobians=True, V0=V0_
    )
    model.subsystems["hull"] = hull

    ## Regression:
    regression_pipeline = pipeline(df_VCT_prime=df_VCT_prime, hull=hull)
    models, new_parameters = fit(regression_pipeline=regression_pipeline)
    model.parameters.update(new_parameters)
    model_summaries = {key: fit.summary().as_text() for key, fit in models.items()}

    ## Add propeller:
    propellers = PropellersSystem(ship=model, create_jacobians=True)
    model.subsystems["propellers"] = propellers
    # Propeller coefficients (regressed in 06.10_wPCC_vmm_propeller_model.ipynb)
    params = {
        "C0_w_p0": 0.10378571428571445,
        "C1_w_p0": 0.24690520231438584,
        "k_0": 0.576581716472807,
        "k_1": -0.3683675998138215,
        "k_2": -0.07542975438913463,
    }
    model.parameters.update(params)
    g_ = 9.81
    model.parameters["g"] = g_
    model.parameters["Xthrust"] = 1 - model.ship_parameters["tdf"]

    prop_data = {
        "r_0": model.ship_parameters["D"] / 2,
        "x": 0.10,  # Guessing...
    }
    model.ship_parameters.update(prop_data)

    ## Add rudder:
    rudders = SemiempiricalRudderSystem(ship=model, create_jacobians=False)
    model.subsystems["rudders"] = rudders
    rudder_particulars = {
        "x_R": model.ship_parameters["x_r"],
        "y_R": 0,
        "z_R": 0,
        "w_f": model.ship_parameters["w_p0"],
    }
    model.ship_parameters.update(rudder_particulars)
    rudder_parameters = {
        "C_L_tune": 1.0,
        # "delta_lim": np.deg2rad(40),
        "delta_lim": 2 * 0.6981317007977318,
        "kappa": 0.5,  # (Small value means much flow straightening)
        "nu": 1.18849e-06,
    }
    model.parameters.update(rudder_parameters)

    ## Add dummy wind system:
    wind_force = DummyWindForceSystem(ship=model, create_jacobians=True)
    model.subsystems["wind_force"] = wind_force

    return model, model_summaries


def correct_vct_resistance(
    model: ModularVesselSimulator,
    time_series_meta_data: pd.DataFrame,
    time_series_smooth: dict,
) -> ModularVesselSimulator:
    log.info("Correcting the resistance based on MDL reference speed tests")

    time_series_meta_data.index = time_series_meta_data.index.astype(str)
    time_series_meta_data.rename(columns={"test_type": "test type"}, inplace=True)

    ## Gather reference speed runs from MDL:
    from .mdl_resistance import reference_test_resistance

    df_reference_speed = reference_test_resistance(
        model=model,
        time_series_meta_data=time_series_meta_data,
        time_series_smooth=time_series_smooth,
    )

    df_reference_speed_u = df_reference_speed.copy()
    hull = model.subsystems["hull"]
    df_reference_speed_u["u"] -= hull.V0
    df_reference_speed_u_prime = model.prime_system.prime(
        df_reference_speed_u[["u", "v", "r", "X_H"]], U=df_reference_speed_u["V"]
    )

    ## Regression
    log.info("Regression MDL reference run")
    eq = hull.equations["X_H"].subs(
        [
            (v, 0),
            (r, 0),
        ]
    )
    label = X_H
    eq_to_matrix = DiffEqToMatrix(
        eq, label=label, base_features=[u, v, r, thrust], exclude_parameters={}
    )

    data_ = df_reference_speed_u_prime
    assert len(data_) > 0
    key = eq_to_matrix.acceleration_equation.lhs.name
    X, y = eq_to_matrix.calculate_features_and_label(data=data_, y=data_[key])

    ols = sm.OLS(y, X)
    ols_fit = ols.fit()
    model2 = model.copy()
    model2.parameters.update(ols_fit.params)
    log.info(ols_fit.summary().as_text())

    return model2


def optimize_kappa(
    model: ModularVesselSimulator, time_series_smooth: dict
) -> ModularVesselSimulator:
    from .optimize_kappa import fit_kappa

    log.info(
        "Optimizing rudder flow straightening 'kappa' based on mz forces from MDL tests"
    )
    data = time_series_smooth["wpcc.updated.joined.ek_smooth"]()
    data["V"] = data["U"] = np.sqrt(data["u"] ** 2 + data["v"] ** 2)
    data["rev"] = data[["Prop/SB/Rpm", "Prop/PS/Rpm"]].mean(axis=1)
    data["beta"] = -np.arctan2(data["v"], data["u"])

    R_min = 10
    mask = (
        data["r"].abs() > (data["u"].mean() / (R_min * model.ship_parameters["L"]))
    ) & (data["beta"].abs() > np.deg2rad(5))
    data_selected = data.loc[mask]
    log.info(
        f"Using {len(data_selected)} from {len(data)} total samples in the kappa optimization (excluding tests with small drift angle or yaw rate)."
    )

    result = fit_kappa(model=model.copy(), data=data_selected)
    model_optimized_kappa = model.copy()
    if result.success:
        kappa = result.x[0]
        log.info(f"Optimisation suceeded with estimated kappa:{kappa}")
        model_optimized_kappa.parameters["kappa"] = kappa
    else:
        log.warning(f"Optimisation failed with message:{result.message}")

    return model_optimized_kappa
