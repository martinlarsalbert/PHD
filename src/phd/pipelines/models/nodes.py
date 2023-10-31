"""
This is a boilerplate pipeline 'models'
generated using Kedro 0.18.7
"""
import pandas as pd
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
from vessel_manoeuvring_models.substitute_dynamic_symbols import run
from wPCC_pipeline.pipelines.vct_data.nodes import vct_scaling
from vessel_manoeuvring_models.models.diff_eq_to_matrix import DiffEqToMatrix
import statsmodels.api as sm
from vessel_manoeuvring_models.models.modular_simulator import subs_simpler
from .subsystems import (
    add_propeller,
    add_propeller_simple,
    add_rudder,
    add_dummy_wind_force_system,
)
from .subsystems import add_wind_force_system as add_wind
from vessel_manoeuvring_models.prime_system import PrimeSystem

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
    f_Y_P = sp.Function("Y_P")(u, v, r, rev)
    f_N_P = sp.Function("N_P")(u, v, r, rev)

    eq_X_force = fx_eq.subs(X_D, f_X_H + f_X_R + f_X_P + f_X_W)
    eq_Y_force = fy_eq.subs(Y_D, f_Y_H + f_Y_R + f_Y_P + f_Y_W)
    eq_N_force = mz_eq.subs(N_D, f_N_H + f_N_R + f_N_P + f_N_W)

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


def vmm_7m_vct(main_model: ModularVesselSimulator) -> ModularVesselSimulator:
    from vessel_manoeuvring_models.models.vmm_7m_vct import eq_X_H, eq_Y_H, eq_N_H

    model = main_model.copy()

    equations_hull = [eq_X_H, eq_Y_H, eq_N_H]
    hull = PrimeEquationSubSystem(
        ship=model, equations=equations_hull, create_jacobians=True
    )
    model.subsystems["hull"] = hull

    add_propeller(model=model)
    add_rudder(model=model)
    add_dummy_wind_force_system(model=model)

    return model


def vmm_martins_simple(main_model: ModularVesselSimulator) -> ModularVesselSimulator:
    from vessel_manoeuvring_models.models.vmm_martin_simple import (
        X_qs_eq,
        Y_qs_eq,
        N_qs_eq,
    )

    model = main_model.copy()

    subs = [
        (X_D, X_H),
        (Y_D, Y_H),
        (N_D, N_H),
        (delta, 0),
        (thrust, 0),
    ]
    eq_X_H = X_qs_eq.subs(subs)
    eq_Y_H = Y_qs_eq.subs(subs)
    eq_N_H = N_qs_eq.subs(subs)

    equations_hull = [eq_X_H, eq_Y_H, eq_N_H]
    hull = PrimeEquationSubSystem(
        ship=model, equations=equations_hull, create_jacobians=True
    )
    model.subsystems["hull"] = hull

    add_propeller(model=model)
    add_rudder(model=model)
    add_dummy_wind_force_system(model=model)

    return model


def vmm_simple(main_model: ModularVesselSimulator) -> ModularVesselSimulator:
    from .vmm_simple import (
        eq_X_H,
        eq_Y_H,
        eq_N_H,
        eq_X_R,
        eq_Y_R,
        eq_N_R,
    )

    model = main_model.copy()
    equations_hull = [eq_X_H, eq_Y_H, eq_N_H]
    hull = PrimeEquationSubSystem(
        ship=model, equations=equations_hull, create_jacobians=True
    )
    model.subsystems["hull"] = hull

    ## Add rudder:
    equations_rudders = [eq_X_R, eq_Y_R, eq_N_R]
    rudders = PrimeEquationSubSystem(
        ship=model, equations=equations_rudders, create_jacobians=True
    )
    model.subsystems["rudders"] = rudders

    add_propeller(model=model)
    add_dummy_wind_force_system(model=model)

    return model


def vmm_martins_simple_thrust(
    main_model: ModularVesselSimulator,
) -> ModularVesselSimulator:
    from vessel_manoeuvring_models.models.vmm_martin_simple import (
        X_qs_eq,
        Y_qs_eq,
        N_qs_eq,
    )

    model = main_model.copy()
    model.control_keys = ["delta", "thrust"]  # Note!

    subs = [
        (X_D, X_H),
        (Y_D, Y_H),
        (N_D, N_H),
        (delta, 0),
        (thrust, 0),
    ]
    eq_X_H = X_qs_eq.subs(subs)
    eq_Y_H = Y_qs_eq.subs(subs)
    eq_N_H = N_qs_eq.subs(subs)

    equations_hull = [eq_X_H, eq_Y_H, eq_N_H]
    hull = PrimeEquationSubSystem(
        ship=model, equations=equations_hull, create_jacobians=True
    )
    model.subsystems["hull"] = hull

    ## Simple propellers:
    eq_X_P = sp.Eq(X_P, X_qs_eq.rhs - eq_X_H.rhs).subs(delta, 0)
    eq_Y_P = sp.Eq(Y_P, Y_qs_eq.rhs - eq_Y_H.rhs).subs(delta, 0)
    eq_N_P = sp.Eq(N_P, N_qs_eq.rhs - eq_N_H.rhs).subs(delta, 0)
    equations_propellers = [eq_X_P, eq_Y_P, eq_N_P]
    propellers = PrimeEquationSubSystem(
        ship=model, equations=equations_propellers, create_jacobians=True
    )
    model.subsystems["propellers"] = propellers
    model.parameters["Xthrust"] = 1 - model.ship_parameters["tdf"]

    ## Simple rudders:
    eq_X_R = sp.Eq(X_R, X_qs_eq.rhs - eq_X_H.rhs - eq_X_P.rhs)
    eq_Y_R = sp.Eq(Y_R, Y_qs_eq.rhs - eq_Y_H.rhs - eq_Y_P.rhs)
    eq_N_R = sp.Eq(N_R, N_qs_eq.rhs - eq_N_H.rhs - eq_N_P.rhs)
    equations_rudders = [eq_X_R, eq_Y_R, eq_N_R]
    rudders = PrimeEquationSubSystem(
        ship=model, equations=equations_rudders, create_jacobians=True
    )
    model.subsystems["rudders"] = rudders

    add_dummy_wind_force_system(model=model)

    return model


def add_wind_force_system(
    model: ModularVesselSimulator, wind_data_HMD: pd.DataFrame
) -> ModularVesselSimulator:
    model_wind = model.copy()
    add_wind(model_wind)
    model_wind = regress_wind_tunnel_test(model_wind, wind_data_HMD=wind_data_HMD)
    return model_wind


def regress_wind_tunnel_test(
    vmm_model: ModularVesselSimulator, wind_data_HMD: pd.DataFrame
) -> ModularVesselSimulator:
    from vessel_manoeuvring_models.models.wind_force import (
        eq_C_x,
        eq_C_y,
        eq_C_n,
        C_x,
        C_y,
        C_n,
    )

    eq_to_matrix_C_x = DiffEqToMatrix(eq_C_x, label=C_x, base_features=[awa])
    eq_to_matrix_C_y = DiffEqToMatrix(eq_C_y, label=C_y, base_features=[awa])
    eq_to_matrix_C_n = DiffEqToMatrix(eq_C_n, label=C_n, base_features=[awa])

    ## Regression:
    params_wind = {}
    for key, eq_to_matrix in zip(
        ["cx", "cy", "cn"], [eq_to_matrix_C_x, eq_to_matrix_C_y, eq_to_matrix_C_n]
    ):
        X, y = eq_to_matrix.calculate_features_and_label(
            data=wind_data_HMD, y=wind_data_HMD[key], simplify_names=False
        )
        ols = sm.OLS(y, X, hasconst=False)
        ols_fit = ols.fit()
        params_wind.update(ols_fit.params)

    params_wind = {key: value / 2 for key, value in params_wind.items()}  # Note 1/2
    model = vmm_model.copy()
    model.parameters.update(params_wind)

    return model


def regress_VCT(
    vmm_model: ModularVesselSimulator,
    df_VCT: pd.DataFrame,
    pipeline: dict,
    exclude_parameters: dict = {},
):
    from .regression_pipeline import fit

    model = vmm_model.copy()

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


def regress_hull_VCT(
    vmm_model: ModularVesselSimulator,
    df_VCT: pd.DataFrame,
    full_output=False,
    exclude_parameters: dict = {},
):
    log.info("Regressing hull VCT")
    from .regression_pipeline import pipeline, fit

    model, fits = regress_VCT(
        vmm_model=vmm_model,
        df_VCT=df_VCT,
        pipeline=pipeline,
        exclude_parameters=exclude_parameters,
    )

    if full_output:
        return model, fits
    else:
        return model


def regress_hull_rudder_VCT(
    vmm_model: ModularVesselSimulator,
    df_VCT: pd.DataFrame,
    full_output=False,
    exclude_parameters: dict = {},
):
    log.info("Regressing hull and rudder VCT")
    from .regression_pipeline import pipeline_with_rudder, fit

    # Note! Including the rudder forcces in the hull forces in this regression:
    df_VCT["fy_hull"] = df_VCT["fy_hull"] + df_VCT["fy_rudders"]
    df_VCT["mz_hull"] = df_VCT["mz_hull"] + df_VCT["mz_rudders"]

    model, fits = regress_VCT(
        vmm_model=vmm_model,
        df_VCT=df_VCT,
        pipeline=pipeline_with_rudder,
        exclude_parameters=exclude_parameters,
    )

    if full_output:
        return model, fits
    else:
        return model


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
    df_reference_speed_u["u"] -= hull.U0
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
    """Attempt to determine flow straightening 'kappa' based on mz forces from MDL tests

    Parameters
    ----------
    model : ModularVesselSimulator
        _description_
    time_series_smooth : dict
        _description_

    Returns
    -------
    ModularVesselSimulator
        _description_
    """
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


def regress_hull_inverse_dynamics(
    vmm_model: ModularVesselSimulator,
    data: pd.DataFrame,
    exclude_parameters: dict = {},
    full_output=False,
) -> ModularVesselSimulator:
    log.info("Regressing only hull from MDL with inverse dynamics")

    f_X_H = sp.Function("X_H")(u, v, r, delta)
    f_Y_H = sp.Function("Y_H")(u, v, r, delta)
    f_N_H = sp.Function("N_H")(u, v, r, delta)

    model = vmm_model.copy()

    # Solving X_H, Y_H, N_H from the model main equation:
    # This result in an expression like:
    # X_H(u, v, r, delta) = -X_{\dot{u}}*\dot{u} + \dot{u}*m - m*r**2*x_G - m*r*v - X_P(u, v, r, rev) - X_R(u, v, r, delta, thrust) - X_W
    eq_f_X_H = sp.Eq(f_X_H, sp.solve(model.X_eq, f_X_H)[0])
    eq_f_Y_H = sp.Eq(f_Y_H, sp.solve(model.Y_eq, f_Y_H)[0])
    eq_f_N_H = sp.Eq(f_N_H, sp.solve(model.N_eq, f_N_H)[0])

    data["V"] = data["U"] = np.sqrt(data["u"] ** 2 + data["v"] ** 2)
    data["rev"] = data[["Prop/SB/Rpm", "Prop/PS/Rpm"]].mean(axis=1)

    if not "twa" in data:
        data["twa"] = 0

    if not "tws" in data:
        data["tws"] = 0

    if "thrust" in data:
        data = data[
            model.states_str + model.control_keys + ["u1d", "v1d", "r1d", "U"]
        ].copy()
        # data.drop(columns=["thrust"], inplace=True)

    data["beta"] = -np.arctan2(data["v"], data["u"])

    # Precalculate the rudders, propellers and wind_force:
    calculation = {}
    model.parameters.update(exclude_parameters)
    for system_name, system in model.subsystems.items():
        if system_name == "hull":
            continue
        try:
            system.calculate_forces(
                states_dict=data[model.states_str],
                control=data[model.control_keys],
                calculation=calculation,
            )
        except KeyError as e:
            raise KeyError(f"Failed in subsystem:{system_name}")

    df_calculation = pd.DataFrame(calculation, index=data.index)
    data = pd.concat((data, df_calculation), axis=1)
    data_u0 = data.copy()
    # U0_ = float(data["u"].min())
    # data_u0["u"] -= U0_
    hull = model.subsystems["hull"]
    # hull.U0 = U0_
    data_u0["u"] -= model.U0

    calculation_columns = list(set(model.sub_system_keys) & set(df_calculation.columns))

    data_prime = model.prime_system.prime(
        data_u0[model.states_str + ["u1d", "v1d", "r1d"] + calculation_columns],
        U=data["U"],
    )

    eqs = [eq_f_X_H, eq_f_Y_H, eq_f_N_H]
    lambdas = {
        eq.lhs.name: lambdify(eq.rhs.subs(subs_simpler), substitute_functions=True)
        for eq in eqs
    }

    for key, lambda_ in lambdas.items():
        data_prime[key] = run(
            lambda_,
            inputs=data_prime,
            **model.ship_parameters_prime,
            **model.parameters,
        )

    ## Regression
    if not "Xthrust" in exclude_parameters:
        exclude_parameters["Xthrust"] = model.parameters["Xthrust"]
    else:
        model.parameters.update(exclude_parameters)

    eq_to_matrix_X_H = DiffEqToMatrix(
        hull.equations["X_H"],
        label=X_H,
        base_features=[u, v, r, thrust],
        exclude_parameters=exclude_parameters,
    )

    eq_to_matrix_Y_H = DiffEqToMatrix(
        hull.equations["Y_H"],
        label=Y_H,
        base_features=[u, v, r],
        exclude_parameters=exclude_parameters,
    )

    eq_to_matrix_N_H = DiffEqToMatrix(
        hull.equations["N_H"],
        label=N_H,
        base_features=[u, v, r],
        exclude_parameters=exclude_parameters,
    )

    models = {}
    new_parameters = {}
    fits = {}
    for eq_to_matrix in [eq_to_matrix_X_H, eq_to_matrix_Y_H, eq_to_matrix_N_H]:
        key = eq_to_matrix.acceleration_equation.lhs.name
        log.info(f"Regressing:{key}")
        X, y = eq_to_matrix.calculate_features_and_label(
            data=data_prime, y=data_prime[key]
        )
        ols = sm.OLS(y, X)
        models[key] = ols_fit = ols.fit()
        ols_fit.X = X
        ols_fit.y = y
        fits[key] = ols_fit
        new_parameters.update(ols_fit.params)
        log.info(ols_fit.summary().as_text())

    model.parameters.update(new_parameters)

    if full_output:
        return model, fits
    else:
        return model


def regress_inverse_dynamics(
    vmm_model: ModularVesselSimulator,
    data: pd.DataFrame,
    exclude_parameters={},
    full_output=False,
) -> ModularVesselSimulator:
    log.info("Regressing hull, propellers, and rudders from MDL with inverse dynamics")
    exclude_parameters = exclude_parameters.copy()
    model = vmm_model.copy()

    data["V"] = data["U"] = np.sqrt(data["u"] ** 2 + data["v"] ** 2)
    data["beta"] = -np.arctan2(data["v"], data["u"])

    data = model.forces_from_motions(data=data)
    data["X_D"] = data["fx"]
    data["Y_D"] = data["fy"]
    data["N_D"] = data["mz"]

    data_u0 = data.copy()
    # U0_ = float(data["u"].min())

    data_u0["u"] -= model.U0
    # model.subsystems["hull"].U0 = U0_
    # model.subsystems["propellers"].U0 = U0_
    # model.subsystems["rudders"].U0 = U0_

    data_prime = model.prime_system.prime(
        data_u0[
            model.states_str
            + ["u1d", "v1d", "r1d", "X_D", "Y_D", "N_D"]
            + model.control_keys
        ],
        U=data["U"],
    )

    ## Regression
    if not "Xthrust" in exclude_parameters:
        exclude_parameters["Xthrust"] = model.parameters["Xthrust"]

    if not "Xthrustport" in exclude_parameters:
        exclude_parameters["Xthrustport"] = model.parameters["Xthrustport"]

    if not "Xthruststbd" in exclude_parameters:
        exclude_parameters["Xthruststbd"] = model.parameters["Xthruststbd"]

    eq_X_D = model.expand_subsystemequations(model.X_D_eq)
    eq_to_matrix_X_D = DiffEqToMatrix(
        eq_X_D,
        label=X_D_,
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
        exclude_parameters=exclude_parameters,
    )

    eq_Y_D = model.expand_subsystemequations(model.Y_D_eq)
    eq_to_matrix_Y_D = DiffEqToMatrix(
        eq_Y_D,
        label=Y_D_,
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
        exclude_parameters=exclude_parameters,
    )

    eq_N_D = model.expand_subsystemequations(model.N_D_eq)
    eq_to_matrix_N_D = DiffEqToMatrix(
        eq_N_D,
        label=N_D_,
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
        exclude_parameters=exclude_parameters,
    )

    models = {}
    new_parameters = {}
    fits = {}
    for eq_to_matrix in [eq_to_matrix_X_D, eq_to_matrix_Y_D, eq_to_matrix_N_D]:
        key = eq_to_matrix.acceleration_equation.lhs.name
        log.info(f"Regressing:{key}")
        X, y = eq_to_matrix.calculate_features_and_label(
            data=data_prime,
            y=data_prime[key],
            parameters=model.ship_parameters,
        )
        ols = sm.OLS(y, X)
        fits[key] = ols_fit = ols.fit()
        new_parameters.update(ols_fit.params)
        log.info(ols_fit.summary().as_text())

    model.parameters.update(new_parameters)
    model.parameters.update(exclude_parameters)

    if full_output:
        return model, fits
    else:
        return model


def scale_model(
    model: ModularVesselSimulator, ship_data: dict
) -> ModularVesselSimulator:
    model_scaled = model.copy()

    scale_7 = ship_data["scale_factor"]
    scale_5 = model.ship_parameters["scale_factor"]
    scaling = scale_5 / scale_7
    ship_data["r_0"] = ship_data["D"] / 2
    ship_data["w_f"] = model.ship_parameters["w_p0"]

    if "x" in model.ship_parameters:
        ship_data["x"] = model.ship_parameters["x"] * scaling
    # ship_data["A_R"] = ship_parameters["A_R"] * (scale_5**2) / (scale_7**2)
    # ship_data["b_R"] = ship_parameters["b_R"] * scaling
    ship_data["x_p"] = model.ship_parameters["x_p"] * scaling
    ship_data["x_R"] = ship_data["x_r"]
    if "y_R" in model.ship_parameters:
        ship_data["y_R"] = model.ship_parameters["y_R"] * scaling

    model_scaled.set_ship_parameters(ship_data)
    return model_scaled
