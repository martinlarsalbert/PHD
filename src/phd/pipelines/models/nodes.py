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
from vessel_manoeuvring_models.substitute_dynamic_symbols import run
from wPCC_pipeline.pipelines.vct_data.nodes import vct_scaling
from vessel_manoeuvring_models.models.diff_eq_to_matrix import DiffEqToMatrix
import statsmodels.api as sm
from vessel_manoeuvring_models.models.modular_simulator import subs_simpler
from .subsystems import add_propeller, add_rudder, add_dummy_wind_force_system

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


def regress_hull_VCT(vmm_model: ModularVesselSimulator, df_VCT: pd.DataFrame):

    log.info("Regressing VCT")
    from .regression_pipeline import pipeline, fit

    model = vmm_model.copy()

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

    hull = model.subsystems["hull"]
    hull.V0 = V0_  # Important!

    ## Regression:
    regression_pipeline = pipeline(df_VCT_prime=df_VCT_prime, hull=hull)
    models, new_parameters = fit(regression_pipeline=regression_pipeline)
    model.parameters.update(new_parameters)
    model_summaries = {key: fit.summary().as_text() for key, fit in models.items()}

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
    vmm_model: ModularVesselSimulator, time_series_smooth: dict
) -> ModularVesselSimulator:

    log.info("Regressing MDL with inverse dynamics")
    from vessel_manoeuvring_models.models.vmm_7m_vct import eq_X_H, eq_Y_H, eq_N_H

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

    data = time_series_smooth["wpcc.updated.joined.ek_smooth"]()
    data["V"] = data["U"] = np.sqrt(data["u"] ** 2 + data["v"] ** 2)
    data["rev"] = data[["Prop/SB/Rpm", "Prop/PS/Rpm"]].mean(axis=1)
    data.drop(columns=["thrust"], inplace=True)
    data["beta"] = -np.arctan2(data["v"], data["u"])

    # Precalculate the rudders, propellers and wind_force:
    calculation = {}
    for system_name, system in model.subsystems.items():
        if system_name == "hull":
            continue
        system.calculate_forces(
            states_dict=data[model.states_str],
            control=data[model.control_keys],
            calculation=calculation,
        )

    df_calculation = pd.DataFrame(calculation, index=data.index)
    data = pd.concat((data, df_calculation), axis=1)
    data_u0 = data.copy()
    V0_ = float(data["u"].min())
    data_u0["u"] -= V0_
    hull = model.subsystems["hull"]
    hull.V0 = V0_

    data_prime = model.prime_system.prime(
        data_u0[
            model.states_str + ["u1d", "v1d", "r1d"] + list(df_calculation.columns)
        ],
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
    exclude_parameters = {"Xthrust": model.parameters["Xthrust"]}

    eq_to_matrix_X_H = DiffEqToMatrix(
        eq_X_H,
        label=X_H,
        base_features=[u, v, r, thrust],
        exclude_parameters=exclude_parameters,
    )

    eq_to_matrix_Y_H = DiffEqToMatrix(eq_Y_H, label=Y_H, base_features=[u, v, r])

    eq_to_matrix_N_H = DiffEqToMatrix(eq_N_H, label=N_H, base_features=[u, v, r])

    models = {}
    new_parameters = {}
    for eq_to_matrix in [eq_to_matrix_X_H, eq_to_matrix_Y_H, eq_to_matrix_N_H]:
        key = eq_to_matrix.acceleration_equation.lhs.name
        log.info(f"Regressing:{key}")
        X, y = eq_to_matrix.calculate_features_and_label(
            data=data_prime, y=data_prime[key]
        )
        ols = sm.OLS(y, X)
        models[key] = ols_fit = ols.fit()
        new_parameters.update(ols_fit.params)
        log.info(ols_fit.summary().as_text())

    model.parameters.update(new_parameters)

    return model
