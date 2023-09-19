import logging

log = logging.getLogger(__name__)

import pandas as pd
from vessel_manoeuvring_models.models.modular_simulator import (
    ModularVesselSimulator,
    find_functions,
)
from vessel_manoeuvring_models.symbols import *
from vessel_manoeuvring_models.models.modular_simulator import subs_simpler
from vessel_manoeuvring_models.substitute_dynamic_symbols import run
from vessel_manoeuvring_models.models.diff_eq_to_matrix import DiffEqToMatrix
import statsmodels.api as sm


def fit_resistance(
    model_base: ModularVesselSimulator, resistance: pd.DataFrame
) -> ModularVesselSimulator:
    resistance["u"] = resistance.index
    resistance["X_D"] = -resistance["Rts(N)"]
    resistance[["x0", "y0", "psi", "v", "r", "delta", "rev", "twa", "tws"]] = 0

    df_resistance_prime = model_base.prime_system.prime(
        resistance[["X_D", "u"]], U=resistance["u"]
    ).fillna(0)

    f_X_H = sp.Function("X_H")(u, v, r, delta)
    f_X_P = sp.Function("X_P")(u, v, r, rev)
    f_X_R = sp.Function("X_R")(u, v, r, delta, thrust)

    eq_X_H = sp.Eq(X_H, sp.solve(model_base.X_D_eq, f_X_H)[0]).subs(
        [
            (f_X_P, 0),
            (f_X_R, 0),
        ]
    )
    lambda_X_H = lambdify(eq_X_H.rhs, substitute_functions=True)

    # Precalculate the rudder and wind_force:
    resistance_calculation = {}
    for system_name, system in model_base.subsystems.items():
        try:
            system.calculate_forces(
                states_dict=resistance[model_base.states_str],
                control=resistance[model_base.control_keys],
                calculation=resistance_calculation,
            )
        except KeyError as e:
            continue  # One of the system that should be regressed (no parameters yet)

    df_resistance_calculation = pd.DataFrame(
        resistance_calculation, index=resistance.index
    )
    columns = list(set(resistance.columns) - set(df_resistance_calculation.columns))
    resistance = pd.concat((resistance[columns], df_resistance_calculation), axis=1)
    resistance["X_H"] = run(lambda_X_H, resistance)

    eq = (
        model_base.subsystems["hull"]
        .equations["X_H"]
        .subs(
            [
                (delta, 0),
                (v, 0),
                (r, 0),
            ]
        )
    )

    eq_to_matrix_X_resistance_regression = DiffEqToMatrix(
        eq,
        label=X_H,
        base_features=[u, v, r, thrust, delta],
        exclude_parameters={"X0": 0},
    )

    df_resistance_prime = model_base.prime_system.prime(
        resistance[["X_H", "u"]], U=resistance["u"]
    ).fillna(0)

    X, y = eq_to_matrix_X_resistance_regression.calculate_features_and_label(
        data=df_resistance_prime, y=df_resistance_prime["X_H"]
    )
    ols = sm.OLS(y, X)
    ols_fit = ols.fit()
    return ols_fit


def inverse_dynamics_hull_rudder(
    model_base: ModularVesselSimulator, data: pd.DataFrame, exclude_parameters={}
) -> ModularVesselSimulator:
    exclude_parameters = exclude_parameters.copy()

    ## Inverse the dynamics:
    data = model_base.forces_from_motions(data=data)
    data["X_D"] = data["fx"]
    data["Y_D"] = data["fy"]
    data["N_D"] = data["mz"]

    ## Regression
    f_X_H = sp.Function("X_H")(u, v, r, delta)
    f_X_R = sp.Function("X_R")(u, v, r, delta, thrust)
    f_Y_H = sp.Function("Y_H")(u, v, r, delta)
    f_Y_R = sp.Function("Y_R")(u, v, r, delta, thrust)
    f_N_H = sp.Function("N_H")(u, v, r, delta)
    f_N_R = sp.Function("N_R")(u, v, r, delta, thrust)

    lhs = f_X_H + f_X_R
    eq_X_regression = sp.Eq(lhs, sp.solve(model_base.X_eq, lhs)[0])

    lhs = f_Y_H + f_Y_R
    eq_Y_regression = sp.Eq(lhs, sp.solve(model_base.Y_eq, lhs)[0])

    lhs = f_N_H + f_N_R
    eq_N_regression = sp.Eq(lhs, sp.solve(model_base.N_eq, lhs)[0])

    eq_X_regression_expanded = eq_X_regression.subs(
        [
            (f_X_H, model_base.subsystems["hull"].equations["X_H"].rhs),
            (f_X_R, model_base.subsystems["rudders"].equations["X_R"].rhs),
        ]
    )

    eq_Y_regression_expanded = eq_Y_regression.subs(
        [
            (f_Y_H, model_base.subsystems["hull"].equations["Y_H"].rhs),
            (f_Y_R, model_base.subsystems["rudders"].equations["Y_R"].rhs),
        ]
    )

    eq_N_regression_expanded = eq_N_regression.subs(
        [
            (f_N_H, model_base.subsystems["hull"].equations["N_H"].rhs),
            (f_N_R, model_base.subsystems["rudders"].equations["N_R"].rhs),
        ]
    )

    return inverse_dynamics(
        model_base=model_base,
        data=data,
        eq_X_regression_expanded=eq_X_regression_expanded,
        eq_Y_regression_expanded=eq_Y_regression_expanded,
        eq_N_regression_expanded=eq_N_regression_expanded,
        exclude_parameters=exclude_parameters,
    )


def inverse_dynamics_hull(
    model_base: ModularVesselSimulator, data: pd.DataFrame, exclude_parameters={}
) -> ModularVesselSimulator:
    exclude_parameters = exclude_parameters.copy()

    ## Inverse the dynamics:
    data = model_base.forces_from_motions(data=data)
    data["X_D"] = data["fx"]
    data["Y_D"] = data["fy"]
    data["N_D"] = data["mz"]

    ## Regression
    f_X_H = sp.Function("X_H")(u, v, r, delta)
    f_Y_H = sp.Function("Y_H")(u, v, r, delta)
    f_N_H = sp.Function("N_H")(u, v, r, delta)

    lhs = f_X_H
    eq_X_regression = sp.Eq(lhs, sp.solve(model_base.X_eq, lhs)[0])

    lhs = f_Y_H
    eq_Y_regression = sp.Eq(lhs, sp.solve(model_base.Y_eq, lhs)[0])

    lhs = f_N_H
    eq_N_regression = sp.Eq(lhs, sp.solve(model_base.N_eq, lhs)[0])

    eq_X_regression_expanded = eq_X_regression.subs(
        [
            (f_X_H, model_base.subsystems["hull"].equations["X_H"].rhs),
        ]
    )

    eq_Y_regression_expanded = eq_Y_regression.subs(
        [
            (f_Y_H, model_base.subsystems["hull"].equations["Y_H"].rhs),
        ]
    )

    eq_N_regression_expanded = eq_N_regression.subs(
        [
            (f_N_H, model_base.subsystems["hull"].equations["N_H"].rhs),
        ]
    )

    return inverse_dynamics(
        model_base=model_base,
        data=data,
        eq_X_regression_expanded=eq_X_regression_expanded,
        eq_Y_regression_expanded=eq_Y_regression_expanded,
        eq_N_regression_expanded=eq_N_regression_expanded,
        exclude_parameters=exclude_parameters,
    )


def inverse_dynamics_hull_wind(
    model_base: ModularVesselSimulator, data: pd.DataFrame, exclude_parameters={}
) -> ModularVesselSimulator:
    exclude_parameters = exclude_parameters.copy()

    ## Inverse the dynamics:
    data = model_base.forces_from_motions(data=data)
    data["X_D"] = data["fx"]
    data["Y_D"] = data["fy"]
    data["N_D"] = data["mz"]

    ## Regression
    functions_X_eq = find_functions(model_base.X_eq)
    functions_Y_eq = find_functions(model_base.Y_eq)
    functions_N_eq = find_functions(model_base.N_eq)

    f_X_H = functions_X_eq["X_H"]
    f_Y_H = functions_Y_eq["Y_H"]
    f_N_H = functions_N_eq["N_H"]

    wind_force = model_base.subsystems["wind_force"]
    f_X_W = functions_X_eq["X_W"]
    f_Y_W = functions_Y_eq["Y_W"]
    f_N_W = functions_N_eq["N_W"]

    lhs = f_X_H + f_X_W
    eq_X_regression = sp.Eq(lhs, sp.solve(model_base.X_eq, lhs)[0])

    lhs = f_Y_H + f_Y_W
    eq_Y_regression = sp.Eq(lhs, sp.solve(model_base.Y_eq, lhs)[0])

    lhs = f_N_H + f_N_W
    eq_N_regression = sp.Eq(lhs, sp.solve(model_base.N_eq, lhs)[0])

    eq_X_regression_expanded = eq_X_regression.subs(
        [
            (f_X_H, model_base.subsystems["hull"].equations["X_H"].rhs),
            (f_X_W, wind_force.equations["X_W"].rhs),
        ]
    )

    eq_Y_regression_expanded = eq_Y_regression.subs(
        [
            (f_Y_H, model_base.subsystems["hull"].equations["Y_H"].rhs),
            (f_Y_W, wind_force.equations["Y_W"].rhs),
        ]
    )

    eq_N_regression_expanded = eq_N_regression.subs(
        [
            (f_N_H, model_base.subsystems["hull"].equations["N_H"].rhs),
            (f_N_W, wind_force.equations["N_W"].rhs),
        ]
    )

    return inverse_dynamics(
        model_base=model_base,
        data=data,
        eq_X_regression_expanded=eq_X_regression_expanded,
        eq_Y_regression_expanded=eq_Y_regression_expanded,
        eq_N_regression_expanded=eq_N_regression_expanded,
        exclude_parameters=exclude_parameters,
    )


def inverse_dynamics(
    model_base: ModularVesselSimulator,
    data: pd.DataFrame,
    eq_X_regression_expanded: sp.Eq,
    eq_Y_regression_expanded: sp.Eq,
    eq_N_regression_expanded: sp.Eq,
    exclude_parameters={},
) -> ModularVesselSimulator:
    X_regression, Y_regression, N_regression = sp.symbols(
        "X_regression,Y_regression,N_regression"
    )
    eq_X_regression_expanded_subs = sp.Eq(X_regression, eq_X_regression_expanded.rhs)
    eq_Y_regression_expanded_subs = sp.Eq(Y_regression, eq_Y_regression_expanded.rhs)
    eq_N_regression_expanded_subs = sp.Eq(N_regression, eq_N_regression_expanded.rhs)

    eqs = [
        eq_X_regression_expanded_subs,
        eq_Y_regression_expanded_subs,
        eq_N_regression_expanded_subs,
    ]
    lambdas = {
        eq.lhs.name: lambdify(eq.rhs.subs(subs_simpler), substitute_functions=True)
        for eq in eqs
    }

    model = model_base.copy()

    # Precalculate the propellers and wind_force:
    calculation = {}
    for system_name, system in model.subsystems.items():
        try:
            system.calculate_forces(
                states_dict=data[model.states_str],
                control=data[model.control_keys],
                calculation=calculation,
            )
        except KeyError as e:
            continue  # One of the system that should be regressed (no parameters yet)
        except ValueError as e:
            continue  # One of the system that should be regressed (no parameters yet)

    df_calculation = pd.DataFrame(calculation, index=data.index)
    columns = list(set(data.columns) - set(df_calculation.columns))
    data_ = pd.concat((data[columns], df_calculation), axis=1)

    data_u0 = data_.copy()
    # U0_ = float(data_["u"].min())  # Check why this is not working...
    U0_ = 0

    data_u0["u"] -= U0_

    for subsystem_name, subsystem in model.subsystems.items():
        subsystem.U0 = U0_

    columns = list(
        set(
            model.states_str
            + model.control_keys
            + ["u1d", "v1d", "r1d"]
            + list(df_calculation.columns)
        )
    )

    data_prime = model.prime_system.prime(
        data_u0[columns],
        U=data["U"],
    )

    # Calculating the labels (the y in the regression)
    # y = X*beta + epsilon
    for key, lambda_ in lambdas.items():
        data_prime[key] = run(
            lambda_,
            inputs=data_prime,
            **model.ship_parameters_prime,
            **model.parameters,
        )

    ## Fitting:
    exclude_parameters["Xthrust"] = model_base.parameters["Xthrust"]

    eq = sp.Eq(X_regression, eq_X_regression_expanded.lhs)
    X_WC, Y_WC, N_WC = sp.symbols("X_WC, Y_WC, N_WC")
    eq_to_matrix_X_regression = DiffEqToMatrix(
        eq,
        label=X_regression,
        base_features=[u, v, r, thrust, delta, X_WC],
        exclude_parameters=exclude_parameters,
    )

    eq = sp.Eq(Y_regression, eq_Y_regression_expanded.lhs)
    eq_to_matrix_Y_regression = DiffEqToMatrix(
        eq,
        label=Y_regression,
        base_features=[u, v, r, thrust, delta, Y_WC],
        exclude_parameters=exclude_parameters,
    )

    eq = sp.Eq(N_regression, eq_N_regression_expanded.lhs)
    eq_to_matrix_N_regression = DiffEqToMatrix(
        eq,
        label=N_regression,
        base_features=[u, v, r, thrust, delta, N_WC],
        exclude_parameters=exclude_parameters,
    )

    ols_fits = {}
    new_parameters = exclude_parameters.copy()
    for eq_to_matrix in [
        eq_to_matrix_X_regression,
        eq_to_matrix_Y_regression,
        eq_to_matrix_N_regression,
    ]:
        key = eq_to_matrix.acceleration_equation.lhs.name
        log.info(f"Regressing:{key}")
        X, y = eq_to_matrix.calculate_features_and_label(
            data=data_prime, y=data_prime[key]
        )
        ols = sm.OLS(y, X)
        ols_fits[key] = ols_fit = ols.fit()
        ols_fit.X = X
        ols_fit.y = y
        new_parameters.update(ols_fit.params)

    model.parameters.update(new_parameters)
    return model, ols_fits
