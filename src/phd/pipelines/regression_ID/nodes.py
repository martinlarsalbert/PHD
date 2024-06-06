"""
This is a boilerplate pipeline 'regression_ID'
generated using Kedro 0.18.7
"""

from .inverse_dynamics import (
    fit_resistance,
    inverse_dynamics_hull_rudder,
    inverse_dynamics_hull,
)

import pandas as pd
from vessel_manoeuvring_models.symbols import *
from vessel_manoeuvring_models.models.modular_simulator import (
    ModularVesselSimulator,
)

from vessel_manoeuvring_models.substitute_dynamic_symbols import run
from vessel_manoeuvring_models.models.diff_eq_to_matrix import DiffEqToMatrix
import statsmodels.api as sm
from vessel_manoeuvring_models.models.modular_simulator import subs_simpler

import pandas as pd
import numpy as np
from vessel_manoeuvring_models.models.modular_simulator import ModularVesselSimulator
import sympy as sp
import logging
import statsmodels.api as sm
from vessel_manoeuvring_models.models.diff_eq_to_matrix import DiffEqToMatrix
from pyfiglet import figlet_format

log = logging.getLogger(__name__)

exclude_parameters_global = {
    # "a_H": 0.07,  # hull rudder interaction,
    "Yrrr": 0,
    "Nvvv": 0,
    "Nvvr": 0,
    "Nvrr": 0,
    "Yvvr": 0,
    "Yvrr": 0,
    "Yr":0,
    "Yrrr":0,
    "Yvvv":0,
    "Y0": 0,
    "N0": 0,
    "Nrrr": 0,
}


def gather_data(tests_ek_smooth_joined: pd.DataFrame) -> pd.DataFrame:
    ids = [
        22773,
        22772,
        22770,
        22765,
    ]

    mask = tests_ek_smooth_joined["id"].isin(ids)
    data = tests_ek_smooth_joined.loc[mask].copy()
    assert set(ids) == set(data['id'].unique()), f"One of the required tests ({ids}) is missing"    
    
    log.info(f"Training model with tests: {data['id'].unique()}")

    data["V"] = data["U"] = np.sqrt(data["u"] ** 2 + data["v"] ** 2)
    data["beta"] = -np.arctan2(data["v"], data["u"])
    if "Prop/PS/Rpm" in data:
        data["rev"] = data[["Prop/PS/Rpm", "Prop/SB/Rpm"]].mean(axis=1)
    
    data["twa"] = 0
    data["tws"] = 0

    if not "thrust_port" in data:
        data["thrust_port"] = data["Prop/PS/Thrust"]
    
    if not "thrust_stbd" in data:
        data["thrust_stbd"] = data["Prop/SB/Thrust"]
        
    data['thrust'] = data["thrust_port"] + data["thrust_stbd"]

    return data


def regress_hull_inverse_dynamics(
    base_models: dict,
    tests_ek_smooth_joined: pd.DataFrame,
) -> dict:
    models = {}

    log.info(figlet_format("Semi-empirical ID", font="starwars"))

    data = gather_data(tests_ek_smooth_joined=tests_ek_smooth_joined)

    for name, loader in base_models.items():
        base_model = loader()
        exclude_parameters = exclude_parameters_global.copy()

        ## Stealing the resistance
        exclude_parameters["X0"] = base_model.parameters["X0"]
        exclude_parameters["Xu"] = base_model.parameters["Xu"]

        base_model.parameters["Xthrust"] = 1 - base_model.ship_parameters["tdf"]

        models[name], fits = _regress_hull_inverse_dynamics(
            vmm_model=base_model,
            data=data,
            exclude_parameters=exclude_parameters,
            full_output=True,
        )
        
        for key, fit in fits.items():
            log.info(key)
            log.info(fit.summary2())

    return models


def _regress_hull_inverse_dynamics(
    vmm_model: ModularVesselSimulator,
    data: pd.DataFrame,
    exclude_parameters: dict = {},
    full_output=False,
) -> ModularVesselSimulator:
    data = data.copy()
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

    if not full_output:
        log.info(sp.pprint(eq_f_X_H))
        log.info(sp.pprint(eq_f_Y_H))
        log.info(sp.pprint(eq_f_N_H))

    data["V"] = data["U"] = np.sqrt(data["u"] ** 2 + data["v"] ** 2)
    if "Prop/SB/Rpm" in data and "Prop/PS/Rpm" in data:
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

    #data["beta"] = -np.arctan2(data["v"], data["u"])

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
        except Exception as e:
            raise Exception(f"Failed in subsystem:{system_name}")

    df_calculation = pd.DataFrame(calculation, index=data.index)
    if "beta" in df_calculation:
        df_calculation.drop(columns="beta",inplace=True)
    
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
        if not full_output:
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
    base_models: dict, tests_ek_smooth_joined: pd.DataFrame, steal_models: dict
) -> dict:
    models = {}

    log.info(figlet_format("ID", font="starwars"))

    data = gather_data(tests_ek_smooth_joined=tests_ek_smooth_joined)

    steal_model = steal_models["semiempirical_covered"]()

    for name, loader in base_models.items():
        log.info(figlet_format(f"{name}", font="starwars"))
        base_model = loader()
        exclude_parameters = exclude_parameters_global.copy()

        ## Stealing the resistance
        exclude_parameters["X0"] = steal_model.parameters["X0"]
        exclude_parameters["Xu"] = steal_model.parameters["Xu"]

        base_model.parameters["Xthrust"] = 1 - base_model.ship_parameters["tdf"]
        steals = ["Nrdot", "Nvdot", "Yrdot", "Yvdot"]
        for steal in steals:
            base_model.parameters[steal] = steal_model.parameters[steal]

        models[name], fits = _regress_inverse_dynamics(
            vmm_model=base_model,
            data=data,
            exclude_parameters=exclude_parameters,
            full_output=True,
        )

    return models


def _regress_inverse_dynamics(
    vmm_model: ModularVesselSimulator,
    data: pd.DataFrame,
    exclude_parameters={},
    full_output=False,
) -> ModularVesselSimulator:
    data = data.copy()
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
        if not full_output:
            log.info(f"Regressing:{key}")
        
        try:
            X, y = eq_to_matrix.calculate_features_and_label(
                data=data_prime,
                y=data_prime[key],
                parameters=model.ship_parameters,
            )
        except Exception as e:
            raise ValueError(f"Failed on equation:{eq_to_matrix.acceleration_equation}")
            
        ols = sm.OLS(y, X)
        ols_fit = ols.fit()
        ols_fit.X=X
        ols_fit.y=y
        fits[key] = ols_fit
        new_parameters.update(ols_fit.params)
        if not full_output:
            log.info(ols_fit.summary().as_text())

    model.parameters.update(new_parameters)
    model.parameters.update(exclude_parameters)

    if full_output:
        return model, fits
    else:
        return model
