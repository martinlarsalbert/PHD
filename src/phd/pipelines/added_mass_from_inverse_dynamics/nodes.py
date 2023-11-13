"""
This is a boilerplate pipeline 'added_mass_from_inverse_dynamics'
generated using Kedro 0.18.14
"""
import pandas as pd
from vessel_manoeuvring_models.models.modular_simulator import ModularVesselSimulator
from vessel_manoeuvring_models.prime_system import PrimeSystem
import numpy as np
from phd.visualization.plot_prediction import (
    predict,
)
from vessel_manoeuvring_models.substitute_dynamic_symbols import run, lambdify
from vessel_manoeuvring_models.symbols import *
import statsmodels.api as sm
import logging

log = logging.getLogger(__name__)

from vessel_manoeuvring_models.parameters import df_parameters

p = df_parameters["symbol"]


def regress(tests_ek_smooth_joined: dict, models: {}) -> ModularVesselSimulator:
    log.info("Added mass from inverse dynamics")

    new_models = {}

    for name, loader in models.items():
        log.info(f"Added mass for: {name}")
        model = loader()
        new_models[name] = _regress(
            tests_ek_smooth_joined=tests_ek_smooth_joined, model=model
        )

    return new_models


def _regress(
    tests_ek_smooth_joined: dict, model: ModularVesselSimulator
) -> ModularVesselSimulator:
    log.info("Nvdot and Yrdot are assumed to be zero.")
    model.parameters["Nvdot"] = 0
    model.parameters["Yrdot"] = 0

    log.info("regressing Nrdot")
    model = regress_Nr_dot(tests_ek_smooth_joined=tests_ek_smooth_joined, model=model)
    log.info("regressing Yvdot")
    model = regress_Yv_dot(tests_ek_smooth_joined=tests_ek_smooth_joined, model=model)

    return model


def gather_data(tests_ek_smooth_joined: dict) -> pd.DataFrame:
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

    return data


def regress_Nr_dot(
    tests_ek_smooth_joined: dict, model: ModularVesselSimulator
) -> ModularVesselSimulator:
    data = gather_data(tests_ek_smooth_joined=tests_ek_smooth_joined)
    prediction = predict(model=model, data=data)

    mask = prediction["r1d"].abs() > prediction["r1d"].abs().quantile(0.3)
    prediction_ = prediction.loc[mask]

    eq = model.N_eq
    move = p.Nrdot * r1d
    eq_added_masses = sp.Eq(sp.solve(eq, move)[0], move)
    eq_added_masses

    eq_ = eq_added_masses.subs(
        [
            (v1d, "v1d"),
            (r1d, "r1d"),
            (p.Nvdot, "Nvdot"),
        ]
    )
    lambda_y = lambdify(eq_.lhs, substitute_functions=True)

    y = run(lambda_y, inputs=prediction_, Nvdot=0, **model.ship_parameters)
    X = prediction_[["r1d"]]

    ols_model = sm.OLS(y, X)
    fit = ols_model.fit()
    log.info(fit.summary2())

    Nrdot_denominator = run(
        lambdify(df_parameters.loc["Nrdot", "denominator"]),
        inputs=model.ship_parameters,
    )

    Nrdot_new = fit.params["r1d"] / Nrdot_denominator
    change = Nrdot_new / model.parameters["Nrdot"]
    if change > 0:
        log.info("New Nrdot has the correct sign - added mass is updated")
        log.info(f"The New Nrdot is {np.round(change,1)} times the old one")
        model.parameters["Nrdot"] = Nrdot_new
    else:
        log.info("New Nrdot has the WRONG sign - added mass is NOT updated!")

    return model


def regress_Yv_dot(
    tests_ek_smooth_joined: dict, model: ModularVesselSimulator
) -> ModularVesselSimulator:
    data = gather_data(tests_ek_smooth_joined=tests_ek_smooth_joined)
    prediction = predict(model=model, data=data)

    mask = prediction["v1d"].abs() > prediction["v1d"].abs().quantile(0.3)
    prediction_ = prediction.loc[mask]

    eq = model.Y_eq
    move = p.Yvdot * v1d
    eq_added_masses = sp.Eq(sp.solve(eq, move)[0], move)
    eq_added_masses

    eq_ = eq_added_masses.subs(
        [
            (v1d, "v1d"),
            (r1d, "r1d"),
            (p.Yrdot, "Yrdot"),
        ]
    )
    lambda_y = lambdify(eq_.lhs, substitute_functions=True)

    y = run(lambda_y, inputs=prediction_, Yrdot=0, **model.ship_parameters)
    X = prediction_[["v1d"]]

    ols_model = sm.OLS(y, X)
    fit = ols_model.fit()
    log.info(fit.summary2())

    Yvdot_denominator = run(
        lambdify(df_parameters.loc["Yvdot", "denominator"]),
        inputs=model.ship_parameters,
    )

    Yvdot_new = fit.params["v1d"] / Yvdot_denominator
    change = Yvdot_new / model.parameters["Yvdot"]
    if change > 0:
        log.info("New Yvdot has the correct sign - added mass is updated")
        log.info(f"The New Yvdot is {np.round(change,1)} times the old one")
        model.parameters["Yvdot"] = Yvdot_new
    else:
        log.info("New Yvdot has the WRONG sign - added mass is NOT updated!")

    return model
