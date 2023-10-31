from vessel_manoeuvring_models.models.modular_simulator import ModularVesselSimulator
from vessel_manoeuvring_models.models import semiempirical_rudder_MAK

from phd.pipelines.models.subsystems import add_propeller_simple

import numpy as np
import sympy as sp
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

from vessel_manoeuvring_models.models.modular_simulator import subs_simpler
from vessel_manoeuvring_models.models.propeller import (
    PropellerSimpleSystem,
)

import logging

log = logging.getLogger(__name__)


def model(
    ship_data: dict,
    model: ModularVesselSimulator,
    create_jacobians=False,
    in_propeller_race=True,
):
    # model = main_model()
    model.set_ship_parameters(ship_data)
    if not "Xthrust" in model.parameters:
        model.parameters["Xthrust"] = 1 - model.ship_parameters["tdf"]

    if not "gamma_0" in model.parameters:
        model.parameters["gamma_0"] = 0

    if not "'C_L_tune'" in model.parameters:
        model.parameters["C_L_tune"] = 1

    if not "'C_D_tune'" in model.parameters:
        model.parameters["C_D_tune"] = 1

    if not "'C_D0_tune'" in model.parameters:
        model.parameters["C_D0_tune"] = 1

    propeller = PropellerSimpleSystem(ship=model, create_jacobians=create_jacobians)
    model.subsystems["propeller"] = propeller

    ## Add rudder:
    if in_propeller_race:
        propeller_race = semiempirical_rudder_MAK.PropellerRace(
            ship=model,
            create_jacobians=create_jacobians,
            suffix="",
        )
        model.subsystems["propeller_race"] = propeller_race
    else:
        wake = semiempirical_rudder_MAK.Wake(
            ship=model,
            create_jacobians=create_jacobians,
        )
        model.subsystems["wake"] = wake

    rudder = semiempirical_rudder_MAK.SemiempiricalRudderSystemMAK(
        ship=model,
        create_jacobians=create_jacobians,
        in_propeller_race=in_propeller_race,
        suffix="",
    )
    model.subsystems["rudder"] = rudder

    rudder_particulars = {
        "x_R": model.ship_parameters["x_r"],
        "y_R": 0,
        "z_R": 0,
        "w_f": model.ship_parameters["w_p0"],
        "r_0": model.ship_parameters["D"] / 2,
    }
    model.ship_parameters.update(rudder_particulars)
    if not "x" in model.ship_parameters:
        model.ship_parameters["x"] = ship_data["x_p"] - ship_data["x_r"]
        assert (
            model.ship_parameters["x"] > 0
        ), "The rudder is in front of the propeller!"

    rudder_parameters = {
        "nu": 1.18849e-06,
        "e_0": 0.9,
        "kappa": 0.85,
        "l_R": model.ship_parameters["x_r"],
        "delta_lim": np.deg2rad(70),
        "delta_alpha_s": 0
        if in_propeller_race
        else np.deg2rad(
            -5
        ),  # Delayed stall in propeller race (-5 if not propeller race)
        "Omega": 0,
    }
    for key, value in rudder_parameters.items():
        if not key in model.parameters:
            model.parameters[key] = value

    return model


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

    # eq_X_force = fx_eq.subs(X_D, f_X_H + f_X_R + f_X_P + f_X_W)
    # eq_Y_force = fy_eq.subs(Y_D, f_Y_H + f_Y_R + f_Y_P + f_Y_W)
    # eq_N_force = mz_eq.subs(N_D, f_N_H + f_N_R + f_N_P + f_N_W)

    eq_X_force = fx_eq.subs(X_D, f_X_R + f_X_P)
    eq_Y_force = fy_eq.subs(Y_D, f_Y_R + f_Y_P)
    eq_N_force = mz_eq.subs(N_D, f_N_R + f_N_P)

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
        control_keys=["delta", "thrust"],
        do_create_jacobian=False,
    )
    return main_model
