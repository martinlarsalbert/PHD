from vessel_manoeuvring_models.models.wind_force import (
    WindForceSystem,
    DummyWindForceSystem,
    WindForceSystemSimple,
)
from vessel_manoeuvring_models.models.propeller import (
    PropellersSystem,
    PropellerSystem,
    PropellersSimpleSystem,
    PropellerSimpleSystem
)
from vessel_manoeuvring_models.models.modular_simulator import ModularVesselSimulator
from vessel_manoeuvring_models.models.semiempirical_rudder import (
    SemiempiricalRudderSystem,
    SemiempiricalRudderWithoutPropellerInducedSpeedSystem,
)
from vessel_manoeuvring_models.models import semiempirical_rudder_MAK
from vessel_manoeuvring_models.models.subsystem import PrimeEquationSubSystem
import numpy as np


def add_propeller(model: ModularVesselSimulator, create_jacobians=True):
    ## Add propeller:
    propeller_port = PropellerSystem(
        ship=model, create_jacobians=create_jacobians, suffix="port"
    )
    propeller_stbd = PropellerSystem(
        ship=model, create_jacobians=create_jacobians, suffix="stbd"
    )

    propellers = PropellersSystem(ship=model, create_jacobians=create_jacobians)

    if "propellers" in model.subsystems:
        model.subsystems.pop("propellers")

    # Put the propellers first among the models subsytems:
    new_subsystems = {
        "propeller_port": propeller_port,
        "propeller_stbd": propeller_stbd,
        "propellers": propellers,
    }
    new_subsystems.update(model.subsystems)
    model.subsystems = new_subsystems

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


def add_propellers_simple(model: ModularVesselSimulator, create_jacobians=True):
    ## Add propeller:
    propellers = PropellersSimpleSystem(ship=model, create_jacobians=create_jacobians)
    model.subsystems["propellers"] = propellers

def add_propeller_simple(model: ModularVesselSimulator, create_jacobians=True):
    ## Add propeller:
    propellers = PropellerSimpleSystem(ship=model, create_jacobians=create_jacobians)
    model.subsystems["propellers"] = propellers

def add_rudder(model: ModularVesselSimulator, create_jacobians=True):
    ## Add rudder:
    rudders = SemiempiricalRudderSystem(ship=model, create_jacobians=create_jacobians)
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
        "nu": 1.18849e-06,
    }
    model.parameters.update(rudder_parameters)
    model.parameters.update(rudder_parameters)
    if not "kappa" in model.parameters:
        model.parameters["kappa"] = (
            0.85,
        )  # (Small value means much flow straightening)

    if not "l_R" in model.parameters:
        model.parameters["l_R"] = model.ship_parameters["x_r"]


def add_rudder_without_propeller_induced_speed(model: ModularVesselSimulator):
    ## Add rudder:
    rudders = SemiempiricalRudderWithoutPropellerInducedSpeedSystem(
        ship=model, create_jacobians=True
    )
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
        "nu": 1.18849e-06,
    }
    model.parameters.update(rudder_parameters)
    if not "kappa" in model.parameters:
        model.parameters["kappa"] = (
            0.85,
        )  # (Small value means much flow straightening)

    if not "l_R" in model.parameters:
        model.parameters["l_R"] = model.parameters["x_r"]


def add_rudder_simplest(model: ModularVesselSimulator):
    from .vmm_simple import (
        eq_X_R,
        eq_Y_R,
        eq_N_R,
    )

    ## Add rudder:
    equations_rudders = [eq_X_R, eq_Y_R, eq_N_R]
    rudders = PrimeEquationSubSystem(
        ship=model, equations=equations_rudders, create_jacobians=True
    )
    model.subsystems["rudders"] = rudders


def add_rudder_simple(model: ModularVesselSimulator):
    from .vmm_simple import (
        eq_X_R_thrust,
        eq_Y_R_thrust,
        eq_N_R_thrust,
    )

    ## Add rudder:
    equations_rudders = [eq_X_R_thrust, eq_Y_R_thrust, eq_N_R_thrust]
    rudders = PrimeEquationSubSystem(
        ship=model, equations=equations_rudders, create_jacobians=True
    )
    model.subsystems["rudders"] = rudders


def add_rudder_MAK(model: ModularVesselSimulator, create_jacobians=True):
    ## Add rudder:
    propeller_race_port = semiempirical_rudder_MAK.PropellerRace(
        ship=model,
        create_jacobians=create_jacobians,
        suffix="port",
    )

    propeller_race_stbd = semiempirical_rudder_MAK.PropellerRace(
        ship=model,
        create_jacobians=create_jacobians,
        suffix="stbd",
    )

    rudders = semiempirical_rudder_MAK.Rudders(
        ship=model, create_jacobians=create_jacobians
    )

    rudder_port = semiempirical_rudder_MAK.SemiempiricalRudderSystemMAK(
        ship=model,
        create_jacobians=create_jacobians,
        in_propeller_race=True,
        # rudder_eq_renames={X_R: X_R_p, Y_R: Y_R_p, N_R: N_R_p},
        suffix="port",
    )

    rudder_stbd = semiempirical_rudder_MAK.SemiempiricalRudderSystemMAK(
        ship=model,
        create_jacobians=create_jacobians,
        in_propeller_race=True,
        # rudder_eq_renames={X_R: X_R_s, Y_R: Y_R_s, N_R: N_R_s},
        suffix="stbd",
    )

    remove_systems = [
        "rudders",
        "propeller_race_port",
        "propeller_race_stbd",
        "wake",
        "rudder_port",
        "rudder_stbd",
        "rudders",
    ]

    for remove_system in remove_systems:
        if remove_system in model.subsystems:
            model.subsystems.pop(remove_system)

    model.subsystems["propeller_race_port"] = propeller_race_port
    model.subsystems["propeller_race_stbd"] = propeller_race_stbd
    model.subsystems["rudder_port"] = rudder_port
    model.subsystems["rudder_stbd"] = rudder_stbd
    model.subsystems["rudders"] = rudders

    rudder_particulars = {
        "x_R": model.ship_parameters["x_r"],
        "y_R": 0,
        "z_R": 0,
        "w_f": model.ship_parameters["w_p0"],
        "r_0": model.ship_parameters["D"] / 2,
    }
    model.ship_parameters.update(rudder_particulars)
    if not "x" in model.ship_parameters:
        model.ship_parameters["x"] = (
            model.ship_parameters["x_p"] - model.ship_parameters["x_r"]
        )
        assert (
            model.ship_parameters["x"] > 0
        ), "The rudder is in front of the propeller!"

    rudder_parameters = {
        "nu": 1.18849e-06,
        "e_0": 0.9,
        "kappa": 0.85,
        "l_R": model.ship_parameters["x_r"],
        "delta_lim": np.deg2rad(70),
        "delta_alpha_s": 0,  # Delayed stall in propeller race (-10 if not propeller race)
        "Omega": 0,
    }
    for key, value in rudder_parameters.items():
        if not key in model.parameters:
            model.parameters[key] = value


def add_rudder_MAK_no_prop(model: ModularVesselSimulator, create_jacobians=True):
    ## Add rudder:
    wake = semiempirical_rudder_MAK.Wake(ship=model, create_jacobians=create_jacobians)

    rudders = semiempirical_rudder_MAK.Rudders(
        ship=model, create_jacobians=create_jacobians
    )

    rudder_port = semiempirical_rudder_MAK.SemiempiricalRudderSystemMAK(
        ship=model,
        create_jacobians=create_jacobians,
        in_propeller_race=False,
        # rudder_eq_renames={X_R: X_R_p, Y_R: Y_R_p, N_R: N_R_p},
        suffix="port",
    )

    rudder_stbd = semiempirical_rudder_MAK.SemiempiricalRudderSystemMAK(
        ship=model,
        create_jacobians=create_jacobians,
        in_propeller_race=False,
        # rudder_eq_renames={X_R: X_R_s, Y_R: Y_R_s, N_R: N_R_s},
        suffix="stbd",
    )

    if "rudders" in model.subsystems:
        model.subsystems.pop("rudders")

    model.subsystems["wake"] = wake
    model.subsystems["rudder_port"] = rudder_port
    model.subsystems["rudder_stbd"] = rudder_stbd
    model.subsystems["rudders"] = rudders

    rudder_particulars = {
        "x_R": model.ship_parameters["x_r"],
        "y_R": 0,
        "z_R": 0,
        "w_f": model.ship_parameters["w_p0"],
    }
    model.ship_parameters.update(rudder_particulars)
    rudder_parameters = {
        "nu": 1.18849e-06,
        "e_0": 0.9,
        "kappa": 0.85,
        "l_R": model.ship_parameters["x_r"],
        "delta_lim": np.deg2rad(70),
        "delta_alpha_s": np.deg2rad(-5),  # Earlier stall when no propeller race
        "Omega": 0,
    }
    for key, value in rudder_parameters.items():
        if not key in model.parameters:
            model.parameters[key] = value


def add_wind_force_system(model: ModularVesselSimulator, create_jacobians=True):
    ## Add a real wind force system:
    wind_force = WindForceSystem(ship=model, create_jacobians=create_jacobians)
    model.subsystems["wind_force"] = wind_force
    if not "twa" in model.control_keys:
        model.control_keys.append("twa")  # NOTE!

    if not "tws" in model.control_keys:
        model.control_keys.append("tws")  # NOTE!

    return model


def add_wind_force_system_simple(model: ModularVesselSimulator,create_jacobians=True):
    ## Add a real wind force system:
    wind_force = WindForceSystemSimple(ship=model, create_jacobians=create_jacobians)
    model.subsystems["wind_force"] = wind_force
    if not "twa" in model.control_keys:
        model.control_keys.append("twa")  # NOTE!

    if not "tws" in model.control_keys:
        model.control_keys.append("tws")  # NOTE!

    model.parameters["CX"] = 1
    model.parameters["CY"] = 1
    model.parameters["CN"] = 1

    return model


def add_dummy_wind_force_system(model: ModularVesselSimulator, create_jacobians=True):
    ## Add dummy wind system:
    wind_force = DummyWindForceSystem(ship=model, create_jacobians=create_jacobians)
    model.subsystems["wind_force"] = wind_force
