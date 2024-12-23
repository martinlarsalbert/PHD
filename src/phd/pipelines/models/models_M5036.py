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
from .subsystems import (
    add_propeller,
    add_propeller_simple,
    add_rudder,
    add_rudder_MAK,
    add_rudder_MAK_no_prop,
    add_rudder_simple,
    add_dummy_wind_force_system,
    add_wind_force_system_simple,
)
from vessel_manoeuvring_models.models.semiempirical_rudder import (
    RudderHullInteractionSystem,
    RudderHullInteractionDummySystem,
)
from vessel_manoeuvring_models.models import semiempirical_rudder_MAK
from vessel_manoeuvring_models.models.semiempirical_covered_system import (
    SemiempiricalRudderSystemCovered,
)
from vessel_manoeuvring_models.models.abkowitz_rudder_system import AbkowitzRudderSystem
from vessel_manoeuvring_models.prime_system import PrimeSystem
from vessel_manoeuvring_models.models.propeller import (
    PropellerSimpleSystem,
)

import logging

log = logging.getLogger(__name__)


class MainModel(ModularVesselSimulator):
    def __init__(self, create_jacobians=True):
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

        eq_X_force = fx_eq.subs(X_D, f_X_H + f_X_R + f_X_P + f_X_W + X_RHI)
        eq_Y_force = fy_eq.subs(Y_D, f_Y_H + f_Y_R + f_Y_P + f_Y_W + Y_RHI)
        eq_N_force = mz_eq.subs(N_D, f_N_H + f_N_R + f_N_P + f_N_W + N_RHI)

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

        super().__init__(
            X_eq=X_eq,
            Y_eq=Y_eq,
            N_eq=N_eq,
            ship_parameters={},
            parameters={},
            control_keys=["delta", "thrust"],
            do_create_jacobian=create_jacobians,
        )
        self.create_jacobians = create_jacobians


class ModelTowed(MainModel):
    def __init__(self, ship_data: dict, create_jacobians=True):
        super().__init__(create_jacobians=create_jacobians)

        self.setup_parameters(ship_data=ship_data)
        self.setup_subsystems()
        self.add_added_mass()

    def setup_subsystems(self):
        ## Add hull:
        self.add_hull()

        ## Add propeller:
        self.add_propellers()

        ## Add wake:
        wake = semiempirical_rudder_MAK.Wake(
            ship=self, create_jacobians=self.create_jacobians
        )
        self.subsystems["wake"] = wake

        ## Add rudders:
        self.add_rudders(in_propeller_race=False)

        ## Add dummy wind system:
        add_dummy_wind_force_system(model=self)

        self.control_keys = ["delta", "thrust_port", "thrust_stbd"]

    def setup_parameters(self, ship_data: dict):
        ship_data = ship_data.copy()
        ship_data["r_0"] = ship_data["D"] / 2
        ship_data["x"] = ship_data["x_p"] - ship_data["x_r"]
        self.set_ship_parameters(ship_data)

        if not "Xthrust" in self.parameters:
            self.parameters["Xthrust"] = 1 - self.ship_parameters["tdf"]

        if not "gamma_0" in self.parameters:
            self.parameters["gamma_0"] = 0

        if not "'C_L_tune'" in self.parameters:
            self.parameters["C_L_tune"] = 1

        if not "'C_D_tune'" in self.parameters:
            self.parameters["C_D_tune"] = 1

        if not "'C_D0_tune'" in self.parameters:
            self.parameters["C_D0_tune"] = 1

    def add_added_mass(self):
        mask = df_parameters["state"] == "dot"
        lambdas_added_mass = df_parameters.loc[mask, "brix_lambda"].dropna()
        added_masses = {
            key: run(lambda_, **self.ship_parameters)
            for key, lambda_ in lambdas_added_mass.items()
        }
        for key, value in added_masses.items():
            if not key in self.parameters:
                self.parameters[key] = value

    def add_hull(self):
        self.subsystems["hull"] = hull(
            model=self, create_jacobians=self.create_jacobians
        )

    def add_propellers(self):
        propeller = PropellerSimpleSystem(
            ship=self, create_jacobians=self.create_jacobians
        )
        self.subsystems["propeller"] = propeller

    def add_rudders(self, in_propeller_race: bool):
        rudder_port = semiempirical_rudder_MAK.SemiempiricalRudderSystemMAK(
            ship=self,
            create_jacobians=self.create_jacobians,
            in_propeller_race=in_propeller_race,
            suffix="port",
        )
        self.subsystems["rudder_port"] = rudder_port

        rudder_stbd = semiempirical_rudder_MAK.SemiempiricalRudderSystemMAK(
            ship=self,
            create_jacobians=self.create_jacobians,
            in_propeller_race=in_propeller_race,
            suffix="stbd",
        )
        self.subsystems["rudder_stbd"] = rudder_stbd

        ## Add rudders system (joining the rudder forces)
        rudders = semiempirical_rudder_MAK.Rudders(
            ship=self, create_jacobians=self.create_jacobians
        )
        self.subsystems["rudders"] = rudders

        ## Default parameters:
        rudder_particulars = {
            "x_R": self.ship_parameters["x_r"],
            "y_R": 0,
            "z_R": 0,
            "w_f": self.ship_parameters["w_p0"],
        }
        self.ship_parameters.update(rudder_particulars)
        rudder_parameters = {
            "nu": 1.18849e-06,
            "e_0": 0.9,
            "kappa": 0.85,
            "l_R": self.ship_parameters["x_r"],
            "delta_lim": np.deg2rad(70),
            "delta_alpha_s": np.deg2rad(-5),  # Earlier stall when no propeller race
            "Omega": 0,
        }
        for key, value in rudder_parameters.items():
            if not key in self.parameters:
                self.parameters[key] = value

        ## Add rudder hull interaction subsystem:
        self.subsystems["rudder_hull_interaction"] = RudderHullInteractionSystem(
            ship=self, create_jacobians=self.create_jacobians
        )


class ModelTowedSemiempiricalCovered(ModelTowed):
    def setup_subsystems(self):
        ## Add hull:
        self.add_hull()

        ## Add propeller:
        self.add_propellers()

        ## Add rudders:
        self.add_rudders(in_propeller_race=True)

        ## Add dummy wind system:
        add_dummy_wind_force_system(model=self)

        self.control_keys = ["delta", "thrust"]

    def add_rudders(self, in_propeller_race: bool):
        rudder = SemiempiricalRudderSystemCovered(
            ship=self,
            create_jacobians=self.create_jacobians,
            in_propeller_race=in_propeller_race,
            suffix="",
        )
        self.subsystems["rudder"] = rudder

        rudder_stbd = SemiempiricalRudderSystemCovered(
            ship=self,
            create_jacobians=self.create_jacobians,
            in_propeller_race=in_propeller_race,
            suffix="stbd",
        )
        self.subsystems["rudder_stbd"] = rudder_stbd

        ## Add rudders system (joining the rudder forces)
        rudders = semiempirical_rudder_MAK.Rudders(
            ship=self, create_jacobians=self.create_jacobians
        )
        self.subsystems["rudders"] = rudders

        ## Default parameters:
        rudder_particulars = {
            "x_R": self.ship_parameters["x_r"],
            "y_R": 0,
            "z_R": 0,
            "w_f": self.ship_parameters["w_p0"],
            "A_R_C": 0,  # No propeller to cover the rudder
            "A_R_U": self.ship_parameters["A_R"],  # ... everything is uncovered
        }
        self.ship_parameters.update(rudder_particulars)
        rudder_parameters = {
            "nu": 1.18849e-06,
            "e_0": 0.9,
            "kappa": 0.85,
            "l_R": self.ship_parameters["x_r"],
            "delta_lim": np.deg2rad(70),
            "delta_alpha_s": np.deg2rad(-5),  # Earlier stall when no propeller race
            "Omega": 0,
        }
        for key, value in rudder_parameters.items():
            if not key in self.parameters:
                self.parameters[key] = value

        ## Add rudder hull interaction subsystem:
        self.subsystems["rudder_hull_interaction"] = RudderHullInteractionSystem(
            ship=self, create_jacobians=self.create_jacobians
        )


class ModelSemiempiricalCovered(ModelTowedSemiempiricalCovered):
    def add_rudders(self, in_propeller_race: bool):
        rudder = SemiempiricalRudderSystemCovered(
            ship=self,
            create_jacobians=self.create_jacobians,
            in_propeller_race=in_propeller_race,
            suffix="",
        )
        self.subsystems["rudder"] = rudder

        ## Add rudders system (joining the rudder forces)
        # rudders = semiempirical_rudder_MAK.Rudders(
        #    ship=self, create_jacobians=self.create_jacobians
        # )
        # self.subsystems["rudders"] = rudders

        ## Default parameters:
        rudder_particulars = {
            "x_R": self.ship_parameters["x_r"],
            "y_R": 0,
            "z_R": 0,
            "w_f": self.ship_parameters["w_p0"],
            "A_R_C": self.ship_parameters["D"] * self.ship_parameters["c"],
            "A_R_U": (self.ship_parameters["b_R"] - self.ship_parameters["D"])
            * self.ship_parameters["c"],
        }
        self.ship_parameters.update(rudder_particulars)
        rudder_parameters = {
            "nu": 1.18849e-06,
            "e_0": 0.9,
            "kappa": 0.85,
            "l_R": self.ship_parameters["x_r"],
            "delta_lim": np.deg2rad(35),
            "delta_alpha_s": np.deg2rad(0),  # Later stall when propeller race
            "Omega": 0,
        }
        for key, value in rudder_parameters.items():
            if not key in self.parameters:
                self.parameters[key] = value

        ## Add rudder hull interaction subsystem:
        self.subsystems["rudder_hull_interaction"] = RudderHullInteractionSystem(
            ship=self, create_jacobians=self.create_jacobians
        )


def hull(model: ModularVesselSimulator, create_jacobians=True):
    eq_X_H = sp.Eq(
        X_H,
        p.X0
        + p.Xu * u
        + p.Xuu * u**2
        + p.Xuuu * u**3
        + p.Xvv * v**2
        + p.Xrr * r**2
        + p.Xvr * v * r
        ## + p.Xthrust * thrust,
        # + p.Xuvv * u * v**2 + p.Xurr * u * r**2 + p.Xuvr * u * v * r,
    )
    eq_Y_H = sp.Eq(
        Y_H,
        p.Yv * v + p.Yr * r
        # + p.Yvr * v * r
        + p.Yvvv * v**3 + p.Yvvr * v**2 * r + p.Yrrr * r**3 + p.Yvrr * v * r**2
        # + p.Yuuv * u**2 * v
        # + p.Yuur * u**2 * r
        # + p.Yuv * u * v
        # + p.Yur * u * r
        ## + p.Ythrust * thrust
        + p.Y0  # Very important!
        # + p.Y0u * u + p.Y0uu * u**2,
    )
    eq_N_H = sp.Eq(
        N_H,
        p.Nv * v + p.Nr * r + p.Nvvv * v**3
        # + p.Nvr * v * r
        + p.Nvvr * v**2 * r
        + p.Nrrr * r**3
        + p.Nvrr * v * r**2  # This one is very important to not get the drift...
        # + p.Nuuv * u**2 * v
        # + p.Nuur * u**2 * r
        # + p.Nuv * u * v
        # + p.Nur * u * r
        ## + p.Nthrust * thrust
        + p.N0  # Very important !
        # + p.N0u * u + p.N0uu * u**2,
    )
    equations_hull = [eq_X_H, eq_Y_H, eq_N_H]
    hull = PrimeEquationSubSystem(
        ship=model, equations=equations_hull, create_jacobians=create_jacobians
    )
    return hull


def hull_quadratic(model: ModularVesselSimulator, create_jacobians=True):
    eq_X_H = sp.Eq(
        X_H,
        p.X0 + p.Xu * u
        # + p.Xuu * u**2
        # + p.Xuuu * u**3
        + p.Xvv * v**2 + p.Xrr * r**2 + p.Xvr * v * r
        ## + p.Xthrust * thrust,
        # + p.Xuvv * u * v**2 + p.Xurr * u * r**2 + p.Xuvr * u * v * r,
    )
    eq_Y_H = sp.Eq(
        Y_H,
        p.Yv * v + p.Yr * r
        # + p.Yvr * v * r
        + p.Yvv * v * sp.Abs(v)
        + p.Yvvr * v**2 * r
        + p.Yrr * r * sp.Abs(r)
        + p.Yvrr * v * r**2
        # + p.Yuuv * u**2 * v
        # + p.Yuur * u**2 * r
        # + p.Yuv * u * v
        # + p.Yur * u * r
        ## + p.Ythrust * thrust
        + p.Y0  # Very important!
        # + p.Y0u * u + p.Y0uu * u**2,
    )
    eq_N_H = sp.Eq(
        N_H,
        p.Nv * v + p.Nr * r + p.Nvv * v * sp.Abs(v)
        # + p.Nvr * v * r
        + p.Nvvr * sp.Abs(v) * r
        + p.Nrr * r * sp.Abs(r)
        + p.Nvrr * v * sp.Abs(r)  # This one is very important to not get the drift...
        # + p.Nuuv * u**2 * v
        # + p.Nuur * u**2 * r
        # + p.Nuv * u * v
        # + p.Nur * u * r
        ## + p.Nthrust * thrust
        + p.N0  # Very important !
        # + p.N0u * u + p.N0uu * u**2,
    )
    equations_hull = [eq_X_H, eq_Y_H, eq_N_H]
    hull = PrimeEquationSubSystem(
        ship=model, equations=equations_hull, create_jacobians=create_jacobians
    )
    return hull
