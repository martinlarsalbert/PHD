import pandas as pd
from vessel_manoeuvring_models.symbols import *

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
    add_propeller_simple,
    add_propellers_simple,
    add_rudder_simple,
    add_dummy_wind_force_system,
)
from vessel_manoeuvring_models.models.rudder_hull_interaction import (
    RudderHullInteractionSystem,
    RudderHullInteractionDummySystem,
)
from vessel_manoeuvring_models.models import semiempirical_rudder_MAK
from vessel_manoeuvring_models.models.semiempirical_covered_system import (
    SemiempiricalRudderSystemCovered,
)
from vessel_manoeuvring_models.models.MMG_wake_system import MMGWakeSystem
from vessel_manoeuvring_models.models.abkowitz_rudder_system import AbkowitzRudderSystem
from vessel_manoeuvring_models.models.simple_rudder_system import SimpleRudderSystem
from .subsystems import add_wind_force_system as add_wind
from vessel_manoeuvring_models.prime_system import PrimeSystem
from .models_wPCC import hull

import logging

log = logging.getLogger(__name__)

from vessel_manoeuvring_models.models.fossen import eq_main_expanded_matrix


class MainModel(ModularVesselSimulator):
    def __init__(self, create_jacobians=True):
        """General model that is used to create all other models

        Returns
        -------
        ModularVesselSimulator
            _description_
        """
        log.info("Creating the general model")
        
        #X_eq = sp.Eq(m * (u1d - r * v - x_G * r ** 2),
        #     p.Xudot * u1d - p.Yvdot*v*r - p.Yrdot*r**2 + X_D)
        #
        #Y_eq = sp.Eq(m * (v1d + r * u + x_G * r1d),
        #     p.Yvdot*v1d + p.Xudot*u*r + p.Yrdot*r1d + Y_D)
        #
        #N_eq = sp.Eq(I_z * r1d + m * x_G * (v1d + u * r),
        #    p.Nrdot * r1d + (p.Yvdot-p.Xudot)*u*v + p.Yrdot*(v1d+u*r) + N_D)
        
        X_eq = sp.Eq(eq_main_expanded_matrix.lhs[0],eq_main_expanded_matrix.rhs[0])
        Y_eq = sp.Eq(eq_main_expanded_matrix.lhs[1],eq_main_expanded_matrix.rhs[1])
        N_eq = sp.Eq(eq_main_expanded_matrix.lhs[2],eq_main_expanded_matrix.rhs[2])
        
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

        f_X_RHI = sp.Function("X_RHI")()
        f_Y_RHI = sp.Function("Y_RHI")()
        f_N_RHI = sp.Function("N_RHI")()

        eq_X_force = sp.Eq(X_D, f_X_H + f_X_R + f_X_P + f_X_W + f_X_RHI)
        eq_Y_force = sp.Eq(Y_D, f_Y_H + f_Y_R + f_Y_P + f_Y_W + f_Y_RHI)
        eq_N_force = sp.Eq(N_D, f_N_H + f_N_R + f_N_P + f_N_W + f_N_RHI)
        
        X_eq = X_eq.subs(X_D,eq_X_force.rhs)
        Y_eq = Y_eq.subs(Y_D,eq_Y_force.rhs)
        N_eq = N_eq.subs(N_D,eq_N_force.rhs)
        
        
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
        add_dummy_wind_force_system(model=self, create_jacobians=self.create_jacobians)

        self.control_keys = ["delta", "thrust_port", "thrust_stbd"]

    def setup_parameters(self, ship_data: dict):
        ship_data = ship_data.copy()
        ship_data["r_0"] = ship_data["D"] / 2
        
        if not "x" in ship_data:
            ship_data["x"] = ship_data["x_p"] - ship_data["x_r"]
        
        self.set_ship_parameters(ship_data)

        if not "Xthrustport" in self.parameters:
            self.parameters["Xthrustport"] = 1 - self.ship_parameters["tdf"]

        if not "Xthruststbd" in self.parameters:
            self.parameters["Xthruststbd"] = 1 - self.ship_parameters["tdf"]

        if not "Xthrust" in self.parameters:
            self.parameters["Xthrust"] = 1 - self.ship_parameters["tdf"]

        if not "gamma_0" in self.parameters:
            self.parameters["gamma_0"] = 0

        if not "C_L_tune" in self.parameters:
            self.parameters["C_L_tune"] = 1

        if not "C_D_tune" in self.parameters:
            self.parameters["C_D_tune"] = 1

        if not "C_D0_tune" in self.parameters:
            self.parameters["C_D0_tune"] = 1

        if not "x_H" in self.parameters:
            self.parameters["x_H"] = -1 / 2

    def add_added_mass(self):
        mask = df_parameters["state"] == "dot"
        lambdas_added_mass = df_parameters.loc[mask, "brix_lambda"].dropna()
        added_masses = {
            key: run(lambda_, **self.ship_parameters)
            for key, lambda_ in lambdas_added_mass.items()
        }
        for key,value in added_masses.items():
            if not key in self.parameters:
                self.parameters[key] = added_masses[key]  # (avoid overwriting)

    def add_hull(self):
        self.subsystems["hull"] = hull(
            model=self, create_jacobians=self.create_jacobians
        )

    def add_propellers(self):
        if self.is_twin_screw:
            add_propellers_simple(
                model=self, create_jacobians=self.create_jacobians
            )
        else:
            add_propeller_simple(
                model=self, create_jacobians=self.create_jacobians
            )

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
        
    @property
    def is_twin_screw(self):
        if self.ship_parameters['TWIN']==1:
            return True
        elif self.ship_parameters['TWIN']==0:
            return False
        else:
            raise ValueError(f"Bad value for TWIN:{self.ship_parameters['TWIN']}") 


class ModelTowedSemiempiricalCovered(ModelTowed):
    def setup_subsystems(self):
        ## Add hull:
        self.add_hull()

        if not self.is_twin_screw:
            #For single screw use different wake for different inflow angles (beta_p):
            self.subsystems['mmg_wake_system'] = MMGWakeSystem(ship=self, create_jacobians=self.create_jacobians, suffix='')
            
        ## Add propeller:
        self.add_propellers()

        ## Add rudders:
        self.add_rudders(in_propeller_race=True)

        ## Add dummy wind system:
        add_dummy_wind_force_system(model=self, create_jacobians=self.create_jacobians)

        if self.is_twin_screw:
            self.control_keys = ["delta", "thrust_port", "thrust_stbd"]
        else:
            self.control_keys = ["delta", "thrust"]

    def add_rudders(self, in_propeller_race: bool):
        rudder_port = SemiempiricalRudderSystemCovered(
            ship=self,
            create_jacobians=self.create_jacobians,
            in_propeller_race=in_propeller_race,
            suffix="port",
        )
        self.subsystems["rudder_port"] = rudder_port

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
        if self.is_twin_screw:
            self.add_rudders_twin_screw(in_propeller_race=in_propeller_race)
        else:
            self.add_rudders_single_screw(in_propeller_race=in_propeller_race)
    
    def add_rudders_twin_screw(self, in_propeller_race: bool):
        rudder_port = SemiempiricalRudderSystemCovered(
            ship=self,
            create_jacobians=self.create_jacobians,
            in_propeller_race=in_propeller_race,
            suffix="port",
        )
        self.subsystems["rudder_port"] = rudder_port

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
            "A_R_C": self.ship_parameters["D"] * self.ship_parameters["c"],
            "A_R_U": (self.ship_parameters["b_R"] - self.ship_parameters["D"])
            * self.ship_parameters["c"],
        }
        self.ship_parameters.update(rudder_particulars)
        rudder_parameters = {
            "nu": 1.18849e-06,
            "e_0": 0.9,
            "kappa": 0.85,
            "kappa_gamma": 0,
            "l_R": self.ship_parameters["x_r"],
            "delta_lim": np.deg2rad(35),
            "delta_alpha_s": np.deg2rad(0),  # Later stall when propeller race
            "Omega": 0,
            "delta_0": np.deg2rad(20),  # start of lift gap loss (S-curve)
            "delta_1": np.deg2rad(30),  # end of lift gap loss (S-curve)
            "s": 0,  # "stall" due to gap loss (S-curve), alternative to use delta_0 and delta_1
        }
        for key, value in rudder_parameters.items():
            if not key in self.parameters:
                self.parameters[key] = value

        ## Add rudder hull interaction subsystem:
        self.subsystems["rudder_hull_interaction"] = RudderHullInteractionSystem(
            ship=self, create_jacobians=self.create_jacobians
        )

    def add_rudders_single_screw(self, in_propeller_race: bool):
        rudder = SemiempiricalRudderSystemCovered(
            ship=self,
            create_jacobians=self.create_jacobians,
            in_propeller_race=in_propeller_race,
            suffix="",
        )
        self.subsystems["rudder"] = rudder

        ## Add rudders system (joining the rudder forces)
        #rudders = semiempirical_rudder_MAK.Rudders(
        #    ship=self, create_jacobians=self.create_jacobians
        #)
        #self.subsystems["rudders"] = rudders

        ## Default parameters:
        c = (self.ship_parameters['c_t'] + self.ship_parameters['c_r'])/2
        
        rudder_particulars = {
            "x_R": self.ship_parameters["x_r"],
            "y_R": 0,
            "z_R": 0,
            #"w_f": self.ship_parameters["w_p0"], # (w_f is calculated from the MMGWakeSystem)
            "A_R_C": self.ship_parameters["D"] * c,
            "A_R_U": (self.ship_parameters["b_R"] - self.ship_parameters["D"])
            * c,
        }
        self.ship_parameters.update(rudder_particulars)
        rudder_parameters = {
            "nu": 1.18849e-06,
            "e_0": 0.9,
            "kappa": 0.85,
            "kappa_gamma": 0,
            "l_R": self.ship_parameters["x_r"],
            "delta_lim": np.deg2rad(35),
            "delta_alpha_s": np.deg2rad(0),  # Later stall when propeller race
            "Omega": 0,
            "delta_0": np.deg2rad(20),  # start of lift gap loss (S-curve)
            "delta_1": np.deg2rad(30),  # end of lift gap loss (S-curve)
            "s": 0,  # "stall" due to gap loss (S-curve), alternative to use delta_0 and delta_1
        }
        for key, value in rudder_parameters.items():
            if not key in self.parameters:
                self.parameters[key] = value

        ## Add rudder hull interaction subsystem:
        self.subsystems["rudder_hull_interaction"] = RudderHullInteractionSystem(
            ship=self, create_jacobians=self.create_jacobians)