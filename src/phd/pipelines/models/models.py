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
    add_wind_force_system_simple,
)
from .subsystems import add_wind_force_system as add_wind
from vessel_manoeuvring_models.prime_system import PrimeSystem
from .nodes import regress_wind_tunnel_test

import logging

log = logging.getLogger(__name__)


def vmm_abkowitz(main_model: ModularVesselSimulator) -> ModularVesselSimulator:
    eq_X_H = sp.Eq(
        X_H,
        p.X0 + p.Xu * u
        # + p.Xuu * u**2
        # + p.Xuuu * u**3
        + p.Xvv * v**2
        # + p.Xrr * r**2
        + p.Xvr * v * r
        ## + p.Xthrust * thrust,
        # + p.Xuvv * u * v**2 + p.Xurr * u * r**2 + p.Xuvr * u * v * r,
    )

    eq_Y_H = sp.Eq(
        Y_H,
        p.Yv * v
        # + p.Yr * r
        + p.Yvvv * v**3
        # + p.Yvvr * v**2 * r
        # + p.Yrrr * r**3
        + p.Yvrr * v * r**2
        # + p.Yuuv * u**2 * v + p.Yuur * u**2 * r
        # + p.Yuv * u * v + p.Yur * u * r
        ## + p.Ythrust * thrust
        + p.Y0
        # + p.Y0u * u + p.Y0uu * u**2,
    )
    eq_N_H = sp.Eq(
        N_H,
        p.Nv * v + p.Nr * r + p.Nvvv * v**3 + p.Nvvr * v**2 * r
        # + p.Nrrr * r**3
        + p.Nvrr * v * r**2  # This one is very important to not get the drift...
        # + p.Nuuv * u**2 * v
        # + p.Nuur * u**2 * r
        # + p.Nuv * u * v + p.Nur * u * r
        ## + p.Nthrust * thrust
        + p.N0
        # + p.N0u * u + p.N0uu * u**2,
    )

    eq_X_R = sp.Eq(
        X_R,
        # p.Xrdelta * r * delta
        # + p.Xurdelta * u * r * delta
        +p.Xdeltadelta * delta**2
        # + p.Xudeltadelta * u * delta**2
        # + p.Xvdelta * v * delta
        # + p.Xuvdelta * u * v * delta,
    )
    eq_Y_R = sp.Eq(
        Y_R,
        p.Ydelta * delta
        # + p.Ydeltadeltadelta * delta**3 + p.Yudelta * u * delta
        # + p.Yuudelta * u**2 * delta
        # + p.Yvdeltadelta * v * delta**2
        # + p.Yvvdelta * v**2 * delta
        # + p.Yrdeltadelta * r * delta**2
        # + p.Yrrdelta * r**2 * delta
        # + p.Yvrdelta * v * r * delta
        # + p.Ythrustdelta * thrust * delta,
    )
    eq_N_R = sp.Eq(
        N_R,
        p.Ndelta * delta
        # + p.Ndeltadeltadelta * delta**3 + p.Nudelta * u * delta
        # + p.Nuudelta * u**2 * delta
        # + p.Nrrdelta * r**2 * delta + p.Nvrdelta * v * r * delta
        # + p.Nvdeltadelta * v * delta**2
        # + p.Nrdeltadelta * r * delta**2
        # + p.Nvvdelta * v**2 * delta
        + p.Nthrustdelta * thrust * delta,
    )

    model = main_model.copy()
    equations_hull = [eq_X_H, eq_Y_H, eq_N_H]
    hull = PrimeEquationSubSystem(
        ship=model, equations=equations_hull, create_jacobians=True
    )
    model.subsystems["hull"] = hull

    ## Add propeller:
    add_propeller(model=model)

    ## Add rudder:
    equations_rudders = [eq_X_R, eq_Y_R, eq_N_R]
    rudders = PrimeEquationSubSystem(
        ship=model, equations=equations_rudders, create_jacobians=True
    )
    model.subsystems["rudders"] = rudders

    add_dummy_wind_force_system(model=model)

    return model


def vmm_abkowitz_rudder_wind(
    main_model: ModularVesselSimulator,
    wind_data_HMD: pd.DataFrame,
) -> ModularVesselSimulator:
    eq_X_H = sp.Eq(
        X_H,
        p.Xu * u
        # + p.Xuu * u**2
        # + p.Xuuu * u**3
        + p.Xvv * v**2
        # + p.Xrr * r**2
        + p.Xvr * v * r
        ## + p.Xthrust * thrust,
        # + p.Xuvv * u * v**2 + p.Xurr * u * r**2 + p.Xuvr * u * v * r,
    )

    eq_Y_H = sp.Eq(
        Y_H,
        p.Yv * v
        # + p.Yr * r
        + p.Yvvv * v**3
        # + p.Yvvr * v**2 * r
        # + p.Yrrr * r**3
        + p.Yvrr * v * r**2
        # + p.Yuuv * u**2 * v + p.Yuur * u**2 * r
        # + p.Yuv * u * v + p.Yur * u * r
        ## + p.Ythrust * thrust
        + p.Y0
        # + p.Y0u * u + p.Y0uu * u**2,
    )
    eq_N_H = sp.Eq(
        N_H,
        p.Nv * v + p.Nr * r + p.Nvvv * v**3 + p.Nvvr * v**2 * r
        # + p.Nrrr * r**3
        + p.Nvrr * v * r**2  # This one is very important to not get the drift...
        # + p.Nuuv * u**2 * v
        # + p.Nuur * u**2 * r
        # + p.Nuv * u * v + p.Nur * u * r
        ## + p.Nthrust * thrust
        + p.N0
        # + p.N0u * u + p.N0uu * u**2,
    )

    model = main_model.copy()
    equations_hull = [eq_X_H, eq_Y_H, eq_N_H]
    hull = PrimeEquationSubSystem(
        ship=model, equations=equations_hull, create_jacobians=True
    )
    model.subsystems["hull"] = hull

    ## Add propeller:
    add_propeller(model=model)

    ## Add rudder:
    add_rudder(model=model)

    add_wind_force_system_simple(model=model)
    model = regress_wind_tunnel_test(model, wind_data_HMD=wind_data_HMD)

    return model


def vmm_full_abkowitz_rudder_wind(
    main_model: ModularVesselSimulator,
    wind_data_HMD: pd.DataFrame,
) -> ModularVesselSimulator:
    eq_X_H = sp.Eq(
        X_H,
        p.X0 + p.Xu * u
        # + p.Xuu * u**2
        # + p.Xuuu * u**3
        + p.Xvv * v**2 + p.Xrr * r**2 + p.Xvr * v * r
        ## + p.Xthrust * thrust,
        + p.Xuvv * u * v**2 + p.Xurr * u * r**2 + p.Xuvr * u * v * r,
    )

    eq_Y_H = sp.Eq(
        Y_H,
        p.Yv * v
        + p.Yr * r
        + p.Yvvv * v**3
        + p.Yvvr * v**2 * r
        + p.Yrrr * r**3
        + p.Yvrr * v * r**2
        + p.Yuuv * u**2 * v
        + p.Yuur * u**2 * r
        + p.Yuv * u * v
        + p.Yur * u * r
        ## + p.Ythrust * thrust
        + p.Y0 + p.Y0u * u + p.Y0uu * u**2,
    )
    eq_N_H = sp.Eq(
        N_H,
        p.Nv * v
        + p.Nr * r
        + p.Nvvv * v**3
        + p.Nvvr * v**2 * r
        + p.Nrrr * r**3
        + p.Nvrr * v * r**2  # This one is very important to not get the drift...
        + p.Nuuv * u**2 * v
        + p.Nuur * u**2 * r
        + p.Nuv * u * v
        + p.Nur * u * r
        ## + p.Nthrust * thrust
        + p.N0 + p.N0u * u + p.N0uu * u**2,
    )

    model = main_model.copy()
    equations_hull = [eq_X_H, eq_Y_H, eq_N_H]
    hull = PrimeEquationSubSystem(
        ship=model, equations=equations_hull, create_jacobians=True
    )
    model.subsystems["hull"] = hull

    ## Add propeller:
    add_propeller(model=model)

    ## Add rudder:
    add_rudder(model=model)

    add_wind_force_system_simple(model=model)
    model = regress_wind_tunnel_test(model, wind_data_HMD=wind_data_HMD)

    return model


def vmm_abkowitz_complex_rudder_wind(
    main_model: ModularVesselSimulator,
    wind_data_HMD: pd.DataFrame,
) -> ModularVesselSimulator:
    eq_X_H = sp.Eq(
        X_H,
        p.Xu * u
        # + p.Xuu * u**2
        # + p.Xuuu * u**3
        + p.Xvv * v**2
        # + p.Xrr * r**2
        + p.Xvr * v * r
        ## + p.Xthrust * thrust,
        # + p.Xuvv * u * v**2 + p.Xurr * u * r**2 + p.Xuvr * u * v * r,
    )

    eq_Y_H = sp.Eq(
        Y_H,
        p.Yv * v + p.Yr * r + p.Yvvv * v**3 + p.Yvvr * v**2 * r
        # + p.Yrrr * r**3
        + p.Yvrr * v * r**2
        # + p.Yuuv * u**2 * v + p.Yuur * u**2 * r
        + p.Yuv * u * v
        # + p.Yur * u * r
        ## + p.Ythrust * thrust
        + p.Y0 + p.Y0u * u
        # + p.Y0uu * u**2,
    )
    eq_N_H = sp.Eq(
        N_H,
        p.Nv * v + p.Nr * r + p.Nvvv * v**3 + p.Nvvr * v**2 * r
        # + p.Nrrr * r**3
        + p.Nvrr * v * r**2  # This one is very important to not get the drift...
        # + p.Nuuv * u**2 * v
        # + p.Nuur * u**2 * r
        + p.Nuv * u * v + p.Nur * u * r
        ## + p.Nthrust * thrust
        + p.N0 + p.N0u * u
        # + p.N0uu * u**2,
    )

    model = main_model.copy()
    equations_hull = [eq_X_H, eq_Y_H, eq_N_H]
    hull = PrimeEquationSubSystem(
        ship=model, equations=equations_hull, create_jacobians=True
    )
    model.subsystems["hull"] = hull

    ## Add propeller:
    add_propeller(model=model)

    ## Add rudder:
    add_rudder(model=model)

    add_wind_force_system_simple(model=model)
    model = regress_wind_tunnel_test(model, wind_data_HMD=wind_data_HMD)

    return model
