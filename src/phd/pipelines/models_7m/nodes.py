"""
This is a boilerplate pipeline 'models_7m'
generated using Kedro 0.18.14
"""
from vessel_manoeuvring_models.models.modular_simulator import ModularVesselSimulator
from vessel_manoeuvring_models.prime_system import PrimeSystem
from phd.pipelines.models.nodes import add_propeller

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
#+from wPCC_pipeline.pipelines.vct_data.nodes import vct_scaling
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

from .models_7m import ModelSemiempiricalCovered, ModelWithSimpleAbkowitzRudder, ModelMartinsSimple

import logging

log = logging.getLogger(__name__)


def base_models(ship_data: dict, parameters: dict, wind_data:pd.DataFrame) -> dict:
    models = {}

    name = "semiempirical_covered"
    log.info(f'Creating: "{name}"')
    model = ModelSemiempiricalCovered(ship_data=ship_data, create_jacobians=True)
    
    log.info("Regressing wind tunnel data")
    model.regress_wind_tunnel_test(wind_data=wind_data)
    
    if "rudder_port" in model.subsystems:
        delattr(
            model.subsystems["rudder_port"], "lambdas"
        )  # These do not work with pickle (for some reason)
    
    if "rudder_stbd" in model.subsystems:
        delattr(model.subsystems["rudder_stbd"], "lambdas")

    models[name] = model

    # Updating the parameters:
    for name, model in models.items():
        #parameters_ = parameters.get(name, parameters["default"])
        parameters_ = parameters["default"]
        log.info(f"Using default parameters:{parameters_}")
        model.parameters.update(parameters_)

    return models

def scale_model(model_loaders, ship_data:dict)->ModularVesselSimulator:
    
    prime_system_7m = PrimeSystem(L=ship_data['L'], rho=1012.5)
    
    models = {}
    
    for model_name, model_loader in model_loaders.items():
        model = model_loader()
        
        ship_data_scaled = model.ship_parameters
        ship_parameters_prime = model.prime_system.prime(model.ship_parameters)
        wPCC_ship_data_scaled = prime_system_7m.unprime(ship_parameters_prime)
        ship_data_scaled.update(wPCC_ship_data_scaled)
        ship_data_scaled.update(ship_data)  # Override, rudder x coord is for instance different...
                
        model.prime_system = prime_system_7m
        model.set_ship_parameters(ship_data_scaled)
        add_propeller(model=model)
        model.control_keys = ['delta', 'rev']
        
        models[model_name] = model
        
    return models

