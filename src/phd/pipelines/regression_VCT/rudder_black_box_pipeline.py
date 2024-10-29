import pandas as pd
from vessel_manoeuvring_models.models.subsystem import PrimeEquationSubSystem
from vessel_manoeuvring_models.models.modular_simulator import ModularVesselSimulator
from vessel_manoeuvring_models.symbols import *
from vessel_manoeuvring_models.models.diff_eq_to_matrix import DiffEqToMatrix
import statsmodels.api as sm
import logging
from vessel_manoeuvring_models.substitute_dynamic_symbols import run
from vessel_manoeuvring_models.models import rudder_black_box
from vessel_manoeuvring_models.parameters import df_parameters
p=df_parameters['symbol']

log = logging.getLogger(__name__)


def pipeline_rudder_black_box(df_VCT_prime: pd.DataFrame, model: ModularVesselSimulator, rudder_suffix: str) -> dict:
    
    tests = df_VCT_prime.groupby(by="test type")

    if len(rudder_suffix) > 0:
        rudder_system_name = f"rudder_{rudder_suffix}"
        y_key = f"Y_R_{rudder_suffix}"
        x_key = f"X_R_{rudder_suffix}"
    else:
        rudder_system_name = "rudders"
        y_key = "Y_R"
        x_key = "X_R"
            
    
    rudder_system = model.subsystems[rudder_system_name]

    regression_pipeline = {
        "Thrust variation": {
            "eq": rudder_system.equations[y_key].subs([
                (p.Y_Rdelta,0),
                (p.Y_R0,0),
                (v,0),
                (r,0)
                ]),
            "data": tests.get_group("Thrust variation").copy(),
            "const": True,
        },
        
        "Rudder angle": {
            "eq": rudder_system.equations[y_key].subs([
                (v,0),
                (r,0),
                ]),
            "data": tests.get_group("Rudder angle").copy(),
        },
        
        "Rudder angle X": {
            "eq": rudder_system.equations[x_key].subs([
                (v,0),
                (r,0),
                ]),
            "data": tests.get_group("Rudder angle").copy(),
        },
        
        "Drift angle": {
            "eq": rudder_system.equations[y_key].subs([
                (r,0),
                (delta,0)
                ]),
            "data": tests.get_group("Drift angle").copy(),
        },
        
        "Drift angle X": {
            "eq": rudder_system.equations[x_key].subs([
                (r,0),
                (delta,0)
                ]),
            "data": tests.get_group("Drift angle").copy(),
        },
                
        "Circle": {
            "eq": rudder_system.equations[y_key].subs([
                (v,0),
                (delta,0)
                ]),
            "data": tests.get_group("Circle").copy(),
        },
        
        "Circle X": {
            "eq": rudder_system.equations[x_key].subs([
                (v,0),
                (delta,0)
                ]),
            "data": tests.get_group("Circle").copy(),
        },
        
        "Circle + Drift": {
            "eq": rudder_system.equations[y_key].subs([
                (delta,0)
                ]),
            "data": tests.get_group("Circle + Drift").copy(),
        },
        
        'Rudder and drift angle': {
            "eq": rudder_system.equations[y_key].subs(r,0),
            "data": tests.get_group('Rudder and drift angle').copy(),
        },
        
        'Circle + rudder angle': {
            "eq": rudder_system.equations[y_key].subs(v,0),
            "data": tests.get_group('Circle + rudder angle').copy(),
        },
        
        'Circle + Drift + rudder angle': {
            "eq": rudder_system.equations[y_key],
            "data": tests.get_group('Circle + Drift + rudder angle').copy(),
        },
        
        
    }

    return regression_pipeline