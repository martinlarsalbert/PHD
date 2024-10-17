"""
This is a boilerplate pipeline 'add_propeller'
generated using Kedro 0.18.14
"""

from vessel_manoeuvring_models.models.modular_simulator import ModularVesselSimulator
from vessel_manoeuvring_models.models.propeller import PropellerSystem, PropellersSystem, PropellerSimpleSystem, PropellersSimpleSystem
from phd.helpers import lazy, lazy_iteration
import pandas as pd
from vessel_manoeuvring_models.models.diff_eq_to_matrix import DiffEqToMatrix
import vessel_manoeuvring_models.models.propeller as propeller
import logging
import statsmodels.api as sm
log = logging.getLogger(__name__)

def replace_simple_propellers(models: dict)->dict:
    
    #models_propellers = {}
    #    
    #for name, loader in models.items():
    #    models_propellers[name] = lazy(loaders={'model':loader}, function=_replace_simple_propellers)
    #    
    #return models_propellers
    
    return lazy_iteration(models, var_name='model', function=_replace_simple_propellers)
    

def _replace_simple_propellers(model: ModularVesselSimulator)->ModularVesselSimulator:
    
    if model.is_twin_screw:
        model=replace_simple_propeller_twin_screw(model=model)
    else:
        # Single screw:
        model=replace_simple_propeller_single_screw(model=model)
        
    return model


def replace_simple_propeller_single_screw(model: ModularVesselSimulator)->ModularVesselSimulator:
    
    simple_systems = [name for name, subsystem in model.subsystems.items() if isinstance(subsystem,PropellerSimpleSystem)]
    assert len(simple_systems) == 1
    simple_system_name = simple_systems[0]
    
    center_propeller = PropellerSystem(ship=model, create_jacobians=False, suffix="")
    
    # replace simple propeller: 
    model.insert_subsystem_before(simple_system_name, insert_name='propeller', insert_system=center_propeller)
    model.subsystems.pop(simple_system_name)
    
    i = model.control_keys.index('thrust')
    model.control_keys[i] = 'rev'
    
    return model

def replace_simple_propeller_twin_screw(model: ModularVesselSimulator)->ModularVesselSimulator:
    
    simple_systems = [name for name, subsystem in model.subsystems.items() if isinstance(subsystem,PropellersSimpleSystem)]
    assert len(simple_systems) == 1
    
    simple_system_name = simple_systems[0]
    
    propeller_port = PropellerSystem(ship=model, create_jacobians=False, suffix="port")
    propeller_stbd = PropellerSystem(ship=model, create_jacobians=False, suffix="stbd")
    propellers = PropellersSystem(ship=model, create_jacobians=False, suffix='')
    
    
    # replace simple propellers:
    model.insert_subsystem_before(simple_system_name, insert_name='propeller_port', insert_system=propeller_port)
    model.insert_subsystem_before(simple_system_name, insert_name='propeller_stbd', insert_system=propeller_stbd)
    model.subsystems.pop(simple_system_name)
    model.insert_subsystem_after('propeller_stbd', insert_name='propellers', insert_system=propellers)
    
    return model

def fit_open_water_characteristics(models:dict, open_water_characteristics:pd.DataFrame)->dict:
    return lazy_iteration(models, var_name='model', function=_fit_open_water_characteristics, open_water_characteristics=open_water_characteristics)

def _fit_open_water_characteristics(model:ModularVesselSimulator, open_water_characteristics:pd.DataFrame)->ModularVesselSimulator:
    
    log.info("Fitting propeller characteristics")
    
    propeller_systems = [name for name, subsystem in model.subsystems.items() if isinstance(subsystem,PropellerSystem)]
    
    diff_eq_to_matrix = DiffEqToMatrix(propeller.eq_K_T, label=propeller.K_T, base_features=[propeller.J,])
    X,y=diff_eq_to_matrix.calculate_features_and_label(data=open_water_characteristics,y=open_water_characteristics['K_T'])
    fit = sm.OLS(y,X).fit()
    
    log.info(fit.summary2())
    
    model.parameters.update(
        {
                "C0_w_p0": model.ship_parameters['w_p0'],
                "C1_w_p0": 0,
                "k_0": fit.params['k0'],
                "k_1": fit.params['k1'],
                "k_2": fit.params['k2'],
        })
    
    return model