"""
This is a boilerplate pipeline 'models_7m'
generated using Kedro 0.18.14
"""
from vessel_manoeuvring_models.models.modular_simulator import ModularVesselSimulator
from vessel_manoeuvring_models.prime_system import PrimeSystem
from phd.pipelines.models.nodes import add_propeller

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