"""
This is a boilerplate pipeline 'models_7m'
generated using Kedro 0.18.14
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    scale_model,
    base_models,
    #base_models_simple,
)

# from .subsystems import add_wind_force_system


def create_pipeline(**kwargs) -> Pipeline:
    nodes = [
        node(
            func=base_models,
            inputs=["ship_data", "params:parameters", "wind_data"],
            outputs="base_models",
            name="base_models",
            tags=["generate_model"]
        ),
        #node(
        #    func=base_models_simple,
        #    inputs=["ship_data", "params:parameters"],
        #    outputs="base_models_simple",
        #    name="base_models_simple",
        #    tags=["generate_model"]
        #),
    ]
    return pipeline(nodes)


#def create_pipeline(**kwargs) -> Pipeline:
#    return pipeline([
#        
#        node(
#            func=scale_model,
#            inputs=["wPCC.models_rudder_VCT", "7m.ship_data"],
#            outputs="7m.models_rudder_VCT",
#            name="scale_model",
#            tags=["generate_model"]
#        ),
#        
#        node(
#            func=scale_model,
#            inputs=["wPCC.models_VCT", "7m.ship_data"],
#            outputs="7m.models_VCT",
#            name="scale_model2",
#            tags=["generate_model"]
#        ),
#        
#    ])
