"""
This is a boilerplate pipeline 'models_7m'
generated using Kedro 0.18.14
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import scale_model

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        
        node(
            func=scale_model,
            inputs=["wPCC.models_rudder_VCT", "7m.ship_data"],
            outputs="7m.models_rudder_VCT",
            name="scale_model",
            tags=["generate_model"]
        ),
        
        node(
            func=scale_model,
            inputs=["wPCC.models_VCT", "7m.ship_data"],
            outputs="7m.models_VCT",
            name="scale_model2",
            tags=["generate_model"]
        ),
        
    ])
