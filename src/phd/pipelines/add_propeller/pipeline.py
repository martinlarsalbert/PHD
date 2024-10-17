"""
This is a boilerplate pipeline 'add_propeller'
generated using Kedro 0.18.14
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import replace_simple_propellers, fit_open_water_characteristics

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
        
        node(
                func=replace_simple_propellers,
                inputs=["models_VCT"],
                outputs="models_VCT_propeller_",
                name="replace_simple_propellers",
                tags=['add_propeller',"generate_model"],
            ),
        node(
                func=fit_open_water_characteristics,
                inputs=["models_VCT_propeller_","open_water_characteristics"],
                outputs="models_VCT_propeller",
                name="fit_open_water_characteristics",
                tags=['add_propeller',"generate_model"],
            ),
                
    ])
