"""
This is a boilerplate pipeline 'regression_ID'
generated using Kedro 0.18.7
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import regress_hull_inverse_dynamics, regress_inverse_dynamics


def create_pipeline(**kwargs) -> Pipeline:
    
    tags = ['kedro']
    
    return pipeline(
        [
            node(
                func=regress_hull_inverse_dynamics,
                inputs=["models_VCT_MDL_optimize", "tests_ek_smooth_joined"],
                outputs="models_ID_hull",
                name="regress_hull_inverse_dynamics",
                tags=tags,
            ),
            node(
                func=regress_inverse_dynamics,
                inputs=["base_models_simple", "tests_ek_smooth_joined", "models_VCT_MDL_optimize"],
                outputs="models_ID_hull_rudder",
                name="regress_inverse_dynamics",
                tags=tags,
            ),
        ]
    )
