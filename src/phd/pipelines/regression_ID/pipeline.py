"""
This is a boilerplate pipeline 'regression_ID'
generated using Kedro 0.18.7
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import regress_hull_inverse_dynamics, regress_inverse_dynamics

tags = ['inverse_dynamics_regression']

def create_pipeline(**kwargs) -> Pipeline:
    
    nodes = []
    #N=3
    N=1
    for i in range(1,N+1):
        nodes+=create_regressions(i=i)
    
    return pipeline(
            nodes
    )
    
def create_regressions(i:int):
    
    nodes = [
        node(
                func=regress_hull_inverse_dynamics,
                #inputs=["models_VCT_MDL", f"tests_ek_joined{i}"],
                inputs=["models_VCT_MDL", f"tests_ek_smooth_joined{i}"],
                outputs=f"models_ID_hull{i}",
                name=f"regress_hull_inverse_dynamics{i}",
                tags=tags,
            ),
            node(
                func=regress_inverse_dynamics,
                #inputs=["base_models_simple", f"tests_ek_joined{i}", "models_VCT_MDL"],
                inputs=["base_models_simple", f"tests_ek_smooth_joined{i}", "models_VCT_MDL"],
                outputs=f"models_ID_hull_rudder{i}",
                name=f"regress_inverse_dynamics{i}",
                tags=tags,
            ),
    ]
    
    return nodes