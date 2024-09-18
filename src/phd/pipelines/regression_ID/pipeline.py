"""
This is a boilerplate pipeline 'regression_ID'
generated using Kedro 0.18.7
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import regress_hull_inverse_dynamics, regress_inverse_dynamics, training_data

tags = ['inverse_dynamics_regression']

def create_pipeline(**kwargs) -> Pipeline:
    
    nodes = [
        
        # Small:
        node(
                func=training_data,
                inputs=[f"tests_ek_smooth1", f"params:training_data_small_ids"],
                outputs=f"training_data_small",
                name=f"training_data_small",
                tags=tags,
            ),
        node(
                func=regress_hull_inverse_dynamics,
                #inputs=["models_VCT_MDL", f"tests_ek_joined{i}"],
                inputs=["models_VCT_nonlinear_MDL", f"training_data_small"],
                outputs=f"models_ID_hull_small",
                name=f"regress_hull_inverse_dynamics_small",
                tags=tags,
            ),
        
        # Medium
        node(
                func=training_data,
                inputs=[f"tests_ek_smooth1", f"params:training_data_medium_ids"],
                outputs=f"training_data_medium",
                name=f"training_data_medium",
                tags=tags,
            ),
        node(
                func=regress_hull_inverse_dynamics,
                #inputs=["models_VCT_MDL", f"tests_ek_joined{i}"],
                inputs=["models_VCT_nonlinear_MDL", f"training_data_medium"],
                outputs=f"models_ID_hull_medium",
                name=f"regress_hull_inverse_dynamics_medium",
                tags=tags,
            ),

        # All
        node(
                func=training_data,
                inputs=[f"tests_ek_smooth1", f"params:training_data_all_ids"],
                outputs=f"training_data_all",
                name=f"training_data_all",
                tags=tags,
            ),
        node(
                func=regress_hull_inverse_dynamics,
                #inputs=["models_VCT_MDL", f"tests_ek_joined{i}"],
                inputs=["models_VCT_nonlinear_MDL", f"training_data_all"],
                outputs=f"models_ID_hull_all",
                name=f"regress_hull_inverse_dynamics_all",
                tags=tags,
            ),

            
    ]
    
    return nodes
    
    