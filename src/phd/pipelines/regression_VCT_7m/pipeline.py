"""
This is a boilerplate pipeline 'regression_VCT'
generated using Kedro 0.18.7
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    load_VCT,
    select,
    scale,
    prime,
    add_extra_circle_drift,
    #limit_states_for_regression,
    regress_hull_VCT,
    #regress_hull_rudder_VCT,
    wave_generation_correction,
    scale_resistance,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=load_VCT,
                inputs=["df_VCT_all_raw"],
                outputs="df_VCT_all",
                name="load_VCT_node",
                tags=['load_VCT','regression_VCT'],
            ),
            node(
                func=select,
                inputs=["df_VCT_all"],
                outputs="df_VCT_raw",
                name="select_VCT_node",
                tags=['load_VCT','regression_VCT'],
            ),
            node(
                func=add_extra_circle_drift,
                inputs=["df_VCT_raw"],
                outputs="df_VCT",
                name="add_extra_circle_drift_node",
                tags=['load_VCT','regression_VCT'],
            ),
            node(
                func=scale,
                inputs=["df_VCT","ship_data"],
                outputs="df_VCT_scaled",
                name="scale_VCT_node",
                tags=['load_VCT','regression_VCT'],
            ),
            node(
                func=prime,
                inputs=["df_VCT","base_models"],
                outputs="df_VCT_prime",
                name="prime_VCT_node",
                tags=['load_VCT','regression_VCT'],
            ),
            
            #node(
            #    func=limit_states_for_regression,
            #    inputs=["df_VCT_scaled","tests","params:skip"],
            #    outputs="df_VCT_scaled_limited",
            #    name="limit_states_for_regression_node",
            #    tags=['load_VCT','regression_VCT'],
            #),
            
            node(
                func=regress_hull_VCT,
                inputs=["base_models", "df_VCT_scaled","params:VCT_exclude_parameters"],
                outputs="models_VCT",
                name="regress_hull_VCT",
                tags=["generate_model", "regression_VCT", "regression"]
            ),
            node(
                func=scale_resistance,
                inputs=["ship_data", "TT_resistance"],
                outputs="resistance",
                name="scale_resistance",
                tags=["generate_model", "regression_VCT"]
            ),
            node(
                func=wave_generation_correction,
                inputs=["models_VCT", "resistance","params:VCT_exclude_parameters"],
                outputs="models_VCT_wave_generation",
                name="wave_generation_correction",
                tags=["generate_model", "regression_VCT", "regression"]
            ),
            
            
        ]
    )
