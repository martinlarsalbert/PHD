"""
This is a boilerplate pipeline 'regression_VCT'
generated using Kedro 0.18.7
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    load,
    load_VCT,
    select,
    scale,
    add_extra_circle_drift,
    regress_hull_VCT,
    regress_hull_rudder_VCT,
    adopting_to_MDL,
    manual_regression,
    shape_optimization,
    adopting_hull_rudder_to_MDL,
    limit_states_for_regression,
    #adopting_nonlinear_to_MDL,
    subtract_centrifugal_and_centrepetal_forces,
)


def create_pipeline(ship_name:str, **kwargs) -> Pipeline:
    return pipeline(
        [
            
            node(
                    func=load_VCT,
                    inputs=["df_VCT_all_raw"],
                    outputs="df_VCT_all",
                    name="load_VCT_node",
                    tags=['load_VCT','regression_VCT'],
                ) if ship_name=='wPCC' else
            node(
                    func=load,
                    inputs=["df_VCT_all_raw"],
                    outputs="df_VCT_all",
                    name="load_VCT_node",
                    tags=['load_VCT','regression_VCT'],
                )
            ,
            node(
                func=select,
                inputs=["df_VCT_all","params:VCT_selection"],
                outputs="df_VCT_raw",
                name="select_VCT_node",
                tags=['load_VCT','regression_VCT'],
            ),
            node(
                func=add_extra_circle_drift,
                inputs=["df_VCT_raw","params:add_mirror_circle_drift"],
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
            #node(
            #    func=limit_states_for_regression,
            #    inputs=["df_VCT_scaled","tests","params:skip"],
            #    outputs="df_VCT_scaled_limited",
            #    name="limit_states_for_regression_node",
            #    tags=['load_VCT','regression_VCT_limited'],
            #),
            node(
                func=regress_hull_VCT,
                inputs=["base_models", "df_VCT_scaled","params:VCT_exclude_parameters", "params:optimize_rudder_inflow", "params:optimize_rudder_drag"],
                outputs="models_VCT",
                name="regress_hull_VCT",
                tags=["generate_model", "regression_VCT"]
            ),
            #node(
            #    func=regress_hull_VCT,
            #    inputs=["base_models", "df_VCT_scaled","params:VCT_nonlinear_exclude_parameters"],
            #    outputs="models_VCT_nonlinear",
            #    name="regress_hull_VCT_nonlinear",
            #    tags=["generate_model", "regression_VCT"]
            #),
            #node(
            #    func=regress_hull_rudder_VCT,
            #    inputs=["base_models", "df_VCT_scaled_limited", "params:VCT_exclude_parameters"],
            #    outputs="models_rudder_VCT",
            #    name="regress_hull_rudder_VCT",
            #    tags=["generate_model", "regression_VCT"]
            #),
            node(
                func=adopting_to_MDL,
                inputs=["models_VCT", "resistance_MDL"],
                outputs="models_VCT_MDL",
                name="adopting_to_MDL",
                tags=["generate_model", "regression_VCT", "adopting_to_MDL"],
            ),
            #node(
            #    func=adopting_hull_rudder_to_MDL,
            #    inputs=["models_rudder_VCT", "resistance_MDL", "models_VCT_MDL"],
            #    outputs="models_rudder_VCT_MDL",
            #    name="adopting_rudder_VCT_to_MDL",
            #    tags=["generate_model", "regression_VCT", "adopting_to_MDL"],
            #),
            #node(
            #    func=adopting_nonlinear_to_MDL,
            #    inputs=["models_VCT_nonlinear", "resistance_MDL"],
            #    outputs="models_VCT_nonlinear_MDL",
            #    name="adopting_nonlinear_to_MDL",
            #    tags=["generate_model", "regression_VCT", "adopting_to_MDL"],
            #),
            #node(
            #    func=shape_optimization,
            #    inputs=["models_VCT", "resistance_MDL", "tests_ek_smooth_joined3"],
            #    outputs="models_VCT_MDL_optimize",
            #    name="shape_optimization",
            #),
        ]
    )
