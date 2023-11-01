"""
This is a boilerplate pipeline 'regression_VCT'
generated using Kedro 0.18.7
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import load_VCT, add_extra_circle_drift, regress_hull_VCT, adopting_to_MDL


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=load_VCT,
                inputs=["df_VCT_all"],
                outputs="df_VCT_raw",
                name="load_VCT_node",
            ),
            node(
                func=add_extra_circle_drift,
                inputs=["df_VCT_raw"],
                outputs="df_VCT",
                name="add_extra_circle_drift_node",
            ),
            node(
                func=regress_hull_VCT,
                inputs=["base_models", "df_VCT"],
                outputs="models_VCT",
                name="regress_hull_VCT",
            ),
            node(
                func=adopting_to_MDL,
                inputs=["models_VCT", "resistance_MDL"],
                outputs="models_VCT_MDL",
                name="adopting_to_MDL",
            ),
        ]
    )
