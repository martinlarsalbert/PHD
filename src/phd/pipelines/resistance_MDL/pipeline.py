"""
This is a boilerplate pipeline 'resistance_MDL'
generated using Kedro 0.18.7
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import resistance


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=resistance,
                inputs=["time_series_meta_data", "tests", "ship_data"],
                outputs="resistance_MDL",
                name="resistance",
            ),
        ]
    )
