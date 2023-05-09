"""
This is a boilerplate pipeline 'load_7m'
generated using Kedro 0.18.7
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import load


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=load,
                inputs=["7m.time_series_raw", "params:7m.GPS_position"],
                outputs=["7m.time_series", "7m.time_series_meta_data"],
            )
        ]
    )
