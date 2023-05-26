"""
This is a boilerplate pipeline 'load_7m'
generated using Kedro 0.18.7
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import load, divide_into_tests


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=load,
                inputs=["7m.time_series_raw", "params:7m.GPS_position", "7m.missions"],
                outputs=["7m.time_series", "7m.time_series_meta_data", "7m.units"],
            ),
            node(
                func=divide_into_tests,
                inputs=["7m.time_series", "7m.units"],
                outputs=["7m.tests", "7m.test_meta_data"],
            ),
        ]
    )
