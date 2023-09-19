"""
This is a boilerplate pipeline 'filter'
generated using Kedro 0.18.7
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    guess_covariance_matrixes_many,
    initial_state_many,
    filter_many,
    smoother_many,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=guess_covariance_matrixes_many,
                inputs=["params:ek_covariance_input", "tests"],
                outputs="covariance_matrixes",
                name="guess_covariance_matrixes_node",
                tags=["ek", "filter"],
            ),
            node(
                func=initial_state_many,
                inputs=["tests"],  # (data has the raw positions)
                outputs="x0",
                name="initial_state_node",
                tags=["ek", "filter"],
            ),
            node(
                func=filter_many,
                inputs=[
                    "tests",
                    "models",
                    "covariance_matrixes",
                    "x0",
                    "params:filter_model_name",
                    "params:accelerometer_position",
                    "params:skip",
                ],
                outputs=[
                    "ek_filtered",
                    "time_steps",
                    "tests_ek",
                ],
                name="filter_node",
                tags=["ek", "filter"],
            ),
            node(
                func=smoother_many,
                inputs=[
                    "ek_filtered",
                    "tests",  # (data has the raw positions)
                    "time_steps",
                    "covariance_matrixes",
                    "params:accelerometer_position",
                    "params:skip",
                ],
                outputs=["ek_smooth", "tests_ek_smooth"],
                name="smoother_node",
                tags=["ek", "filter"],
            ),
        ]
    )
