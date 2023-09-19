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
    estimate_propeller_speed_many,
    divide_into_tests_filtered,
)


def create_pipeline(**kwargs) -> Pipeline:
    # data = "tests"
    data = "time_series"
    # data = "time_series_"
    return pipeline(
        [
            node(
                func=guess_covariance_matrixes_many,
                inputs=["params:ek_covariance_input", data],
                outputs="covariance_matrixes",
                name="guess_covariance_matrixes_node",
                tags=["ek", "filter"],
            ),
            node(
                func=initial_state_many,
                inputs=[data],  # (data has the raw positions)
                outputs="x0",
                name="initial_state_node",
                tags=["ek", "filter"],
            ),
            node(
                func=estimate_propeller_speed_many,
                inputs=[
                    data,
                    "models",
                    "params:filter_model_name",
                ],
                outputs="time_series_rev",
                name="estimate_propeller_speed_node",
                tags=["ek", "filter"],
            ),
            node(
                func=filter_many,
                inputs=[
                    "time_series_rev",
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
                    "time_series_preprocessed.ek",
                ],
                name="filter_node",
                tags=["ek", "filter"],
            ),
            node(
                func=divide_into_tests_filtered,
                inputs=["time_series_preprocessed.ek", "units", "test_meta_data"],
                outputs=["tests_ek", "test_ek_meta_data"],
                name="divide_into_tests_filtered_ek",
            ),
            node(
                func=smoother_many,
                inputs=[
                    "ek_filtered",
                    "time_series_rev",  # (data has the raw positions)
                    "time_steps",
                    "covariance_matrixes",
                    "params:accelerometer_position",
                    "params:skip",
                ],
                outputs=["ek_smooth", "time_series_preprocessed.ek_smooth"],
                name="smoother_node",
                tags=["ek", "filter"],
            ),
            node(
                func=divide_into_tests_filtered,
                inputs=[
                    "time_series_preprocessed.ek_smooth",
                    "units",
                    "test_meta_data",
                ],
                outputs=["tests_ek_smooth", "test_ek_smooth_meta_data"],
                name="divide_into_tests_filtered_ek_smooth",
            ),
        ]
    )
