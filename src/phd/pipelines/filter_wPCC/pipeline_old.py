"""
This is a boilerplate pipeline 'filter'
generated using Kedro 0.18.7
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    guess_covariance_matrixes_many,
    initial_state_many,
    filter_many1,
    filter_many2,
    smoother_many,
    get_tests_ek,
    get_tests_ek_smooth,
    join_tests,
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
                func=guess_covariance_matrixes_many,
                inputs=["params:ek_covariance_input2", "tests"],
                outputs="covariance_matrixes2",
                name="guess_covariance_matrixes_node2",
                tags=["ek2", "filter2"],
            ),
            node(
                func=initial_state_many,
                inputs=["tests"],  # (data has the raw positions)
                outputs="x0",
                name="initial_state_node",
                tags=["ek", "filter"],
            ),
            ## First filtering:
            node(
                func=filter_many1,
                inputs=[
                    "tests",
                    "models_rudder_VCT",
                    #"models_VCT",
                    #"models",
                    "covariance_matrixes",
                    "x0",
                    "params:filter_model_name",
                    "params:accelerometer_position",
                    "params:skip",
                ],
                # outputs=[
                #    "ek_filtered",
                #    "time_steps",
                #    "tests_ek",
                # ],
                outputs="ek_filtered",
                name="filter_node",
                tags=["ek", "filter"],
            ),
            node(
                func=get_tests_ek,
                inputs=["ek_filtered"],
                outputs="tests_ek",
                name="get_tests_ek",
            ),
            node(
                func=join_tests,
                inputs=["tests_ek", "params:skip"],
                outputs="tests_ek_joined",
                name="join_tests_ek",
            ),
            ## 2nd filtering:
            node(
                func=filter_many2,
                inputs=[
                    "tests_ek",
                    "models_ID_hull_rudder",  # Now use this model instead
                    #"models_VCT",
                    #"models",
                    "covariance_matrixes2",
                    "x0",
                    "params:filter_model_name",
                    "params:accelerometer_position",
                    "params:skip",
                ],
                # outputs=[
                #    "ek_filtered",
                #    "time_steps",
                #    "tests_ek",
                # ],
                outputs="ek_filtered2",
                name="filter_node2",
                tags=["ek2", "filter2"],
            ),
            node(
                func=get_tests_ek,
                inputs=["ek_filtered2"],
                outputs="tests_ek2",
                name="get_tests_ek2",
            ),
            node(
                func=join_tests,
                inputs=["tests_ek2", "params:skip"],
                outputs="tests_ek_joined2",
                name="join_tests_ek2",
            ),
            node(
                func=smoother_many,
                inputs=[
                    "ek_filtered",
                    # "tests",  # (data has the raw positions)
                    # "time_steps",
                    # "covariance_matrixes",
                    "params:accelerometer_position",
                    "params:skip",
                ],
                # outputs=["ek_smooth", "tests_ek_smooth"],
                outputs="ek_smooth",
                name="smoother_node",
                tags=["ek", "filter"],
            ),
            node(
                func=get_tests_ek_smooth,
                inputs=["ek_smooth"],
                outputs="tests_ek_smooth",
                name="get_tests_ek_smooth",
            ),
            node(
                func=join_tests,
                inputs=["tests_ek_smooth", "params:skip"],
                outputs="tests_ek_smooth_joined",
                name="join_tests_ek_smooth",
            ),
        ]
    )
