"""
This is a boilerplate pipeline 'filter'
generated using Kedro 0.18.7
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    #guess_covariance_matrixes_many,
    initial_state_many,
    filter_many,
    results_to_dataframe,
    smoother,
    #filter_many2,
    #smoother_many,
    #get_tests_ek,
    #et_tests_ek_smooth,
    join_tests,
    create_kalman_filter
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=create_kalman_filter,
                inputs=["models_rudder_VCT"],
                outputs="ekf",
                name="create_kalman_filter_node",
                tags=["ek", "filter"],
            ),
            node(
                func=initial_state_many,
                inputs=["tests","ekf1"],  # (data has the raw positions)
                outputs="x0",
                name="initial_state_node",
                tags=["ek", "filter"],
            ),
            node(
                func=filter_many,
                inputs=[
                    "tests",
                    "ekf",                
                    "x0",
                    "params:skip",
                ],
                outputs="filtered_result",
                name="filter_node",
                tags=["ek", "filter"],
            ),
            
            node(
                func=results_to_dataframe,
                inputs=[
                    "filtered_result",
                ],
                outputs="tests_ek",
                name="results_to_dataframe",
                tags=["ek", "filter"],
            ),
            
            node(
                func=smoother,
                inputs=[
                    "ekf",
                    "filtered_result",
                ],
                outputs="tests_ek_smooth",
                name="smoother",
                tags=["ek", "filter"],
            ),
            
            
            node(
                func=join_tests,
                inputs=["tests_ek", "params:skip"],
                outputs="tests_ek_joined",
                name="join_tests_ek",
            ),
            
            node(
                func=join_tests,
                inputs=["tests_ek_smooth", "params:skip"],
                outputs="tests_ek_smooth_joined",
                name="join_tests_ek_smooth",
            ),
        ]
    )

def filter_pipeline(n:int, models:str):
    
    return [
            node(
                func=create_kalman_filter,
                inputs=[models],
                outputs=f"ekf{n}",
                name=f"create_kalman_filter_node{n}",
                tags=["ek", "filter"],
            ),
            
            node(
                func=filter_many,
                inputs=[
                    "tests",
                    f"ekf{n}",                
                    "x0",
                    "params:skip",
                ],
                outputs=f"filtered_result{n}",
                name=f"filter_node{n}",
                tags=["ek", "filter"],
            ),
            
            node(
                func=results_to_dataframe,
                inputs=[
                    f"filtered_result{n}",
                ],
                outputs=f"tests_ek{n}",
                name=f"results_to_dataframe{n}",
                tags=["ek", "filter"],
            ),
            
            node(
                func=smoother,
                inputs=[
                    f"ekf{n}",
                    f"filtered_result{n}",
                ],
                outputs=f"tests_ek_smooth{n}",
                name=f"smoother{n}",
                tags=["ek", "filter"],
            ),
            
            
            node(
                func=join_tests,
                inputs=[f"tests_ek{n}", "params:skip"],
                outputs=f"tests_ek_joined{n}",
                name=f"join_tests_ek{n}",
            ),
            
            node(
                func=join_tests,
                inputs=[f"tests_ek_smooth{n}", "params:skip"],
                outputs=f"tests_ek_smooth_joined{n}",
                name=f"join_tests_ek_smooth{n}",
            ),
        ]