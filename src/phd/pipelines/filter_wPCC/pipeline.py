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

tags = ["ek", "filter"]

def create_pipeline(**kwargs) -> Pipeline:
    
    nodes = []
    N = 1
    for n in range(1,N+1):
        nodes+=filter_pipeline(n=n, models="models_VCT_polynomial_rudder", filter_model_name=f"params:filter_model_name{n}", SNR=f"params:SNR{n}")
    
    nodes+=[
            
            node(
                func=initial_state_many,
                inputs=["tests","ekf1"],  # (data has the raw positions)
                outputs="x0",
                name="initial_state_node",
                tags=tags,
            ),     
        ]
    
    return pipeline(nodes)

def filter_pipeline(n:int, models:str, filter_model_name:str, SNR:str):
    
    return [
            node(
                func=create_kalman_filter,
                inputs=[models, filter_model_name, SNR],
                outputs=f"ekf{n}",
                name=f"create_kalman_filter_node{n}",
                tags=tags,
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
                tags=tags,
            ),
            
            node(
                func=results_to_dataframe,
                inputs=[
                    f"filtered_result{n}",
                ],
                outputs=f"tests_ek{n}",
                name=f"results_to_dataframe{n}",
                tags=tags,
            ),
            
            node(
                func=smoother,
                inputs=[
                    f"ekf{n}",
                    f"filtered_result{n}",
                ],
                outputs=f"smoother_result{n}",
                name=f"smoother{n}",
                tags=tags,
            ),
            
            node(
                func=results_to_dataframe,
                inputs=[
                    f"smoother_result{n}",
                ],
                outputs=f"tests_ek_smooth{n}",
                name=f"smoother_result_to_dataframe{n}",
                tags=tags,
            ),
            
            
            node(
                func=join_tests,
                inputs=[f"tests_ek{n}", "params:skip"],
                outputs=f"tests_ek_joined{n}",
                name=f"join_tests_ek{n}",
                tags=tags
            ),
            
            node(
                func=join_tests,
                inputs=[f"tests_ek_smooth{n}", "params:skip"],
                outputs=f"tests_ek_smooth_joined{n}",
                name=f"join_tests_ek_smooth{n}",
                tags=tags
            ),
        ]