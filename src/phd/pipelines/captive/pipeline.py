"""
This is a boilerplate pipeline 'captive'
generated using Kedro 0.18.14
"""

from kedro.pipeline import Pipeline, pipeline, node
from .nodes import load, prime, fit_and_correct


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=load,
                inputs=["captive_raw", "ship_data"],
                outputs="df_CMT",
                name="load_captive",
            ),
            node(
                func=prime,
                inputs=["df_CMT", "ship_data"],
                outputs="df_CMT_prime",
                name="prime_captive",
            ),
            node(
                func=fit_and_correct,
                inputs=["df_CMT_prime"],
                outputs="df_CMT_prime_corrected",
                name="fit_and_correct_captive",
            ),
        ]
    )
