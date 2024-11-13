"""
This is a boilerplate pipeline 'load_wPCC'
generated using Kedro 0.18.7
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import load, move_to_roll_centre, meta_data


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=load,
                inputs=[
                    "time_series",
                    "params:thrust_channels",
                    "params:rev_channels",
                ],
                outputs="tests_WL",
                name="load_node",
                tags=["load",],
            ),
            node(
                func=move_to_roll_centre,
                inputs=[
                    "tests_WL",
                    "params:WL_to_roll_centre",
                ],
                outputs="tests",
                name="move_to_roll_centre",
                tags=["load",],
            ),
            node(
                func=meta_data,
                inputs=["time_series_meta_data","tests"],
                outputs="test_meta_data",
                name="meta_data",
                tags=["load","load_meta_data"],
            ),
            
            # node(
            #    func=filter,
            #    inputs=[
            #        "data",
            #        "params:lowpass.cutoff",
            #        "params:lowpass.order",
            #    ],
            #    outputs="data_lowpass",
            #    name="lowpass_filter_node",
            #    tags=["filter"],
            # ),
        ]
    )


# from .nodes import move_to_lpp_half
# def create_pipeline(**kwargs) -> Pipeline:
#    return pipeline(
#        [
#            # node(
#            #    func=move_to_lpp_half,
#            #    inputs=["wPCC.time_series_preprocessed.ek_smooth_CG", "wPCC.ship_data"],
#            #    outputs="wPCC.time_series_preprocessed.ek_smooth",
#            # )
#        ]
#    )
