"""
This is a boilerplate pipeline 'load_7m'
generated using Kedro 0.18.7
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import load, divide_into_tests, scale_ship_data, remove_GPS_pattern


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=scale_ship_data,
                inputs=["wPCC.ship_data", "params:7m.scale_factor", "params:7m.rho"],
                outputs="7m.ship_data",
                name="7m.scale_ship_data",
            ),
            node(
                func=load,
                inputs=[
                    "7m.time_series_raw",
                    "params:7m.GPS_position",
                    "params:7m.accelerometer_position",
                    "7m.missions",
                    "params:7m.psi_correction",
                    "params:7m.cutting",
                ],
                # outputs=["7m.time_series_", "7m.time_series_meta_data", "7m.units"],
                outputs=["7m.time_series", "7m.time_series_meta_data", "7m.units"],
                name="7m.load",
            ),
            # node(
            #    func=remove_GPS_pattern,
            #    inputs=[
            #        "7m.time_series_",
            #        "params:7m.accelerometer_position",
            #    ],
            #    outputs="7m.time_series",
            #    name="7m.remove_GPS_pattern",
            # ),
            node(
                func=divide_into_tests,
                inputs=["7m.time_series", "7m.units"],
                outputs=["7m.tests", "7m.test_meta_data"],
            ),
        ]
    )
