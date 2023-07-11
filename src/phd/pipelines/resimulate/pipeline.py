"""
This is a boilerplate pipeline 'resimulate'
generated using Kedro 0.18.7
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import resimulate_all


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=resimulate_all,
                inputs=[
                    "wPCC.time_series_preprocessed.ek_smooth",
                    "wPCC.models.VCT_MDL_resistance",
                ],
                outputs="wPCC.MDL.VCT_MDL_resistance.resimulate",
                name="wPCC.MDL.VCT_MDL_resistance.resimulate_node",
                tags=["MDL", "wPCC", "VCT_MDL_resistance"],
            ),
            node(
                func=resimulate_all,
                inputs=[
                    "wPCC.time_series_preprocessed.ek_smooth",
                    "wPCC.models.VCT_MDL_resistance_optimized_kappa",
                ],
                outputs="wPCC.MDL.VCT_MDL_resistance_optimized_kappa.resimulate",
                name="wPCC.MDL.VCT_MDL_resistance_optimized_kappa.resimulate_node",
                tags=["MDL", "wPCC", "VCT_MDL_resistance_optimized_kappa"],
            ),
        ]
    )
