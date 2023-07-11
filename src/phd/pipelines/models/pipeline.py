"""
This is a boilerplate pipeline 'models'
generated using Kedro 0.18.7
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    main_model,
    main_model_with_scale,
    vmm_7m_vct,
    regress_hull_VCT,
    correct_vct_resistance,
    optimize_kappa,
    regress_hull_inverse_dynamics,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=main_model,
                inputs=[],
                outputs="main_model",
                name="main_model_node",
            ),
            node(
                func=main_model_with_scale,
                inputs=["main_model", "wPCC.ship_data"],
                outputs="wPCC.main_model",
                name="wPCC.main_model_node",
            ),
            node(
                func=main_model_with_scale,
                inputs=["main_model", "7m.ship_data"],
                outputs="7m.main_model",
                name="7m.main_model_node",
            ),
            node(
                func=vmm_7m_vct,
                inputs=["wPCC.main_model"],
                outputs="wPCC.vmm_7m_vct",
                name="vmm_7m_vct_node",
                tags="vmm_7m_vct",
            ),
            node(
                func=regress_hull_VCT,
                inputs=["wPCC.vmm_7m_vct", "wPCC.df_VCT"],
                outputs=["wPCC.models.vct", "wPCC.models.vct.model_summaries"],
                name="regress_hull_VCT_node",
                tags="vct",
            ),
            node(
                func=correct_vct_resistance,
                inputs=[
                    "wPCC.models.vct",
                    "wPCC.time_series_meta_data",
                    "wPCC.time_series_preprocessed.ek_smooth",
                ],
                outputs="wPCC.models.VCT_MDL_resistance",
                name="correct_vct_resistance_node",
                tags=["correct_resistance"],
            ),
            node(
                func=optimize_kappa,
                inputs=[
                    "wPCC.models.VCT_MDL_resistance",
                    "wPCC.time_series_preprocessed.ek_smooth",
                ],
                outputs="wPCC.models.VCT_MDL_resistance_optimized_kappa",
                name="optimize_kappa_node",
                tags=["optimize_kappa"],
            ),
            node(
                func=regress_hull_inverse_dynamics,
                inputs=[
                    "wPCC.vmm_7m_vct",
                    "wPCC.time_series_preprocessed.ek_smooth",
                ],
                outputs="wPCC.models.MDL_inverse_dynamics",
                name="regress_hull_inverse_dynamics_node",
                tags=["hull_inverse_dynamics"],
            ),
        ]
    )
