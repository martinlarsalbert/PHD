"""
This is a boilerplate pipeline 'models'
generated using Kedro 0.18.7
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    main_model,
    main_model_with_scale,
    vmm_7m_vct,
    vmm_martins_simple,
    vmm_martins_simple_thrust,
    regress_hull_VCT,
    correct_vct_resistance,
    optimize_kappa,
    regress_hull_inverse_dynamics,
    regress_inverse_dynamics,
    scale_model,
)


def create_pipeline(**kwargs) -> Pipeline:
    nodes = [
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
            func=vmm_martins_simple,
            inputs=["wPCC.main_model"],
            outputs="wPCC.vmm_martins_simple",
            name="vmm_martins_simple_node",
            tags="vmm_martins_simple",
        ),
        node(
            func=vmm_martins_simple_thrust,
            inputs=["wPCC.main_model"],
            outputs="wPCC.vmm_martins_simple_thrust",
            name="vmm_martins_simple_thrust_node",
            tags="vmm_martins_simple_thrust",
        ),
    ]

    ## VCT
    for vmm in ["vmm_7m_vct", "vmm_martins_simple"]:
        vmm_vct_nodes = [
            node(
                func=regress_hull_VCT,
                inputs=[f"wPCC.{vmm}", "wPCC.df_VCT"],
                outputs=f"wPCC.models.{vmm}.vct",
                name=f"{vmm}.regress_hull_VCT_node",
                tags="vct",
            ),
            node(
                func=correct_vct_resistance,
                inputs=[
                    f"wPCC.models.{vmm}.vct",
                    "wPCC.time_series_meta_data",
                    "wPCC.time_series_preprocessed.ek_smooth",
                ],
                outputs=f"wPCC.models.{vmm}.VCT_MDL_resistance",
                name=f"{vmm}.correct_vct_resistance_node",
                tags=["correct_resistance"],
            ),
            node(
                func=optimize_kappa,
                inputs=[
                    f"wPCC.models.{vmm}.VCT_MDL_resistance",
                    "wPCC.time_series_preprocessed.ek_smooth",
                ],
                outputs=f"wPCC.models.{vmm}.VCT_MDL_resistance_optimized_kappa",
                name=f"{vmm}.optimize_kappa_node",
                tags=["optimize_kappa"],
            ),
        ]
        nodes += vmm_vct_nodes

    ## Inverse hull dynamics
    for vmm in ["vmm_7m_vct", "vmm_martins_simple"]:
        vmm_inverse_hull_dynamics_nodes = [
            node(
                func=regress_hull_inverse_dynamics,
                inputs=[
                    f"wPCC.{vmm}",
                    "wPCC.time_series_preprocessed.ek_smooth",
                ],
                outputs=f"wPCC.models.{vmm}.MDL_hull_inverse_dynamics",
                name=f"{vmm}.regress_hull_inverse_dynamics_node",
                tags=["hull_inverse_dynamics"],
            ),
        ]
        nodes += vmm_inverse_hull_dynamics_nodes

    for vmm in ["vmm_martins_simple_thrust"]:
        vmm_inverse_dynamics_nodes = [
            node(
                func=regress_inverse_dynamics,
                inputs=[
                    f"wPCC.{vmm}",
                    "wPCC.time_series_preprocessed.ek_smooth",
                ],
                outputs=f"wPCC.models.{vmm}.MDL_inverse_dynamics",
                name=f"{vmm}.regress_inverse_dynamics_node",
                tags=["inverse_dynamics"],
            ),
        ]
        nodes += vmm_inverse_dynamics_nodes

    vmms = {
        "vmm_7m_vct": ["VCT_MDL_resistance", "MDL_hull_inverse_dynamics"],
        "vmm_martins_simple": ["VCT_MDL_resistance", "MDL_hull_inverse_dynamics"],
        "vmm_martins_simple_thrust": ["MDL_inverse_dynamics"],
    }

    ## Scale from 5m to 7m:
    for vmm, models in vmms.items():
        for model in models:
            scale_nodes = [
                node(
                    func=scale_model,
                    inputs=[
                        f"wPCC.models.{vmm}.{model}",
                        "7m.ship_data",
                    ],
                    outputs=f"7m.models.{vmm}.{model}",
                    name=f"{vmm}.{model}.scale_model_node",
                    tags=["scale_model"],
                ),
            ]
            nodes += scale_nodes

    return pipeline(nodes)
