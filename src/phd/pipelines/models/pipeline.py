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
    vmm_simple,
    regress_hull_VCT,
    regress_hull_rudder_VCT,
    correct_vct_resistance,
    optimize_kappa,
    scale_model,
    # regress_wind_tunnel_test,
    add_wind_force_system,
    base_models,
    base_models_simple,
)

# from .subsystems import add_wind_force_system


def create_pipeline(**kwargs) -> Pipeline:
    nodes = [
        node(
            func=base_models,
            inputs=["ship_data", "params:parameters"],
            outputs="base_models",
            name="base_models_node",
            tags=["generate_model"]
        ),
        node(
            func=base_models_simple,
            inputs=["ship_data", "params:parameters"],
            outputs="base_models_simple",
            name="base_models_simple",
            tags=["generate_model"]
        ),
    ]
    return pipeline(nodes)


# def create_pipeline(**kwargs) -> Pipeline:
#    nodes = [
#        node(
#            func=main_model,
#            inputs=[],
#            outputs="main_model",
#            name="main_model_node",
#        ),
#        node(
#            func=main_model_with_scale,
#            inputs=["main_model", "wPCC.ship_data"],
#            outputs="wPCC.main_model",
#            name="wPCC.main_model_node",
#        ),
#        node(
#            func=main_model_with_scale,
#            inputs=["main_model", "7m.ship_data"],
#            outputs="7m.main_model",
#            name="7m.main_model_node",
#        ),
#        node(
#            func=vmm_7m_vct,
#            inputs=["wPCC.main_model"],
#            outputs="wPCC.vmm_7m_vct",
#            name="vmm_7m_vct_node",
#            tags="vmm_7m_vct",
#        ),
#        node(
#            func=vmm_martins_simple,
#            inputs=["wPCC.main_model"],
#            outputs="wPCC.vmm_martins_simple",
#            name="vmm_martins_simple_node",
#            tags="vmm_martins_simple",
#        ),
#        node(
#            func=vmm_martins_simple_thrust,
#            inputs=["wPCC.main_model"],
#            outputs="wPCC.vmm_martins_simple_thrust",
#            name="vmm_martins_simple_thrust_node",
#            tags="vmm_martins_simple_thrust",
#        ),
#        node(
#            func=vmm_simple,
#            inputs=["wPCC.main_model"],
#            outputs="wPCC.vmm_simple",
#            name="vmm_simple_node",
#            tags="vmm_simple",
#        ),
#    ]
#
#    ## Add models with wind:
#    for vmm in ["vmm_7m_vct", "vmm_simple"]:
#        nodes += [
#            node(
#                func=add_wind_force_system,
#                inputs=[f"wPCC.{vmm}", "HMD_PCTC.wind_data"],
#                outputs=f"wPCC.{vmm}_wind",
#                name=f"{vmm}.add_wind_force_system",
#                tags="add_wind_force_system",
#            ),
#        ]
#
#    ## VCT hull
#    for vmm in [
#        "vmm_7m_vct",
#        "vmm_martins_simple",
#        "vmm_7m_vct_wind",
#    ]:
#        vmm_vct_hull_nodes = [
#            node(
#                func=regress_hull_VCT,
#                inputs=[f"wPCC.{vmm}", "wPCC.df_VCT"],
#                outputs=f"wPCC.models.{vmm}.vct",
#                name=f"{vmm}.regress_hull_VCT_node",
#                tags="vct",
#            ),
#            node(
#                func=correct_vct_resistance,
#                inputs=[
#                    f"wPCC.models.{vmm}.vct",
#                    "wPCC.time_series_meta_data",
#                    "wPCC.time_series_preprocessed.ek_smooth",
#                ],
#                outputs=f"wPCC.models.{vmm}.VCT_MDL_resistance",
#                name=f"{vmm}.correct_vct_resistance_node",
#                tags=["correct_resistance"],
#            ),
#            # node(
#            #    func=optimize_kappa,
#            #    inputs=[
#            #        f"wPCC.models.{vmm}.VCT_MDL_resistance",
#            #        "wPCC.time_series_preprocessed.ek_smooth",
#            #    ],
#            #    outputs=f"wPCC.models.{vmm}.VCT_MDL_resistance_optimized_kappa",
#            #    name=f"{vmm}.optimize_kappa_node",
#            #    tags=["optimize_kappa"],
#            # ),
#        ]
#        nodes += vmm_vct_hull_nodes
#
#    ## VCT hull rudder
#    for vmm in ["vmm_simple", "vmm_simple_wind"]:
#        vmm_vct_hull_rudder_nodes = [
#            node(
#                func=regress_hull_rudder_VCT,
#                inputs=[f"wPCC.{vmm}", "wPCC.df_VCT"],
#                outputs=f"wPCC.models.{vmm}.vct",
#                name=f"{vmm}.regress_hull_rudder_VCT",
#                tags="vct",
#            ),
#            node(
#                func=correct_vct_resistance,
#                inputs=[
#                    f"wPCC.models.{vmm}.vct",
#                    "wPCC.time_series_meta_data",
#                    "wPCC.time_series_preprocessed.ek_smooth",
#                ],
#                outputs=f"wPCC.models.{vmm}.VCT_MDL_resistance",
#                name=f"{vmm}.correct_vct_resistance_node",
#                tags=["correct_resistance"],
#            ),
#            # node(
#            #    func=optimize_kappa,
#            #    inputs=[
#            #        f"wPCC.models.{vmm}.VCT_MDL_resistance",
#            #        "wPCC.time_series_preprocessed.ek_smooth",
#            #    ],
#            #    outputs=f"wPCC.models.{vmm}.VCT_MDL_resistance_optimized_kappa",
#            #    name=f"{vmm}.optimize_kappa_node",
#            #    tags=["optimize_kappa"],
#            # ),
#        ]
#        nodes += vmm_vct_hull_rudder_nodes
#
#    ## Inverse hull dynamics
#    for vmm in ["vmm_7m_vct", "vmm_martins_simple", "vmm_7m_vct_wind"]:
#        vmm_inverse_hull_dynamics_nodes = [
#            node(
#                func=regress_hull_inverse_dynamics,
#                inputs=[
#                    f"wPCC.{vmm}",
#                    "wPCC.tests_ek_smooth_joined",
#                ],
#                outputs=f"wPCC.models.{vmm}.MDL_hull_inverse_dynamics",
#                name=f"{vmm}.regress_hull_inverse_dynamics_node",
#                tags=["hull_inverse_dynamics"],
#            ),
#        ]
#        nodes += vmm_inverse_hull_dynamics_nodes
#
#    ## Inverse dynamics (everything)
#    for vmm in ["vmm_martins_simple_thrust"]:
#        vmm_inverse_dynamics_nodes = [
#            node(
#                func=regress_inverse_dynamics,
#                inputs=[
#                    f"wPCC.{vmm}",
#                    "wPCC.tests_ek_smooth_joined",
#                ],
#                outputs=f"wPCC.models.{vmm}.MDL_inverse_dynamics",
#                name=f"{vmm}.regress_inverse_dynamics_node",
#                tags=["inverse_dynamics"],
#            ),
#        ]
#        nodes += vmm_inverse_dynamics_nodes
#
#    vmms = {
#        "vmm_7m_vct": ["VCT_MDL_resistance", "MDL_hull_inverse_dynamics"],
#        "vmm_7m_vct_wind": ["VCT_MDL_resistance", "MDL_hull_inverse_dynamics"],
#        "vmm_martins_simple": ["VCT_MDL_resistance", "MDL_hull_inverse_dynamics"],
#        "vmm_martins_simple_thrust": ["MDL_inverse_dynamics"],
#        "vmm_simple": ["VCT_MDL_resistance"],
#        "vmm_simple_wind": ["VCT_MDL_resistance"],
#    }
#
#    ## Scale from 5m to 7m:
#    for vmm, models in vmms.items():
#        for model in models:
#            scale_nodes = [
#                node(
#                    func=scale_model,
#                    inputs=[
#                        f"wPCC.models.{vmm}.{model}",
#                        "7m.ship_data",
#                    ],
#                    outputs=f"7m.models.{vmm}.{model}",
#                    name=f"{vmm}.{model}.scale_model_node",
#                    tags=["scale_model"],
#                ),
#            ]
#            nodes += scale_nodes
#
#    return pipeline(nodes)
