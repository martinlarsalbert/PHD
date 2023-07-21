"""
This is a boilerplate pipeline 'resimulate_with_autopilot'
generated using Kedro 0.18.7
"""

"""
This is a boilerplate pipeline 'resimulate'
generated using Kedro 0.18.7
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import resimulate_all
from phd.pipelines.predict.nodes import select_prediction_dataset_7m


def create_pipeline(**kwargs) -> Pipeline:
    nodes = []

    ## Resimulate Field data:
    vmms = {
        "vmm_7m_vct": ["VCT_MDL_resistance", "MDL_hull_inverse_dynamics"],
        "vmm_7m_vct_wind": ["VCT_MDL_resistance", "MDL_hull_inverse_dynamics"],
        "vmm_martins_simple": ["VCT_MDL_resistance", "MDL_hull_inverse_dynamics"],
        "vmm_simple": ["VCT_MDL_resistance"],
        "vmm_simple_wind": ["VCT_MDL_resistance"],
    }
    data_source_name = "Lake"
    nodes.append(
        node(
            func=select_prediction_dataset_7m,
            inputs=[
                f"7m.tests_ek_smooth",
                f"7m.test_meta_data",
            ],
            outputs=f"7m.time_series_resimulation_with_autopilot",
            tags=["7m"],
            name=f"7m.select_resimulation_with_autopilot",
        )
    )
    for vmm, models in vmms.items():
        for model_name in models:
            new_nodes = [
                node(
                    func=resimulate_all,
                    inputs=[
                        f"7m.time_series_resimulation_with_autopilot",
                        f"7m.models.{vmm}.{model_name}",
                    ],
                    outputs=f"7m.{data_source_name}.{vmm}.{model_name}.resimulation_with_autopilot",
                    name=f"7m.{data_source_name}.{vmm}.{model_name}.resimulation_with_autopilot_node",
                    tags=[data_source_name, vmm, "7m", f"{model_name}"],
                )
            ]
            nodes += new_nodes

    return pipeline(nodes)
