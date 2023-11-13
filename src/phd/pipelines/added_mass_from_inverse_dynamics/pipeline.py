"""
This is a boilerplate pipeline 'added_mass_from_inverse_dynamics'
generated using Kedro 0.18.14
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import regress


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=regress,
                inputs=["tests_ek_smooth_joined", "models_VCT_MDL"],
                outputs="models_VCT_MDL_corrected_added_mass",
                name="corrected_added_mass",
            ),
        ]
    )
