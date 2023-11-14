"""Project pipelines."""
from typing import Dict

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline
from kedro.pipeline.modular_pipeline import pipeline
from .pipelines import (
    # load_7m,
    # filter,
    filter_wPCC,
    load_wPCC,
    regression_VCT,
    regression_ID,
    models,
    resistance_MDL,
    added_mass_from_inverse_dynamics,
)


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    # pipelines = find_pipelines()
    pipelines = {}
    # pipelines["load_7m"] = load_7m.create_pipeline()
    pipelines["models"] = pipeline(models.create_pipeline(), namespace="wPCC")

    pipelines["regression_VCT"] = pipeline(
        regression_VCT.create_pipeline(), namespace="wPCC"
    )

    pipelines["added_mass_from_inverse_dynamics"] = pipeline(
        added_mass_from_inverse_dynamics.create_pipeline(), namespace="wPCC"
    )

    pipelines["regression_ID"] = pipeline(
        regression_ID.create_pipeline(), namespace="wPCC"
    )

    # pipelines["load_wPCC"] = pipeline(
    #    pipe=load_wPCC.create_pipeline(), namespace="wPCC"
    # )

    # pipelines["filter"] = pipeline(pipe=filter.create_pipeline(), namespace="7m")

    # pipelines["filter_wPCC"] = pipeline(
    #    pipe=filter_wPCC.create_pipeline(), namespace="wPCC"
    # )

    pipelines["resistance_MDL"] = pipeline(
        resistance_MDL.create_pipeline(), namespace="wPCC"
    )

    pipelines["__default__"] = sum(pipelines.values())
    return pipelines
